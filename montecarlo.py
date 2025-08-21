# montecarlo.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime, timezone

# =========
# Settings
# =========
CSV_PATH   = "BTCUSDTmergeddataset.csv"
USE_WEEKLY = False                 # default; can override with --weekly
IMG_DIR    = "images"
BURN_SIM   = 200                   # simulation burn-in (discarded)

TP_PCT = 0.08
ALPHA  = 5
FEE    = 0.0005
EPS    = 1e-12                     # numerical safety for divisions/logs

# Cap for stability in exp(cumsum(log_returns)): exp(±50) is already enormous
NUM_STAB_EXP_CAP = 50.0

# ------------------
# Helpers
# ------------------
def update_readme_section(section_name: str, new_content: str, readme_path="README.md"):
    """
    Update the README section between:
      <!--- {section_name}_START --->  ...  <!--- {section_name}_END --->
    If the tags are missing, append the tagged block to the end of README.
    """
    start_tag = f"<!--- {section_name}_START --->"
    end_tag   = f"<!--- {section_name}_END --->"
    block     = f"{start_tag}\n{new_content}\n{end_tag}\n"

    if not os.path.exists(readme_path):
        print(f"[README] {readme_path} not found. Creating a new file with the section.")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(block)
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()

    if start_tag in text and end_tag in text:
        pre  = text.split(start_tag)[0]
        post = text.split(end_tag)[1]
        new_text = pre + block + post
        where = "replaced existing"
    else:
        new_text = text.rstrip() + "\n\n" + block
        where = "appended new"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"[README] {where} {section_name} section in {readme_path}")


def bars_per_year(index: pd.DatetimeIndex) -> int:
    years = (index[-1] - index[0]).days / 365.25
    n = len(index)
    return int(round(n / years)) if years > 0 else 252


def infer_annualization(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 252
    f = pd.infer_freq(index)
    if f:
        f = f.upper()
        if f.startswith("W"): return 52
        if f == "B": return 252
        if f == "D": return 365
    mdays = np.median(np.diff(index.values).astype("timedelta64[D]").astype(float))
    if 4.0 <= mdays <= 9.0: return 52
    if 0.6 <= mdays <= 1.5:
        has_weekends = any(ts.weekday() >= 5 for ts in index)
        return 365 if has_weekends else 252
    return bars_per_year(index)


def ann_sharpe(r: pd.Series, k: int) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return (mu / sd) * np.sqrt(k)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()


# ====================================================
# Fit EGARCH to historical log returns (percent scale)
# ====================================================
def fit_egarch(train_returns_pct: pd.Series):
    eg = arch_model(train_returns_pct, vol="EGARCH", p=1, o=1, q=1, dist="t")
    res = eg.fit(disp="off")
    return eg, res


# ====================================================
# Simulate one path of (r_t, sigma_t) from fitted EGARCH
# Returns log-returns (raw scale) and conditional sigma_t
# ====================================================
def simulate_path(model, res, horizon: int, seed: int = None):
    """
    Uses arch's built-in simulate on the *model* with *res.params*.
    Returns (log_returns, sigma_t) in RAW (not percent) units.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(0, 2**32 - 1))  # arch uses NumPy's global RNG

    sim = model.simulate(
        res.params,
        nobs=horizon + BURN_SIM,
        initial_value=None,
        burn=BURN_SIM
    )

    cols = {c.lower(): c for c in sim.columns}

    # Returns are in percent if the model was fit on percent returns.
    if "data" in cols:
        r_pct = sim[cols["data"]]
    elif "y" in cols:
        r_pct = sim[cols["y"]]
    else:
        r_pct = sim.select_dtypes(include=[float, int]).iloc[:, 0]

    # Prefer variance; otherwise compute from volatility; otherwise heuristic.
    if "variance" in cols:
        var_pct = sim[cols["variance"]]
        sigma_pct = np.sqrt(var_pct)
    elif "volatility" in cols:
        sigma_pct = sim[cols["volatility"]]
    else:
        cand = [c for c in sim.columns if c.lower().startswith(("h", "var", "sigma"))]
        if not cand:
            raise KeyError(
                f"Could not find volatility/variance in simulate() output. Columns: {list(sim.columns)}"
            )
        s = sim[cand[0]]
        sigma_pct = np.sqrt(s) if s.abs().median() > 1.0 else s

    # Convert to raw units
    r_log = r_pct / 100.0
    sigma_t = sigma_pct / 100.0

    return r_log.reset_index(drop=True), sigma_t.reset_index(drop=True)


# ====================================================
# Strategy on a simulated path (no re-fitting inside)
# Uses simulated sigma_t as EGARCH vol and breach rule.
# ====================================================
def run_strategy_on_sim(path_log_returns: pd.Series,
                        sigma_t: pd.Series,
                        start_price: float,
                        k_annual: int):
    n = len(path_log_returns)

    # Synthetic price from start (clip cumulative log return to avoid overflow)
    cum_log = path_log_returns.cumsum().clip(-NUM_STAB_EXP_CAP, NUM_STAB_EXP_CAP)
    prices = pd.Series(start_price * np.exp(cum_log))

    # squared log-returns & "egarch variance" from the sim
    sqr = path_log_returns.pow(2)
    egarch_var = sigma_t.pow(2)

    # Trading state
    capital       = 1.0
    position      = 0
    buy_price     = 0.0
    pending_entry = False

    equity = np.full(n, np.nan)
    signal = np.zeros(n, dtype=int)

    for i in range(1, n):
        price = float(prices.iloc[i])

        # Execute pending next-bar entry
        if pending_entry and position == 0:
            position = 1
            buy_price = price
            capital *= (1 - FEE)
            signal[i] = 1
            pending_entry = False

        # Schedule entry on breach if flat
        if position == 0 and not pending_entry:
            if float(sqr.iloc[i]) > float(egarch_var.iloc[i]):
                pending_entry = True

        # Manage open position
        if position == 1:
            if not np.isfinite(buy_price) or buy_price <= EPS:
                position = 0
                signal[i] = -3  # invalid-state close
                equity[i] = capital
                continue

            ratio = price / max(buy_price, EPS)

            # Take-profit
            if ratio >= (1 + TP_PCT):
                position = 0
                capital *= ratio * (1 - FEE)
                signal[i] = -1
            else:
                # Vol-adjusted stop in log space
                log_ret_since_entry = np.log(max(ratio, EPS))
                stop_threshold = -ALPHA * max(float(sigma_t.iloc[i]), 0.0)
                if log_ret_since_entry <= stop_threshold:
                    position = 0
                    capital *= ratio * (1 - FEE)
                    signal[i] = -2

        # Mark-to-market equity
        equity[i] = capital if position == 0 else capital * (price / max(buy_price, EPS))

    # Results
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return {
            "final_equity": 1.0,
            "cagr": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "equity_curve": pd.Series([1.0])
        }

    total_equity = float(eq.iloc[-1])
    years = n / k_annual
    cagr = (total_equity ** (1 / years) - 1) if years > 0 else np.nan
    ret = eq.pct_change().dropna()
    sr  = ann_sharpe(ret, k_annual)
    mdd = max_drawdown_from_equity(eq)

    return {
        "final_equity": total_equity,
        "cagr": float(cagr),
        "sharpe": float(sr),
        "max_dd": float(mdd),
        "equity_curve": eq
    }


# ====================================================
# Main Monte Carlo
# ====================================================
def run_monte_carlo(n_paths: int,
                    horizon: int,
                    seed: int = 42,
                    weekly: bool = USE_WEEKLY,
                    start: str | None = None,
                    end: str | None = None):
    # Load & prep data
    df_raw = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], dayfirst=True)
    df_raw = df_raw.sort_values("timestamp").set_index("timestamp")
    df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

    # Optional date filtering (default: entire sample)
    if start:
        df_raw = df_raw.loc[df_raw.index >= pd.Timestamp(start)]
    if end:
        df_raw = df_raw.loc[df_raw.index <= pd.Timestamp(end)]

    if df_raw.empty:
        raise ValueError("No data after applying date filter. Check --start/--end.")

    df = df_raw.resample("W-MON").last().dropna() if weekly else df_raw.copy()

    # Returns AFTER choosing frequency
    df["log_return"] = np.log(df["close"]).diff()
    returns_pct = (df["log_return"] * 100.0).dropna()
    if returns_pct.empty:
        raise ValueError("Not enough returns after filtering/resampling to fit EGARCH.")

    # Fit once to historical
    model, res = fit_egarch(returns_pct)
    k = infer_annualization(df.index)
    start_price = float(df["close"].iloc[-1])

    os.makedirs(IMG_DIR, exist_ok=True)

    rng = np.random.default_rng(seed)
    results = []
    sample_curves = []

    for p in range(n_paths):
        r_log, sigma_t = simulate_path(model, res, horizon=horizon,
                                       seed=rng.integers(0, 2**32 - 1))
        out = run_strategy_on_sim(r_log, sigma_t, start_price=start_price, k_annual=k)
        results.append({
            "path": p,
            "final_equity": out["final_equity"],
            "cagr": out["cagr"],
            "sharpe": out["sharpe"],
            "max_dd": out["max_dd"],
        })
        if p < 10:
            sample_curves.append(out["equity_curve"].reset_index(drop=True))

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(IMG_DIR, "mc_results.csv"), index=False)

    # ---- Plots ----
    # 1) Distribution of final equity (sanitized)
    plt.figure(figsize=(10, 5))
    vals = (res_df["final_equity"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values)

    if vals.size:
        # optional: clip the craziest 0.1% to keep bins finite and readable
        vmax = np.quantile(vals, 0.999)
        vals = np.clip(vals, 0, vmax)

        plt.hist(vals, bins=40)
        plt.title(f"Monte Carlo — Final Equity Distribution (N={n_paths})")
        plt.xlabel("Final Equity (normalized)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "mc_final_equity_hist.png"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        print("[WARN] No valid final_equity values to plot histogram")

    # 2) Sample equity curves
    if sample_curves:
        plt.figure(figsize=(10, 5))
        for eq in sample_curves:
            plt.plot(eq.values, alpha=0.6)
        plt.title("Monte Carlo — Sample Equity Curves")
        plt.xlabel("Bars")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "mc_sample_equity_curves.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # Summary print (means & medians)
    summary = res_df.agg({
        "final_equity": ["mean", "median", "std", "min", "max"],
        "cagr": ["mean", "median"],
        "sharpe": ["mean", "median"],
        "max_dd": ["mean", "median"]
    })
    print("\n=== Monte Carlo Summary ===")
    print(summary.round(4).to_string())

    # Robust quantiles (less sensitive to outliers)
    q = res_df.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    print("\n=== Monte Carlo Summary (robust) ===")
    print("Final equity  p01={:.2f}  p05={:.2f}  median={:.2f}  p95={:.2f}  p99={:.2f}"
          .format(*q["final_equity"].values))
    print("CAGR          p01={:.2%}  p05={:.2%}  median={:.2%}  p95={:.2%}  p99={:.2%}"
          .format(*q["cagr"].values))
    print("Sharpe        p05={:.2f}  median={:.2f}  p95={:.2f}"
          .format(q.loc[0.05, "sharpe"], q.loc[0.5, "sharpe"], q.loc[0.95, "sharpe"]))
    print("Max DD        p05={:.2%}  median={:.2%}  p95={:.2%}"
          .format(q.loc[0.05, "max_dd"], q.loc[0.5, "max_dd"], q.loc[0.95, "max_dd"]))

    # Tail stats
    ruin_prob = (res_df["final_equity"] < 0.5).mean()     # <50% capital
    underperf_prob = (res_df["cagr"] < 0.0).mean()
    print(f"\nP(final_equity < 0.5) = {ruin_prob:.3f}")
    print(f"P(CAGR < 0) = {underperf_prob:.3f}")

    return res_df


def parse_args():
    ap = argparse.ArgumentParser(description="Monte Carlo for EGARCH breach strategy")
    ap.add_argument("--paths", type=int, default=1000, help="Number of simulated paths")
    ap.add_argument("--horizon", type=int, default=1000, help="Bars to simulate per path")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--weekly", action="store_true", help="Use weekly bars (W-MON)")
    ap.add_argument("--start", type=str, default=None, help="Start date (inclusive), e.g. 2017-01-01")
    ap.add_argument("--end", type=str, default=None, help="End date (inclusive), e.g. 2024-12-31")
    return ap.parse_args()


if __name__ == "__main__":
    print(f"[CWD] {os.getcwd()}")
    args = parse_args()
    res_df = run_monte_carlo(
        n_paths=args.paths,
        horizon=args.horizon,
        seed=args.seed,
        weekly=args.weekly,
        start=args.start,   # None ⇒ whole sample
        end=args.end
    )

    # Build fresh Markdown for README (with timestamp)
    q  = res_df.quantile([0.05, 0.5, 0.95])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    q = res_df.quantile([0.01, 0.05, 0.5, 0.95, 0.99])

    summary_md = f"""## 🎲 Monte Carlo Stress Test (auto-updated)
_Last refreshed: **{ts}**_

### Results (N={len(res_df)})
- Median Final Equity: {res_df['final_equity'].median():.2f}×
- Median CAGR: {res_df['cagr'].median():.2%}
- Median Sharpe: {res_df['sharpe'].median():.2f}
- Median Max DD: {res_df['max_dd'].median():.2%}

**Quantiles**
- Final Equity — p01: {q.loc[0.01,'final_equity']:.2f}× · p05: {q.loc[0.05,'final_equity']:.2f}× · median: {q.loc[0.5,'final_equity']:.2f}× · p95: {q.loc[0.95,'final_equity']:.2f}× · p99: {q.loc[0.99,'final_equity']:.2f}×
- CAGR — p01: {q.loc[0.01,'cagr']:.2%} · p05: {q.loc[0.05,'cagr']:.2%} · median: {q.loc[0.5,'cagr']:.2%} · p95: {q.loc[0.95,'cagr']:.2%} · p99: {q.loc[0.99,'cagr']:.2%}
- Sharpe — p05: {q.loc[0.05,'sharpe']:.2f} · median: {q.loc[0.5,'sharpe']:.2f} · p95: {q.loc[0.95,'sharpe']:.2f}
- Max DD — p05: {q.loc[0.05,'max_dd']:.2%} · median: {q.loc[0.5,'max_dd']:.2%} · p95: {q.loc[0.95,'max_dd']:.2%}

**Tail probabilities**
- P(final_equity < 0.5): {(res_df['final_equity'] < 0.5).mean():.3f}
- P(CAGR < 0): {(res_df['cagr'] < 0).mean():.3f}

**Figures**
- ![Sample equity curves](images/mc_sample_equity_curves.png)
"""
