import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm

# =========================
# === STRATEGY PARAMS   ===
# =========================
TP_PCT        = 0.08     # 8% take-profit
ALPHA         = 5.2      # multiplier for volatility-adjusted stop (log space)
FEE           = 0.0005   # 5 bps per side (spot-like). Tune per venue.
RECALC_EVERY  = 30       # EGARCH re-fit cadence
USE_WEEKLY    = False    # set True to run on weekly bars (W-MON)

# =========================
# === HELPERS           ===
# =========================
def bars_per_year(index: pd.DatetimeIndex) -> int:
    years = (index[-1] - index[0]).days / 365.25
    n = len(index)
    return int(round(n / years)) if years > 0 else 252

def ann_sharpe(r: pd.Series, k: int) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    return (mu / sd) * np.sqrt(k) if sd > 0 else np.nan

def infer_annualization(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 252
    f = pd.infer_freq(index)
    if f:
        f = f.upper()
        if f.startswith("W"):
            return 52
        if f in ("D", "B"):
            return 252
    mdays = np.median(np.diff(index.values).astype("timedelta64[D]").astype(float))
    if 4.0 <= mdays <= 9.0:
        return 52
    if 0.6 <= mdays <= 1.5:
        return 252
    return bars_per_year(index)

def annualized_sharpe_from_returns(r: np.ndarray, freq: int) -> float:
    r = np.asarray(r)
    if r.size == 0:
        return np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    return (mu / sd) * np.sqrt(freq) if sd > 0 else np.nan

def block_bootstrap_sharpe_diff(r1: pd.Series, r2: pd.Series, freq: int,
                                B: int = 5000, block: int = 10, seed: int = 42):
    """
    Paired circular block bootstrap of ΔSharpe = Sharpe(r1) - Sharpe(r2).
    Returns (d_obs, (ci_low, ci_high), p_two_sided, p_one_sided_pos).
    """
    rng = np.random.default_rng(seed)
    rr = pd.concat([r1, r2], axis=1, join="inner").dropna()
    if rr.shape[0] < block + 2:
        return np.nan, (np.nan, np.nan), np.nan, np.nan

    x = rr.iloc[:, 0].to_numpy()
    y = rr.iloc[:, 1].to_numpy()
    T = len(rr)

    d_obs = annualized_sharpe_from_returns(x, freq) - annualized_sharpe_from_returns(y, freq)

    idx = np.arange(T)
    nblocks = int(np.ceil(T / block))
    diffs = np.empty(B)

    for b in range(B):
        starts = rng.integers(0, T, size=nblocks)
        boot_idx = np.concatenate([
            (idx[s:(s + block)] if s + block <= T else np.r_[idx[s:], idx[:(s + block) % T]])
            for s in starts
        ])[:T]
        xb = x[boot_idx]
        yb = y[boot_idx]
        diffs[b] = (annualized_sharpe_from_returns(xb, freq)
                    - annualized_sharpe_from_returns(yb, freq))

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    diffs0 = diffs - diffs.mean()  # center under H0: Δ=0
    p_two = (np.sum(np.abs(diffs0) >= np.abs(d_obs)) + 1) / (B + 1)
    p_one_pos = (np.sum(diffs0 >= d_obs) + 1) / (B + 1)
    return d_obs, (ci_low, ci_high), p_two, p_one_pos

def effective_sample_size(r: pd.Series, max_lag: int = 20) -> int:
    r = r.dropna().values
    if r.size < 3:
        return r.size
    r = (r - r.mean()) / r.std(ddof=1)
    N = len(r)
    acf = []
    for k in range(1, min(max_lag, N - 1)):
        num = np.corrcoef(r[:-k], r[k:])[0, 1]
        acf.append(num)
    ess = N / (1 + 2 * sum(acf_k for acf_k in acf if not np.isnan(acf_k)))
    return max(5, int(ess))

# =========================
# === LOAD & PREP DATA  ===
# =========================
df_raw = pd.read_csv("BTCUSDTmergeddataset.csv", parse_dates=["timestamp"], dayfirst=True)
df_raw = df_raw.sort_values("timestamp").set_index("timestamp")
df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

# ---- choose bar frequency here ----
if USE_WEEKLY:
    # weekly bars (last close of each W-MON bucket)
    df = df_raw.resample("W-MON").last().dropna()
    ANNUALIZE = 52
else:
    # daily bars
    df = df_raw.copy()
    ANNUALIZE = 252

# compute returns AFTER choosing frequency
df["log_return"]       = np.log(df["close"]).diff()
df["squared_returns"]  = df["log_return"] ** 2

# ======================================
# === CHECK FOR MISSING WEEKS (period) ==
# ======================================
full_weeks   = pd.period_range(df.index.min(), df.index.max(), freq="W-MON")
have_weeks   = df.index.to_period("W-MON")
missing_weeks = sorted(set(full_weeks) - set(have_weeks))
print("Number of missing weeks:", len(missing_weeks))

# ==================================
# === EGARCH(1,1) ESTIMATION     ===
# === (out-of-sample forecasts)  ===
# ==================================
returns_pct = df["log_return"] * 100.0  # arch uses percent scale

pred_var = pd.Series(index=df.index, dtype=float)
burn_in  = 400
res_i    = None
valid_idx = returns_pct.dropna().index

for j in range(burn_in, len(valid_idx)):
    # re-fit every RECALC_EVERY steps using only data up to j-1
    if (j - burn_in) % RECALC_EVERY == 0 or res_i is None:
        train_series = returns_pct.loc[valid_idx[:j]]
        eg = arch_model(train_series, vol="EGARCH", p=1, o=1, q=1, dist="t")
        res_i = eg.fit(disp="off")

    # one-step-ahead variance forecast for time valid_idx[j]
    f_var_pct2 = res_i.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    pred_var.loc[valid_idx[j]] = f_var_pct2 / (100.0 ** 2)  # back to raw scale

# keep only rows where we have a forecast
df = df.loc[pred_var.dropna().index].copy()
df["egarch_variance"] = pred_var.loc[df.index]
df["egarch_vol"]      = np.sqrt(df["egarch_variance"])      # σ_t
df["breach"]          = (df["squared_returns"] > df["egarch_variance"]).astype(int)
breaches              = df[df["breach"] == 1]

# ==========================================
# === STRATEGY IMPLEMENTATION (next-bar buy)
# ==========================================
capital        = 1.0
position       = 0
buy_price      = 0.0
pending_entry  = False

df["signal"] = 0
df["equity"] = np.nan
idx = df.index

for i in range(1, len(df)):
    price = df.loc[idx[i], "close"]

    # execute pending entry at next bar (with fee)
    if pending_entry and position == 0:
        position = 1
        buy_price = price
        capital *= (1 - FEE)         # pay entry fee
        df.loc[idx[i], "signal"] = 1
        pending_entry = False

    # schedule entry if breach occurs and we're flat
    if position == 0 and not pending_entry:
        if df.loc[idx[i], "squared_returns"] > df.loc[idx[i], "egarch_variance"]:
            pending_entry = True

    if position == 1:
        # --- Take-profit exit ---
        if price >= buy_price * (1 + TP_PCT):
            position = 0
            capital *= (price / buy_price) * (1 - FEE)   # exit with fee
            df.loc[idx[i], "signal"] = -1
        else:
            # --- Vol-adjusted stop in LOG space ---
            sigma_t = df.loc[idx[i], "egarch_vol"]      # std of log returns
            log_ret_since_entry = np.log(price / buy_price)
            stop_threshold = -ALPHA * sigma_t           # stop if worse than -α·σ
            if log_ret_since_entry <= stop_threshold:
                position = 0
                capital *= (price / buy_price) * (1 - FEE)
                df.loc[idx[i], "signal"] = -2

    # mark equity (MTM)
    df.loc[idx[i], "equity"] = capital if position == 0 else capital * (price / buy_price)

# after the loop, drop rows without equity to avoid NaNs leaking into metrics
df = df.dropna(subset=["equity"])

# entries & exits (unified)
entries = df[df["signal"] == 1]
exits   = df[df["signal"].isin([-1, -2, -3])]

# ==========================================
# === BUY & HOLD EQUITY CURVE + METRICS  ===
# ==========================================
if df.empty:
    raise ValueError("DataFrame is empty after EGARCH filtering. No data to backtest.")

buyhold_initial      = df["close"].iloc[0]
df["buyhold_equity"] = df["close"] / buyhold_initial

bh_final   = df["buyhold_equity"].iloc[-1]
total_days = (df.index[-1] - df.index[0]).days
years      = total_days / 365.25 if total_days > 0 else 0.0
bh_cagr    = (bh_final ** (1 / years) - 1) if years > 0 else np.nan

k          = infer_annualization(df.index)  # more robust than hard-coding
ret_strat  = df["equity"].pct_change().dropna()
ret_bh     = df["buyhold_equity"].pct_change().dropna()

sharpe_ratio = ann_sharpe(ret_strat, k)
bh_sharpe    = ann_sharpe(ret_bh,    k)

bh_rolling   = df["buyhold_equity"].cummax()
bh_drawdown  = (df["buyhold_equity"] - bh_rolling) / bh_rolling
bh_maxdd     = bh_drawdown.min()

# ===========================
# === STRATEGY METRICS    ===
# ===========================
trades = []
position = 0
entry_price = None
entry_date  = None

for i in range(1, len(df)):
    sig = df["signal"].iloc[i]
    if sig == 1:
        entry_date  = df.index[i]
        entry_price = df["close"].iloc[i]
        position    = 1
    elif position == 1 and sig in (-1, -2, -3):
        exit_date  = df.index[i]
        exit_price = df["close"].iloc[i]
        outcome    = { -1: "take_profit", -2: "stop_loss", -3: "forced_close"}[sig]
        pnl        = (exit_price - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "outcome": outcome
        })
        position = 0
        entry_price = None
        entry_date  = None

# force-close any open trade at the final bar for honest stats
if position == 1 and entry_price is not None:
    final_price = df["close"].iloc[-1]
    trades.append({
        "entry_date": entry_date,
        "exit_date" : df.index[-1],
        "entry_price": entry_price,
        "exit_price" : final_price,
        "pnl"        : (final_price - entry_price) / entry_price,
        "outcome"    : "forced_close"
    })
    df.iloc[-1, df.columns.get_loc("signal")] = -3

trades_df   = pd.DataFrame(trades)
total_trades = len(trades_df)
win_rate     = (trades_df["pnl"] > 0).mean() if total_trades > 0 else np.nan
avg_pnl      = trades_df["pnl"].mean() if total_trades > 0 else np.nan

total_return = df["equity"].iloc[-1] - 1
CAGR         = (df["equity"].iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

roll_max      = df["equity"].cummax()
drawdown      = (df["equity"] - roll_max) / roll_max
max_drawdown  = drawdown.min()

# =======================================
# === CREATE /images/ AND SAVE PLOTS  ===
# =======================================
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# 1) Trade chart
plt.figure(figsize=(16, 8))
ax1 = plt.gca()
ax1.plot(df.index, df["close"], label="BTC Close", color="blue", alpha=0.6, linewidth=1.8)

breaches_plot = breaches.loc[breaches.index.intersection(df.index)]
ax1.scatter(breaches_plot.index, breaches_plot["close"], color="orange", marker="o", s=50,
            alpha=0.6, label="Variance Breach")

ax1.scatter(entries.index, entries["close"], color="green", marker="^", s=90,
            edgecolor="black", linewidth=0.6, label="Buy Entry")

ax1.scatter(exits.index, exits["close"], color="red", marker="v", s=90,
            edgecolor="black", linewidth=0.6, label="Exit")

ax1.set_ylabel("BTC Close Price", fontsize=12)
plt.title("BTC: EGARCH Variance Breach Strategy — Signals and Trades", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_trades.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2) Equity curve
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["equity"], label="EGARCH Strategy", color="purple", linewidth=2)
plt.plot(df.index, df["buyhold_equity"], label="Buy & Hold", color="black", linestyle="--", linewidth=2)
plt.yscale("log")  # optional for clarity across years
plt.title("Equity Curve (Log Scale)", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (Normalized)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_equity.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3) Drawdowns
plt.figure(figsize=(14, 4))
plt.plot(drawdown.index, drawdown, color="red")
plt.title("EGARCH Strategy — Drawdowns")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_drawdown.png"), dpi=300, bbox_inches="tight")
plt.close()

# 4) Returns vs EGARCH Vol
egarch_vol_plot = (df["egarch_vol"] * 100.0)  # σ in %
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["log_return"] * 100.0, color="grey", alpha=0.5, label="Log Returns × 100")
plt.plot(df.index, egarch_vol_plot, color="blue", linewidth=2, label="EGARCH Volatility (%)")
plt.title("BTC Log Returns and EGARCH Model Volatility")
plt.xlabel("Date")
plt.ylabel("Log Return / Volatility (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_volatility.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved plots to:", IMG_DIR)

# ============================================
# === STRATEGY vs BUY & HOLD — METRICS TABLE
# ============================================
comparison = pd.DataFrame({
    "EGARCH Strategy": {
        "Total Return": total_return,
        "CAGR": CAGR,
        f"Sharpe Ratio (annualize={k})": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Trades": len(trades_df),
        "Win Rate": win_rate,
        "Avg Trade PnL": avg_pnl,
    },
    "Buy & Hold": {
        "Total Return": bh_final - 1,
        "CAGR": bh_cagr,
        f"Sharpe Ratio (annualize={k})": bh_sharpe,
        "Max Drawdown": bh_maxdd,
        "Trades": np.nan,
        "Win Rate": np.nan,
        "Avg Trade PnL": np.nan,
    }
})

# ===== Save CSV
comparison.to_csv(os.path.join(IMG_DIR, "strategy_vs_buyhold.csv"))

# ===== Nicely formatted version for README/MD
percent_rows = {"Total Return", "CAGR", "Max Drawdown", "Win Rate", "Avg Trade PnL"}

def _fmt_cell(idx, val):
    if pd.isna(val):
        return "—"
    if idx in percent_rows:
        return f"{val*100:.2f}%"
    if idx == "Trades":
        return f"{int(val)}"
    if "Sharpe" in idx:
        return f"{val:.4f}"
    return f"{val:.4f}"

formatted = comparison.copy()
for row in formatted.index:
    formatted.loc[row, :] = [_fmt_cell(row, v) for v in formatted.loc[row, :].values]

print("\n=== Strategy vs Buy & Hold ===")
print(formatted.to_markdown())

# ===== Save Markdown fragment
md_path = os.path.join(IMG_DIR, "strategy_vs_buyhold.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("## 📈 Strategy vs Buy & Hold Metrics\n\n")
    f.write(formatted.to_markdown())

print(f"Saved Markdown metrics table to {md_path}")

# ===== Inject the Markdown fragment into README between tags (if present)
readme_path = "README.md"
start_tag = "<!--- METRICS_TABLE_START --->"
end_tag   = "<!--- METRICS_TABLE_END --->"

try:
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()
        if start_tag in readme and end_tag in readme:
            pre  = readme.split(start_tag)[0]
            post = readme.split(end_tag)[1]
            new_readme = pre + start_tag + "\n" + open(md_path, "r", encoding="utf-8").read() + "\n" + end_tag + post
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_readme)
            print(f"Updated {readme_path} with metrics table.")
        else:
            print("README placeholders not found; skipped injection.")
    else:
        print("README.md not found; skipped injection.")
except Exception as e:
    print("README injection error:", e)

# =========================
# === Sanity checks     ===
# =========================
# 1) Next-bar entry check (cast to bool before ~)
prev_breach = df["breach"].shift(1).fillna(False).astype(bool)
violations = df.loc[(df["signal"].eq(1)) & (~prev_breach)]
print("Next-bar entry violations:", len(violations))

# 2) Same-bar entry & exit (rare but possible with forced_close)
samebar = trades_df.loc[trades_df["entry_date"] == trades_df["exit_date"]]
print("Same-bar entry/exit count:", len(samebar))

# 3) Only flag violations when you were flat at t-1
pos = np.zeros(len(df), dtype=int)
in_pos = 0
for i in range(len(df)):
    sig = df["signal"].iloc[i]
    if sig == 1 and in_pos == 0:
        in_pos = 1
    elif sig in (-1, -2, -3) and in_pos == 1:
        in_pos = 0
    pos[i] = in_pos
df["pos"] = pos

violations_flat = df.loc[(df["signal"].eq(1)) & (~prev_breach) & (df["pos"].shift(1).fillna(0).eq(0))]
print("Next-bar entry violations (only when flat at t-1):", len(violations_flat))

# ================================
# === Sharpe tests (optional)  ===
# ================================
print("\nBootstrap ΔSharpe (Strategy minus B&H)")
for label, r1, r2 in [
    ("Full Sample", df["equity"].pct_change(), df["buyhold_equity"].pct_change()),
    ("Pre-2022",    df.loc[df.index <  "2022-01-01", "equity"].pct_change(),
                    df.loc[df.index <  "2022-01-01", "buyhold_equity"].pct_change()),
    ("Post-2022",   df.loc[df.index >= "2022-01-01", "equity"].pct_change(),
                    df.loc[df.index >= "2022-01-01", "buyhold_equity"].pct_change()),
]:
    d, ci, p2, p1 = block_bootstrap_sharpe_diff(r1, r2, freq=infer_annualization(df.index), B=2000, block=10)
    print(f"{label}: ΔSharpe={d:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], p_two={p2:.4f}, p_one(Δ>0)={p1:.4f}")

# Effective sample sizes
ess_strat = effective_sample_size(df["equity"].pct_change())
ess_bh    = effective_sample_size(df["buyhold_equity"].pct_change())
print("Effective N (strat, bh):", ess_strat, ess_bh)

# Trade distribution quick stats
if not trades_df.empty:
    pnl = trades_df["pnl"].dropna().values
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = (wins.sum() / -losses.sum()) if losses.size else np.inf
    print("Profit Factor:", profit_factor, "Mean trade PnL:", pnl.mean(), "Median:", np.median(pnl))

# ============================================
# === ROBUSTNESS & VALIDATION (tables + README)
# ============================================
import io

def _inject_between_tags(readme_path: str, start_tag: str, end_tag: str, md_block: str):
    """
    Non-destructive: replace only the FIRST start/end tag pair.
    If not found, append the tagged block to the end of README.
    """
    tagged = f"{start_tag}\n{md_block}\n{end_tag}"
    if not os.path.exists(readme_path):
        print("README.md not found; skipped injection.")
        return
    with open(readme_path, "r", encoding="utf-8") as f:
        s = f.read()

    s_start = s.find(start_tag)
    if s_start == -1:
        # Append at end
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + tagged + "\n")
        print(f"Appended section {start_tag}…{end_tag} to README.md")
        return

    s_end = s.find(end_tag, s_start + len(start_tag))
    if s_end == -1:
        # No matching end tag; append safely
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + tagged + "\n")
        print(f"End tag {end_tag} not found; appended block to README.md")
        return

    new_s = s[:s_start] + tagged + s[s_end + len(end_tag):]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_s)
    print(f"Updated README.md section between {start_tag} and {end_tag}")

# ---------- Build ROBUSTNESS table ----------
# Count forced closes
forced_closes = int((df["signal"] == -3).sum())

def _yn(x): return "Yes" if bool(x) else "No"

robust_rows = [
    {"Check": "Transaction costs modeled",
     "Result": f"{FEE*1e4:.1f} bps per side",
     "Notes": "Applied on entries and exits."},
    {"Check": "Execution lag",
     "Result": "Next-bar",
     "Notes": "Breach detected at t, enter at t+1 if flat."},
    {"Check": "Walk-forward re-fitting",
     "Result": f"EGARCH re-fit every {RECALC_EVERY} bars",
     "Notes": f"Burn-in={burn_in}; Student-t innovations."},
    {"Check": "Annualization inference",
     "Result": f"k={k}",
     "Notes": "Detected from index frequency/fallback."},
    {"Check": "Missing weeks (if weekly)",
     "Result": f"{len(missing_weeks)}",
     "Notes": "Data integrity check on W-MON grid."},
    {"Check": "Next-bar entry violations",
     "Result": f"{int(len(violations))}",
     "Notes": "Any entry without prior-bar breach."},
    {"Check": "Next-bar entry violations (flat at t-1 only)",
     "Result": f"{int(len(violations_flat))}",
     "Notes": "Stricter condition; should be ~0."},
    {"Check": "Same-bar entry/exit trades",
     "Result": f"{int(len(samebar))}",
     "Notes": "Typically due to forced close."},
    {"Check": "Forced closes at final bar",
     "Result": f"{forced_closes}",
     "Notes": "Honest stats when position open at end."},
    {"Check": "Effective sample size (returns)",
     "Result": f"Strat={ess_strat}, B&H={ess_bh}",
     "Notes": "Accounts for autocorrelation."},
    {"Check": "Data frequency",
     "Result": "Weekly (W-MON)" if USE_WEEKLY else "Daily",
     "Notes": "Controlled via USE_WEEKLY."},
]
robust_df = pd.DataFrame(robust_rows, columns=["Check", "Result", "Notes"])

# Capture bootstrap ΔSharpe results into a small table (re-using your loop settings)
sharpe_tests = []
for label, r1, r2 in [
    ("Full Sample", df["equity"].pct_change(), df["buyhold_equity"].pct_change()),
    ("Pre-2022",    df.loc[df.index <  "2022-01-01", "equity"].pct_change(),
                    df.loc[df.index <  "2022-01-01", "buyhold_equity"].pct_change()),
    ("Post-2022",   df.loc[df.index >= "2022-01-01", "equity"].pct_change(),
                    df.loc[df.index >= "2022-01-01", "buyhold_equity"].pct_change()),
]:
    d, ci, p2, p1 = block_bootstrap_sharpe_diff(r1, r2, freq=infer_annualization(df.index), B=2000, block=10)
    sharpe_tests.append({
        "Regime": label,
        "Delta_Sharpe": d,
        "CI_Low": (ci[0] if isinstance(ci, tuple) or isinstance(ci, list) else np.nan),
        "CI_High": (ci[1] if isinstance(ci, tuple) or isinstance(ci, list) else np.nan),
        "p_two_sided": p2,
        "p_one_sided_pos": p1
    })
sharpe_df = pd.DataFrame(sharpe_tests, columns=["Regime", "Delta_Sharpe", "CI_Low", "CI_High", "p_two_sided", "p_one_sided_pos"])

def _fmt3(x):
    try:
        return f"{x:.3f}"
    except Exception:
        return "—"

for c in ["Delta_Sharpe", "CI_Low", "CI_High", "p_two_sided", "p_one_sided_pos"]:
    if c in sharpe_df.columns:
        sharpe_df[c] = sharpe_df[c].apply(_fmt3)

# ---------- Write Markdown fragments (metrics already saved earlier) ----------
# Metrics fragment: read what you wrote earlier to images/strategy_vs_buyhold.md
metrics_md_path = os.path.join(IMG_DIR, "strategy_vs_buyhold.md")
if os.path.exists(metrics_md_path):
    with open(metrics_md_path, "r", encoding="utf-8") as f:
        metrics_md_block = f.read()
else:
    # Fallback: rebuild a minimal block on the fly
    metrics_md_block = "## 📈 Strategy vs Buy & Hold Metrics\n\n" + formatted.to_markdown()

# Robustness fragment
robust_md_path = os.path.join(IMG_DIR, "robustness_checks.md")
robust_buf = io.StringIO()
robust_buf.write("## 🛡️ Robustness & Validation Checks\n\n")
robust_buf.write(robust_df.to_markdown(index=False))
robust_buf.write("\n\n### Bootstrap ΔSharpe (Strategy − Buy & Hold)\n\n")
robust_buf.write(sharpe_df.to_markdown(index=False))
robust_md_block = robust_buf.getvalue()
with open(robust_md_path, "w", encoding="utf-8") as f:
    f.write(robust_md_block)
print(f"Saved Robustness markdown to {robust_md_path}")

# ---------- Inject into README (first matching pair only) ----------
readme_path = "README.md"
METRICS_START = "<!--- METRICS_TABLE_START --->"
METRICS_END   = "<!--- METRICS_TABLE_END --->"
ROBUST_START  = "<!--- ROBUSTNESS_TABLE_START --->"
ROBUST_END    = "<!--- ROBUSTNESS_TABLE_END --->"

# Ensure the blocks include headings (so they render nicely in README)
_inject_between_tags(readme_path, METRICS_START, METRICS_END, metrics_md_block)
_inject_between_tags(readme_path, ROBUST_START, ROBUST_END, robust_md_block)
   