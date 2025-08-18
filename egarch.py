import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
# =========================
# === STRATEGY PARAMS   ===
# =========================
TP_PCT = 0.08   # 8% take-profit
ALPHA = 5.2   # multiplier for volatility-adjusted stop

FEE = 0.0005          # 5 bps per side (spot-like). Tune per venue.
RECALC_EVERY = 30     # keep your existing value, expose it here

def bars_per_year(index):
    years = (index[-1] - index[0]).days / 365.25
    n = len(index)
    return int(round(n / years)) if years > 0 else 252

# =========================
# === LOAD & PREP DATA  ===
# =========================
df_raw = pd.read_csv("BTCUSDTmergeddataset.csv", parse_dates=['timestamp'], dayfirst=True)
df_raw = df_raw.sort_values('timestamp').set_index('timestamp')
df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

# ---- choose bar frequency here ----
USE_WEEKLY = False   # set True to run weekly bars
if USE_WEEKLY:
    # weekly bars (last close of each W-MON bucket)
    df = df_raw.resample('W-MON').last().dropna()
    ANNUALIZE = 52
else:
    # daily bars
    df = df_raw.copy()
    ANNUALIZE = 252

# compute returns AFTER choosing frequency
df['log_return'] = np.log(df['close']).diff()
df['squared_returns'] = df['log_return'] ** 2


# ======================================
# === CHECK FOR MISSING WEEKS (period) ==
# ======================================
full_weeks = pd.period_range(df.index.min(), df.index.max(), freq='W-MON')
have_weeks = df.index.to_period('W-MON')
missing_weeks = sorted(set(full_weeks) - set(have_weeks))
print("Number of missing weeks:", len(missing_weeks))

# ==================================
# === EGARCH(1,1) ESTIMATION     ===
# === (out-of-sample forecasts)  ===
# ==================================
returns_pct = df['log_return'] * 100.0  # same scale as before

pred_var = pd.Series(index=df.index, dtype=float)

burn_in = 400
res_i = None

valid_idx = returns_pct.dropna().index
for j in range(burn_in, len(valid_idx)):
    # re-fit every RECALC_EVERY steps using only data up to j-1
    if (j - burn_in) % RECALC_EVERY == 0 or res_i is None:
        train_series = returns_pct.loc[valid_idx[:j]]
        eg = arch_model(train_series, vol='EGARCH', p=1, o=1, q=1, dist='t')
        res_i = eg.fit(disp='off')

    # one-step-ahead variance forecast for time valid_idx[j]
    f_var_pct2 = res_i.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    f_var_raw = f_var_pct2 / (100.0 ** 2)
    pred_var.loc[valid_idx[j]] = f_var_raw


# keep only rows where we have a forecast
df = df.loc[pred_var.dropna().index].copy()
df['egarch_variance'] = pred_var.loc[df.index]
df['breach'] = (df['squared_returns'] > df['egarch_variance']).astype(int)
breaches = df[df['breach'] == 1]
df['egarch_vol'] = np.sqrt(df['egarch_variance'])  # σ_t


# ==========================================
# === STRATEGY IMPLEMENTATION (next-bar buy, 5% TP)
# ==========================================
capital = 1.0
position = 0
buy_price = 0.0
pending_entry = False

df['signal'] = 0
df['equity'] = np.nan

idx = df.index
for i in range(1, len(df)):
    price = df.loc[idx[i], 'close']

    # execute pending entry at next bar (with fee)
    if pending_entry and position == 0:
        position = 1
        buy_price = price
        capital *= (1 - FEE)               # pay entry fee
        df.loc[idx[i], 'signal'] = 1
        pending_entry = False

    # schedule entry if breach occurs and we're flat
    if position == 0 and not pending_entry:
        if df.loc[idx[i], 'squared_returns'] > df.loc[idx[i], 'egarch_variance']:
            pending_entry = True

    if position == 1:
        # --- Take-profit exit (simple space is fine) ---
        if price >= buy_price * (1 + TP_PCT):
            position = 0
            capital *= (price / buy_price) * (1 - FEE)   # exit with fee
            df.loc[idx[i], 'signal'] = -1

        else:
            # --- Vol-adjusted stop in LOG space ---
            sigma_t = df.loc[idx[i], 'egarch_vol']      # std of log returns
            log_ret_since_entry = np.log(price / buy_price)
            stop_threshold = -ALPHA * sigma_t           # stop if worse than -α·σ
            if log_ret_since_entry <= stop_threshold:
                position = 0
                capital *= (price / buy_price) * (1 - FEE)
                df.loc[idx[i], 'signal'] = -2

    # mark equity (MTM)
    df.loc[idx[i], 'equity'] = capital if position == 0 else capital * (price / buy_price)

# after the loop, drop rows without equity to avoid NaNs leaking into metrics
df = df.dropna(subset=['equity'])
entries = df[df['signal'] == 1]
exits   = df[df['signal'] == -1]
# ==========================================
# === BUY & HOLD EQUITY CURVE + METRICS  ===
# ==========================================
buyhold_initial = df['close'].iloc[0]
df['buyhold_equity'] = df['close'] / buyhold_initial

bh_final = df['buyhold_equity'].iloc[-1]
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25 if total_days > 0 else 0.0
bh_cagr = (bh_final ** (1/years) - 1) if years > 0 else np.nan

bh_logret = np.log(df['close'] / df['close'].shift(1)).dropna()

k = ANNUALIZE

ret_strat = df['equity'].pct_change().dropna()
ret_bh    = df['buyhold_equity'].pct_change().dropna()

def ann_sharpe(r, k):
    mu, sd = r.mean(), r.std()
    return (mu / sd) * np.sqrt(k) if sd and sd > 0 else np.nan

sharpe_ratio = ann_sharpe(ret_strat, k)
bh_sharpe    = ann_sharpe(ret_bh,    k)


bh_rolling = df['buyhold_equity'].cummax()
bh_drawdown = (df['buyhold_equity'] - bh_rolling) / bh_rolling
bh_maxdd = bh_drawdown.min()

# ===========================
# === STRATEGY METRICS    ===
# ===========================
trades = []
position = 0
entry_price = None
for i in range(1, len(df)):
    sig = df['signal'].iloc[i]
    if sig == 1:
        entry_date = df.index[i]
        entry_price = df['close'].iloc[i]
        position = 1
    elif position == 1 and sig in (-1, -2):
        exit_date = df.index[i]
        exit_price = df['close'].iloc[i]
        outcome = 'take_profit' if sig == -1 else 'stop_loss'
        pnl = (exit_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'outcome': outcome
        })
        position = 0
        entry_price = None
# Force-close any open trade at the final bar for honest stats
if position == 1 and entry_price is not None:
    final_price = df['close'].iloc[-1]
    trades.append({
        'entry_date': entry_date,
        'exit_date' : df.index[-1],
        'entry_price': entry_price,
        'exit_price' : final_price,
        'pnl'        : (final_price - entry_price) / entry_price,
        'outcome'    : 'forced_close'
    })
    # Optional: mark on chart
    df.iloc[-1, df.columns.get_loc('signal')] = -3

trades_df = pd.DataFrame(trades)

total_trades = len(trades_df)
profitable = trades_df[trades_df['outcome'] == 'take_profit']
stopped    = trades_df[trades_df['outcome'] == 'stop_loss']
win_rate = (trades_df['pnl'] > 0).mean() if total_trades > 0 else np.nan
avg_pnl    = trades_df['pnl'].mean() if total_trades > 0 else np.nan

total_return = df['equity'].iloc[-1] - 1
CAGR = (df['equity'].iloc[-1] ** (1/years) - 1) if years > 0 else np.nan

roll_max = df['equity'].cummax()
drawdown = (df['equity'] - roll_max) / roll_max
max_drawdown = drawdown.min()

equity_returns = df['equity'].pct_change().dropna()



# =======================================
# === CREATE /images/ AND SAVE PLOTS  ===
# =======================================
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# 1. Trade chart
plt.figure(figsize=(16, 8))
ax1 = plt.gca()

# Price
ax1.plot(df.index, df['close'], label='BTC Close', color='blue', alpha=0.6, linewidth=1.8)

# Events
ax1.scatter(breaches.index, breaches['close'], color='orange', marker='o', s=50,
            alpha=0.6, label='Variance Breach')

ax1.scatter(entries.index, entries['close'], color='green', marker='^', s=90,
            edgecolor='black', linewidth=0.6, label='Buy Entry')

ax1.scatter(exits.index, exits['close'], color='red', marker='v', s=90,
            edgecolor='black', linewidth=0.6, label='Exit')

ax1.set_ylabel('BTC Close Price', fontsize=12)
plt.title("BTC: EGARCH Variance Breach Strategy\nSignals and Trades", fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_trades.png"), dpi=300, bbox_inches="tight")
plt.close()


# 2. Equity curve
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['equity'], label='EGARCH Strategy', color='purple', linewidth=2)
plt.plot(df.index, df['buyhold_equity'], label='Buy & Hold', color='black', linestyle='--', linewidth=2)

plt.yscale('log')  # optional for clarity across years
plt.title('Equity Curve (Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Normalized)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_equity.png"), dpi=300, bbox_inches="tight")
plt.close()


# 3. Drawdowns
plt.figure(figsize=(14, 4))
plt.plot(drawdown.index, drawdown, color='red')
plt.title("EGARCH Strategy – Drawdowns")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_drawdown.png"), dpi=300, bbox_inches="tight")
plt.close()

# 4. Returns vs EGARCH Vol
egarch_vol_plot = (df['egarch_vol'] * 100.0)  # σ in %
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['log_return'] * 100.0, color='grey', alpha=0.5, label='Log Returns × 100')
plt.plot(df.index, egarch_vol_plot, color='blue', linewidth=2, label='EGARCH Volatility (%)')
plt.title('BTC Log Returns and EGARCH Model Volatility')
plt.xlabel('Date')
plt.ylabel('Log Return / Volatility (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_volatility.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved plots to:", IMG_DIR)

# ============================================
# === STRATEGY vs BUY & HOLD – METRICS TABLE
# ============================================

comparison = pd.DataFrame({
    "EGARCH Strategy": {
        "Total Return": df['equity'].iloc[-1] - 1,
        "CAGR": CAGR,
        f"Sharpe Ratio (annualize={ANNUALIZE})": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Trades": len(trades_df),
        "Win Rate": win_rate,
        "Avg Trade PnL": avg_pnl,
    },
    "Buy & Hold": {
        "Total Return": bh_final - 1,
        "CAGR": bh_cagr,
        f"Sharpe Ratio (annualize={ANNUALIZE})": bh_sharpe,
        "Max Drawdown": bh_maxdd,
        "Trades": np.nan,
        "Win Rate": np.nan,
        "Avg Trade PnL": np.nan,
    }
})




# ===== Save CSV (for spreadsheets / post-processing)
comparison.to_csv(os.path.join(IMG_DIR, "strategy_vs_buyhold.csv"))

# ===== Make a nicely formatted version for README printing and MD export
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

# ===== Save Markdown fragment (used for README injection)
md_path = os.path.join(IMG_DIR, "strategy_vs_buyhold.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("## 📈 Strategy vs Buy & Hold Metrics\n\n")
    f.write(formatted.to_markdown())

print(f"Saved Markdown metrics table to {md_path}")

# ===== Inject the Markdown fragment into README between tags
readme_path = "README.md"
start_tag = "<!--- METRICS_TABLE_START --->"
end_tag   = "<!--- METRICS_TABLE_END --->"

with open(md_path, "r", encoding="utf-8") as f:
    metrics_md = f.read()

with open(readme_path, "r", encoding="utf-8") as f:
    readme = f.read()

if start_tag in readme and end_tag in readme:
    pre  = readme.split(start_tag)[0]
    post = readme.split(end_tag)[1]
    new_readme = pre + start_tag + "\n" + metrics_md + "\n" + end_tag + post
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_readme)
    print(f"Updated {readme_path} with metrics table.")
else:
    print("README placeholders not found; skipped injection.")

def sweep_params(tp_list, alpha_list):
    rows = []
    for tp in tp_list:
        for a in alpha_list:
            sharpe, maxdd, cagr = run_backtest(tp, a)  # TODO: refactor core into this
            rows.append({'TP_PCT': tp, 'ALPHA': a, 'Sharpe': sharpe, 'MaxDD': maxdd, 'CAGR': cagr})
    res = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    return res

# 1) Next-bar entry check (cast to bool before ~)
prev_breach = df['breach'].shift(1).fillna(False).astype(bool)
violations = df.loc[(df['signal'].eq(1)) & (~prev_breach)]
print("Next-bar entry violations:", len(violations))

# 2) Same-bar entry & exit (safer version)
samebar = trades_df.loc[trades_df['entry_date'] == trades_df['exit_date']]
print("Same-bar entry/exit count:", len(samebar))


# 3) Optional: only flag violations when you were flat at t-1
# (build a simple position series from signals)
pos = np.zeros(len(df), dtype=int)
in_pos = 0
for i in range(len(df)):
    sig = df['signal'].iloc[i]
    if sig == 1 and in_pos == 0:
        in_pos = 1
    elif sig in (-1, -2) and in_pos == 1:
        in_pos = 0
    pos[i] = in_pos
df['pos'] = pos

violations_flat = df.loc[(df['signal'].eq(1)) & (~prev_breach) & (df['pos'].shift(1).fillna(0).eq(0))]
print("Next-bar entry violations (only when flat at t-1):", len(violations_flat))

mid = pd.Timestamp("2022-01-01")
def metrics(e):
    r = e.pct_change().dropna()
    k = bars_per_year(e.index)
    sharpe = r.mean()/r.std()*np.sqrt(k) if r.std()>0 else np.nan
    cagr = (e.iloc[-1]**(1/((e.index[-1]-e.index[0]).days/365.25)) - 1)
    return sharpe, cagr, (e/e.cummax()-1).min()
strat_pre  = metrics(df.loc[:mid, 'equity'])
strat_post = metrics(df.loc[mid:, 'equity'])
bh_pre     = metrics(df.loc[:mid, 'buyhold_equity'])
bh_post    = metrics(df.loc[mid:, 'buyhold_equity'])
print("Strat pre/post:", strat_pre, strat_post)
print("B&H   pre/post:", bh_pre, bh_post)

# ================================
# === Auto freq + Sharpe tests ===
# ================================
from scipy.stats import norm

def infer_annualization(index: pd.DatetimeIndex) -> int:
    """
    Infer annualization factor from the time index.
    Returns 252 for ~daily, 52 for ~weekly; falls back to bars_per_year(index).
    """
    if len(index) < 3:
        return 252
    # try pandas' inference first
    f = pd.infer_freq(index)
    if f:
        f = f.upper()
        if f.startswith("W"):
            return 52
        if f in ("D", "B"):
            return 252
    # fallback: median spacing in days
    mdays = np.median(np.diff(index.values).astype('timedelta64[D]').astype(float))
    if 4.0 <= mdays <= 9.0:
        return 52
    if 0.6 <= mdays <= 1.5:
        return 252
    # ultimate fallback: density-based estimate you already use
    return bars_per_year(index)

def jobson_korkie_test(r1: pd.Series, r2: pd.Series, freq: int) -> tuple:
    """
    Jobson–Korkie Sharpe difference test with Memmel correction.
    r1, r2 must be aligned (same timestamps). Returns (sh1, sh2, diff, z, p).
    """
    # align and drop NaNs
    rr = pd.concat([r1, r2], axis=1, join='inner').dropna()
    if rr.shape[0] < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    x = rr.iloc[:,0].to_numpy()
    y = rr.iloc[:,1].to_numpy()
    T = len(rr)

    mu1, mu2 = x.mean(), y.mean()
    s1, s2   = x.std(ddof=1), y.std(ddof=1)
    if s1 == 0 or s2 == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    sh1 = mu1 / s1 * np.sqrt(freq)
    sh2 = mu2 / s2 * np.sqrt(freq)

    cov12 = np.cov(x, y, ddof=1)[0,1]
    rho = cov12 / (s1*s2)

    diff = sh1 - sh2
    # Memmel correction
    var_diff = ((1 - rho) * ((1 + 0.5*sh1**2)/(T-1) + (1 + 0.5*sh2**2)/(T-1))
                - (sh1*sh2*rho)/(T-1))
    var_diff = max(var_diff, 1e-12)
    z = diff / np.sqrt(var_diff)
    p = 2*(1 - norm.cdf(abs(z)))
    return sh1, sh2, diff, z, p

# Determine correct annualization for YOUR dataset
ANNUALIZE = infer_annualization(df.index)

# Build return series once
ret_strat_full = df['equity'].pct_change()
ret_bh_full    = df['buyhold_equity'].pct_change()

pre_mask  = df.index <  "2022-01-01"
post_mask = df.index >= "2022-01-01"

# === Paired Circular Block Bootstrap (define BEFORE you call it) ===
def annualized_sharpe_from_returns(r, freq: int):
    r = np.asarray(r)
    mu = r.mean()
    sd = r.std(ddof=1)
    return (mu / sd) * np.sqrt(freq) if sd > 0 else np.nan

def block_bootstrap_sharpe_diff(r1: pd.Series, r2: pd.Series, freq: int,
                                B: int = 5000, block: int = 10, seed: int = 42):
    """
    Paired circular block bootstrap of ΔSharpe = Sharpe(r1) - Sharpe(r2).
    Returns (d_obs, (ci_low, ci_high), p_two_sided, p_one_sided_pos).
    """
    rng = np.random.default_rng(seed)
    rr = pd.concat([r1, r2], axis=1, join='inner').dropna()
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
            (idx[s:(s+block)] if s+block <= T else np.r_[idx[s:], idx[:(s+block) % T]])
            for s in starts
        ])[:T]
        xb = x[boot_idx]
        yb = y[boot_idx]
        diffs[b] = (annualized_sharpe_from_returns(xb, freq)
                    - annualized_sharpe_from_returns(yb, freq))

    # 95% percentile CI
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # H0: Δ = 0  (center bootstrap at 0 for p-values)
    diffs0 = diffs - diffs.mean()
    p_two = (np.sum(np.abs(diffs0) >= np.abs(d_obs)) + 1) / (B + 1)
    p_one_pos = (np.sum(diffs0 >= d_obs) + 1) / (B + 1)

    return d_obs, (ci_low, ci_high), p_two, p_one_pos


for label, r1, r2 in [
    ("Full Sample", df['equity'].pct_change(), df['buyhold_equity'].pct_change()),
    ("Pre-2022",    df.loc[df.index <  "2022-01-01", 'equity'].pct_change(),
                    df.loc[df.index <  "2022-01-01", 'buyhold_equity'].pct_change()),
    ("Post-2022",   df.loc[df.index >= "2022-01-01", 'equity'].pct_change(),
                    df.loc[df.index >= "2022-01-01", 'buyhold_equity'].pct_change()),
]:
    d, ci, p2, p1 = block_bootstrap_sharpe_diff(r1, r2, freq=ANNUALIZE, B=5000, block=10)
    print(f"\n{label} Bootstrap ΔSharpe (annualize={ANNUALIZE}, block=10, B=5000)")
    print(f"ΔSharpe = {d:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"p_two_sided = {p2:.4f},  p_one_sided (Δ>0) = {p1:.4f}")

wins = trades_df.loc[trades_df.pnl>0,'pnl']; losses = trades_df.loc[trades_df.pnl<0,'pnl']
profit_factor = wins.sum() / -losses.sum()
expectancy = trades_df['pnl'].mean(); median_pnl = trades_df['pnl'].median()
df_w = df_raw.resample('W-MON').last().dropna()
# recompute log_return, EGARCH, signals, etc., then set annualize=52

def effective_sample_size(r, max_lag=20):
    r = r.dropna().values
    r = (r - r.mean()) / r.std(ddof=1)
    N = len(r)
    acf = [np.corrcoef(r[:-k], r[k:])[0,1] for k in range(1, min(max_lag, N-1))]
    # Bartlett/Newey-West style shrink
    ess = N / (1 + 2*sum(acf_k for acf_k in acf if not np.isnan(acf_k)))
    return max(5, ess)

ess_strat = effective_sample_size(df['equity'].pct_change())
ess_bh    = effective_sample_size(df['buyhold_equity'].pct_change())
print("Effective N (strat, bh):", ess_strat, ess_bh)

# bootstrap mean trade PnL and profit factor
pnl = trades_df['pnl'].dropna().values
wins = pnl[pnl>0]; losses = pnl[pnl<0]
profit_factor = wins.sum() / -losses.sum()
print("Profit Factor:", profit_factor, "Mean trade PnL:", pnl.mean(), "Median:", np.median(pnl))

rng = np.random.default_rng(42)
B=10000
boot_means = np.array([rng.choice(pnl, size=len(pnl), replace=True).mean() for _ in range(B)])
ci = np.percentile(boot_means, [2.5, 97.5])
p_two = (np.sum(np.abs(boot_means) >= np.abs(pnl.mean())) + 1) / (B + 1)
print("Trade-level mean PnL 95% CI:", ci, " two-sided p:", p_two)

target = 0.02  # target 2% bar vol, example; tune
size = np.clip(target / (df['egarch_vol'].replace(0,np.nan)), 0, 1).fillna(0)
# apply 'size' to PnL changes when in position


def block_bootstrap_sharpe_diff(r1: pd.Series, r2: pd.Series, freq: int,
                                B: int = 5000, block: int = 10, seed: int = 42):
    """
    Paired circular block bootstrap of ΔSharpe = Sharpe(r1) - Sharpe(r2).
    Returns (d_obs, (ci_low, ci_high), p_two_sided, p_one_sided_pos).
    """
    rng = np.random.default_rng(seed)
    rr = pd.concat([r1, r2], axis=1, join='inner').dropna()
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
            (idx[s:(s+block)] if s+block <= T else np.r_[idx[s:], idx[:(s+block) % T]])
            for s in starts
        ])[:T]
        xb = x[boot_idx]; yb = y[boot_idx]
        diffs[b] = annualized_sharpe_from_returns(xb, freq) - annualized_sharpe_from_returns(yb, freq)

    # 95% percentile CI
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # H0: Δ = 0  (center bootstrap at 0 for p-values)
    diffs0 = diffs - diffs.mean()
    p_two = (np.sum(np.abs(diffs0) >= np.abs(d_obs)) + 1) / (B + 1)
    p_one_pos = (np.sum(diffs0 >= d_obs) + 1) / (B + 1)

    return d_obs, (ci_low, ci_high), p_two, p_one_pos

df_w = df_raw.resample('W-MON').last().dropna()
# recompute log_return, EGARCH, signals, etc., then set annualize=52

bh_logret = np.log(df['close'] / df['close'].shift(1)).dropna()
equity_returns = df['equity'].pct_change().dropna()
expectancy = trades_df['pnl'].mean(); median_pnl = trades_df['pnl'].median()

breaches_plot = breaches.loc[breaches.index.intersection(df.index)]
ax1.scatter(breaches_plot.index, breaches_plot['close'], color='orange', marker='o', s=60, label='Variance Breach (All)')

