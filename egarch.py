import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# =========================
# === LOAD & PREP DATA ===
# =========================
df = pd.read_csv("BTCUSDTmergeddataset.csv", parse_dates=['timestamp'], dayfirst=True)
df = df.sort_values('timestamp').set_index('timestamp')
df = df[~df.index.duplicated(keep='first')]

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
# ==================================
returns_pct = df['log_return'].dropna() * 100.0
egarch = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1, dist='normal')
res = egarch.fit(disp='off')

egarch_vol_pct = res.conditional_volatility
egarch_vol_raw = egarch_vol_pct / 100.0
egarch_var_raw = egarch_vol_raw ** 2

df['egarch_variance'] = np.nan
df.loc[egarch_var_raw.index, 'egarch_variance'] = egarch_var_raw.values
df = df.loc[~df['egarch_variance'].isna()].copy()

df['breach'] = (df['squared_returns'] > df['egarch_variance']).astype(int)
breaches = df[df['breach'] == 1]

# ==========================================
# === STRATEGY IMPLEMENTATION (breach long)
# ==========================================
capital = 1.0
position = 0
buy_price = 0.0

df['signal'] = 0
df['equity'] = np.nan

idx = df.index
for i in range(1, len(df)):
    price = df.loc[idx[i], 'close']

    if position == 0 and df.loc[idx[i], 'squared_returns'] > df.loc[idx[i], 'egarch_variance']:
        position = 1
        buy_price = price
        df.loc[idx[i], 'signal'] = 1

    elif position == 1 and price >= buy_price * 1.17:
        position = 0
        df.loc[idx[i], 'signal'] = -1
        capital *= (price / buy_price)

    elif position == 1 and price <= buy_price * 0.97:
        position = 0
        df.loc[idx[i], 'signal'] = -2
        capital *= (price / buy_price)

    df.loc[idx[i], 'equity'] = capital if position == 0 else capital * (price / buy_price)

df['equity'] = df['equity'].ffill()

entries = df[df['signal'] == 1]
exits   = df[df['signal'] == -1]
stops   = df[df['signal'] == -2]

# ==========================================
# === BUY & HOLD EQUITY CURVE + METRICS  ===
# ==========================================
buyhold_initial = df['close'].iloc[0]
df['buyhold_equity'] = df['close'] / buyhold_initial

bh_final = df['buyhold_equity'].iloc[-1]
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25 if total_days > 0 else 0.0
bh_cagr = (bh_final ** (1/years) - 1) if years > 0 else np.nan

bh_logret = np.log(df['close'] / df['close'].shift(1))
bh_vol = bh_logret.std()
bh_sharpe = (bh_logret.mean() / bh_vol) * np.sqrt(52) if bh_vol and bh_vol > 0 else np.nan

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

trades_df = pd.DataFrame(trades)

total_trades = len(trades_df)
profitable = trades_df[trades_df['outcome'] == 'take_profit']
stopped    = trades_df[trades_df['outcome'] == 'stop_loss']
win_rate   = len(profitable) / total_trades if total_trades > 0 else np.nan
avg_pnl    = trades_df['pnl'].mean() if total_trades > 0 else np.nan

total_return = df['equity'].iloc[-1] - 1
CAGR = (df['equity'].iloc[-1] ** (1/years) - 1) if years > 0 else np.nan

roll_max = df['equity'].cummax()
drawdown = (df['equity'] - roll_max) / roll_max
max_drawdown = drawdown.min()

equity_returns = df['equity'].pct_change().dropna()
sr_denom = equity_returns.std()
sharpe_ratio = (equity_returns.mean() / sr_denom) * np.sqrt(52) if sr_denom and sr_denom > 0 else np.nan

# =======================================
# === CREATE /images/ AND SAVE PLOTS  ===
# =======================================
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# 1. Trade chart
plt.figure(figsize=(16, 8))
ax1 = plt.gca()
ax1.plot(df.index, df['close'], label='BTC Close', color='blue', alpha=0.7)

ax1.scatter(breaches.index, breaches['close'], color='orange', marker='o', s=60, label='Variance Breach (All)')
ax1.scatter(entries.index, entries['close'], label='Buy (Entry)', color='green', marker='^', s=80)
ax1.scatter(exits.index,   exits['close'],   label='Sell (+17%)', color='red', marker='v', s=80)
ax1.scatter(stops.index,   stops['close'],   label='Sell (Stop -3%)', color='black', marker='x', s=80)

ax1.set_ylabel('BTC Close Price')
ax1.legend(loc='upper left')
plt.title("BTC: EGARCH Variance Breach Strategy\nSignals, Trades, and EGARCH Model Variance")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "egarch_trades.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Equity curve
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['equity'], label='EGARCH Strategy', color='purple', linewidth=2)
plt.plot(df.index, df['buyhold_equity'], label='Buy & Hold', color='black', linestyle='--', linewidth=2)
plt.title('Equity Curve: EGARCH Variance Breach Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Normalized)')
plt.legend()
plt.grid(True)
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
egarch_vol_plot = res.conditional_volatility.reindex(df.index).ffill()
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
        "Sharpe Ratio (weekly)": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Trades": len(trades_df),
        "Win Rate": win_rate,
        "Avg Trade PnL": avg_pnl,
    },
    "Buy & Hold": {
        "Total Return": bh_final - 1,
        "CAGR": bh_cagr,
        "Sharpe Ratio (weekly)": bh_sharpe,
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
