import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# =========================
# === LOAD & PREP DATA ===
# =========================
# Parse dates (day-first in your CSV) and use a DateTimeIndex
df = pd.read_csv("BTCUSDTmergeddataset.csv", parse_dates=['timestamp'], dayfirst=True)
df = df.sort_values('timestamp').set_index('timestamp')
df = df[~df.index.duplicated(keep='first')]   # drop any duplicate timestamps

# Use LOG returns everywhere for consistency
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
# Fit EGARCH on percent log-returns
returns_pct = df['log_return'].dropna() * 100.0
egarch = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1, dist='normal')
res = egarch.fit(disp='off')

# EGARCH conditional vol is in the same units as input (percent)
egarch_vol_pct = res.conditional_volatility                   # %
egarch_vol_raw = egarch_vol_pct / 100.0                       # fraction
egarch_var_raw = egarch_vol_raw ** 2

# Align EGARCH variance back into df by index and keep overlapping rows
df['egarch_variance'] = np.nan
df.loc[egarch_var_raw.index, 'egarch_variance'] = egarch_var_raw.values
df = df.loc[~df['egarch_variance'].isna()].copy()

# Mark variance-breach events (log-return squared vs EGARCH variance)
df['breach'] = (df['squared_returns'] > df['egarch_variance']).astype(int)
breaches = df[df['breach'] == 1]

# ==========================================
# === STRATEGY IMPLEMENTATION (breach long)
# ==========================================
capital = 1.0
position = 0   # 0 = flat, 1 = long
buy_price = 0.0

df['signal'] = 0        # 1 = buy, -1 = take profit, -2 = stop
df['equity'] = np.nan

idx = df.index
for i in range(1, len(df)):
    price = df.loc[idx[i], 'close']

    # Entry: variance breach
    if position == 0 and df.loc[idx[i], 'squared_returns'] > df.loc[idx[i], 'egarch_variance']:
        position = 1
        buy_price = price
        df.loc[idx[i], 'signal'] = 1

    # Take profit: +17%
    elif position == 1 and price >= buy_price * 1.17:
        position = 0
        df.loc[idx[i], 'signal'] = -1
        capital *= (price / buy_price)

    # Stop loss: -3%
    elif position == 1 and price <= buy_price * 0.97:
        position = 0
        df.loc[idx[i], 'signal'] = -2
        capital *= (price / buy_price)

    # Mark-to-market equity
    df.loc[idx[i], 'equity'] = capital if position == 0 else capital * (price / buy_price)

# Forward-fill equity safely
df['equity'] = df['equity'].ffill()

# ==============================
# === MARK ENTRIES / EXITS   ===
# ==============================
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

# ================================================
# === PLOT ENTRIES/EXITS, BREACHES, & EGARCH VAR ==
# ================================================
plt.figure(figsize=(16, 8))
ax1 = plt.gca()
ax1.plot(df.index, df['close'], label='BTC Close', color='blue', alpha=0.7)

# Breaches
ax1.scatter(breaches.index, breaches['close'], color='orange', marker='o', s=60, label='Variance Breach (All)')
for t in breaches.index:
    ax1.axvline(t, color='orange', alpha=0.13, linestyle='-', linewidth=2)

# Trades
ax1.scatter(entries.index, entries['close'], label='Buy (Entry)', color='green', marker='^', s=80)
ax1.scatter(exits.index,   exits['close'],   label='Sell (+17%)', color='red', marker='v', s=80)
ax1.scatter(stops.index,   stops['close'],   label='Sell (Stop -3%)', color='black', marker='x', s=80)

ax1.set_ylabel('BTC Close Price')
ax1.legend(loc='upper left')
plt.title("BTC: EGARCH Variance Breach Strategy\nSignals, Trades, and EGARCH Model Variance")
plt.tight_layout()
plt.show()

# ===============================
# === PLOT BOTH EQUITY CURVES ===
# ===============================
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['equity'], label='EGARCH Strategy', color='purple', linewidth=2)
plt.plot(df.index, df['buyhold_equity'], label='Buy & Hold', color='black', linestyle='--', linewidth=2)
plt.title('Equity Curve: EGARCH Variance Breach Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Normalized)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================================
# === PRINT METRICS SIDE BY SIDE (SUMMARY)   ===
# ==============================================
print("\n" + "="*40)
print("          PERFORMANCE SUMMARY")
print("="*40)
print("Metric".ljust(28), "EGARCH".ljust(16), "Buy & Hold")
print("-"*40)
print(f"Total Return:".ljust(28), f"{total_return:.2%}".ljust(16), f"{bh_final-1:.2%}")
print(f"Annualized (CAGR):".ljust(28), f"{(CAGR if pd.notna(CAGR) else float('nan')):.2%}".ljust(16), f"{(bh_cagr if pd.notna(bh_cagr) else float('nan')):.2%}")
print(f"Sharpe Ratio:".ljust(28), f"{(sharpe_ratio if pd.notna(sharpe_ratio) else float('nan')):.2f}".ljust(16), f"{(bh_sharpe if pd.notna(bh_sharpe) else float('nan')):.2f}")
print(f"Max Drawdown:".ljust(28), f"{max_drawdown:.2%}".ljust(16), f"{bh_maxdd:.2%}")
print("-"*40)
print(f"Total trades:".ljust(28), f"{total_trades}")
print(f"Win rate:".ljust(28), f"{(win_rate if pd.notna(win_rate) else float('nan')):.2%}")
print("="*40 + "\n")

# =======================================================
# === STRATEGY METRICS IN DETAIL + RETURNS/VOL PLOT   ===
# =======================================================
print("\n" + "="*40)
print("   EGARCH VARIANCE BREACH STRATEGY METRICS")
print("="*40)
print(f"Total trades:              {total_trades}")
print(f"Profitable exits (+17%):   {len(profitable)}")
print(f"Stopped out (-3%):         {len(stopped)}")
print(f"Win rate:                  {(win_rate if pd.notna(win_rate) else float('nan')):.2%}")
print(f"Average P&L per trade:     {(avg_pnl if pd.notna(avg_pnl) else float('nan')):.2%}")
print(f"Total return:              {total_return:.2%}")
print(f"Annualized return (CAGR):  {(CAGR if pd.notna(CAGR) else float('nan')):.2%}")
print(f"Maximum drawdown:          {max_drawdown:.2%}")
print(f"Sharpe ratio:              {(sharpe_ratio if pd.notna(sharpe_ratio) else float('nan')):.2f}")
print("="*40 + "\n")

# Academic-style Returns & Volatility Plot
# (Vol series aligned on df's index for plotting continuity)
egarch_vol_plot = res.conditional_volatility.reindex(df.index).ffill()
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['log_return'] * 100.0, color='grey', alpha=0.5, label='Log Returns × 100')
plt.plot(df.index, egarch_vol_plot, color='blue', linewidth=2, label='EGARCH Volatility (%)')
plt.title('BTC Log Returns and EGARCH Model Volatility')
plt.xlabel('Date')
plt.ylabel('Log Return / Volatility (%)')
plt.legend()
plt.tight_layout()
plt.show()
