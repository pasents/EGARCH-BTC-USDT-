# BTC/USDT EGARCH Variance-Breach Backtest

This repository implements and backtests a **variance-breach trading strategy** on BTC/USDT using the **EGARCH(1,1)** volatility model (`arch` library) with **out-of-sample one-step-ahead forecasts** and **next-bar execution**.

## 🔧 Strategy Logic (what actually runs)
- Fit **EGARCH(1,1)** with Student-t errors to BTC **log returns**.
- **Re-fit every 30 bars** on an expanding window; use the **one-step-ahead** variance forecast for bar *t*.
- Flag a **variance breach** when **squared return\_t > forecast variance\_t**.
- If flat, **enter long on the next bar** after a breach (avoids look-ahead).
- **Exit rules**
  - **Take-profit:** **+8%** relative to entry.
  - **Vol-adjusted stop:** stop when **log(price/entry) ≤ −α·σ\_t**, with **α = 5.2** and **σ\_t** the EGARCH-forecasted std. dev.
- **Costs:** **5 bps per side** (spot-like fees) applied on entry and exit.
- **Benchmark:** Buy-and-hold (normalized).

> The code also includes robustness/reporting utilities:
> - Paired **circular block bootstrap** for ΔSharpe (with CIs and p-values)
> - **Jobson–Korkie** Sharpe difference test (Memmel correction)
> - **Pre- vs Post-2022** regime split metrics
> - **Effective sample size (ESS)** estimate for returns
> - Sanity checks for **next-bar entry** and **same-bar entry/exit** violations
> - Auto-export of plots and a Markdown **metrics table** injected into this README between tags

---

## 📂 Project Files

---

.
├─ egarch.py # Main backtest script (signals, execution, metrics, plots, README injection)
├─ BTCUSDTmergeddataset.csv # Input dataset (timestamp, close) — day-first dates
├─ images/
│ ├─ egarch_trades.png
│ ├─ egarch_equity.png
│ ├─ egarch_drawdown.png
│ ├─ egarch_volatility.png
│ ├─ strategy_vs_buyhold.csv
│ └─ strategy_vs_buyhold.md # Markdown fragment injected into README
└─ README.md

---

## 📊 Features
- EGARCH(1,1) volatility **estimation & forecasting**
- **Variance-breach** signal generation
- **Next-bar** entries; TP/vol-stop exits; **fee-aware** PnL
- Equity curve (strategy & B&H), drawdowns, return series
- Metrics: **Total Return, CAGR, Sharpe (annualized), Max Drawdown, Trades, Win Rate, Avg Trade PnL**
- Robustness: **Bootstrap ΔSharpe**, **Jobson–Korkie**, **regime split**, **ESS**

---

## 📑 Data Requirements
CSV must contain:
- `timestamp` — parsed with **day-first** dates
- `close` — BTC/USDT close price

Duplicates are removed. The script reports **missing ISO weeks** for awareness.

---

## ⚙️ Installation
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install pandas numpy matplotlib arch scipy


## ▶️ Usage
```bash
python egarch.py
```

---

## Key parameters inside egarch.py:

---

TP_PCT = 0.08        # 8% take-profit
ALPHA = 5.2          # multiplier for volatility-adjusted stop in log space
FEE = 0.0005         # 5 bps per side
RECALC_EVERY = 30    # EGARCH re-fit cadence
USE_WEEKLY = False   # set True to run on weekly bars (W-MON)

## ⚠️ Disclaimer
This project is **not** financial advice. Use at your own risk.

---

## Results

---

<!--- METRICS_TABLE_START --->
## 📈 Strategy vs Buy & Hold Metrics

|                              | EGARCH Strategy   | Buy & Hold   |
|:-----------------------------|:------------------|:-------------|
| Total Return                 | 6336.94%          | 2722.50%     |
| CAGR                         | 93.35%            | 69.69%       |
| Sharpe Ratio (annualize=252) | 1.2503            | 0.9521       |
| Max Drawdown                 | -54.76%           | -76.63%      |
| Trades                       | 82                | —            |
| Win Rate                     | 84.15%            | —            |
| Avg Trade PnL                | 5.94%             | —            |
<!--- METRICS_TABLE_END --->

---

### Strategy Trades
<img src="images/egarch_trades.png" width="700">

---

### Equity Curve
<img src="images/egarch_equity.png" width="700">

---

### Drawdowns
<img src="images/egarch_drawdown.png" width="700">

---

### Returns vs EGARCH Volatility
<img src="images/egarch_volatility.png" width="700">
