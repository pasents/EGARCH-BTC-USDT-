
# BTC/USDT — EGARCH Variance-Breach Backtest

A quantitative research implementation of a **variance-breach long strategy** on BTC/USDT. The model uses **EGARCH(1,1)** (Student-t) to produce **out-of-sample one-step-ahead** variance forecasts and executes **next-bar** with **fee-aware** P&L.

## 🔧 Strategy Logic
- **Model:** EGARCH(1,1) with Student-t errors on BTC **log returns**; expanding window; **re-fit every 30 bars**.
- **Forecast:** use the **one-step-ahead** variance forecast available at bar *t*.
- **Signal (variance breach):** go long when `squared_return_t > forecast_variance_t`.
- **Execution:** if flat, **enter on the next bar** after the breach (prevents look-ahead).
- **Exits:**
  - **Take-profit:** **+8%** vs. entry.
  - **Vol-adjusted stop (log space):** exit when `log(price/entry) ≤ −α·σ_t`, with **α = 5.2** and **σ_t** the EGARCH-forecasted stdev.
- **Costs:** **5 bps per side** on entry and exit.
- **Benchmark:** normalized **buy-and-hold**.
- **Annualization:** inferred from the index (typ. **252** daily / **52** weekly).

> **Robustness & Reporting**
> - Paired **circular block bootstrap** for ΔSharpe (CIs & p-values)  
> - **Jobson–Korkie** Sharpe difference test (Memmel correction)  
> - **Pre- vs Post-2022** regime split metrics  
> - **Effective sample size (ESS)** for dependent returns  
> - Sanity checks for **next-bar entry** and **same-bar entry/exit**  
> - Auto-exported plots and a Markdown **metrics table** injected between README tags

---

## 📂 Project Files
```text
.
├─ egarch.py
├─ BTCUSDTmergeddataset.csv
├─ images/
│  ├─ egarch_trades.png
│  ├─ egarch_equity.png
│  ├─ egarch_drawdown.png
│  ├─ egarch_volatility.png
│  ├─ strategy_vs_buyhold.csv
│  └─ strategy_vs_buyhold.md
└─ README.md
```

---

## ⚙️ Installation
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib arch scipy
```
---

## requirements.txt`:
```
pandas
numpy
matplotlib
arch
```

---

## ▶️ Usage
```bash
python egarch.py
```

---

##🔑 Key parameters
```
TP_PCT = 0.08        # 8% take-profit
ALPHA = 5.2          # volatility-adjusted stop multiplier
FEE = 0.0005         # 5 bps per side
RECALC_EVERY = 30    # EGARCH re-fit cadence
USE_WEEKLY = False   # True = weekly (W-MON), else daily

```

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
<!--- METRICS_TABLE_START --->
(old metrics will be replaced here)
<!--- METRICS_TABLE_END --->

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
