<<<<<<< HEAD
# BTC/USDT EGARCH Variance-Breach Backtest

This project implements and backtests a **variance-breach trading strategy** on BTC/USDT using the **EGARCH(1,1)** volatility model from the `arch` package.

**Strategy logic**:
- Fits an EGARCH model to BTC log returns.
- Detects **variance breaches** (when squared returns exceed model variance).
- Goes **long** on a breach.
- Exits at **+17% take-profit** or **−3% stop-loss**.
- Compares performance to a **buy-and-hold** benchmark.

---

## 📂 Project Files
```
.
├─ egarch.py                  # Main Python script
├─ BTCUSDTmergeddataset.csv   # Input dataset (timestamp, close)
└─ README.md                  # This file
```

---

## 📊 Features
- EGARCH(1,1) volatility estimation
- Variance-breach signal generation
- Trade execution logic with TP & SL
- Equity curve tracking
- Performance metrics:
  - Total return
  - CAGR
  - Sharpe ratio
  - Max drawdown
  - Win rate
- Plots:
  - Price with breaches & trades
  - Strategy vs buy-and-hold equity
  - EGARCH volatility vs returns

---

## 📑 Data Requirements
The CSV must contain:
- `timestamp` — in **day-first** format
- `close` — closing price

Duplicate timestamps are removed.  
Missing ISO weeks are reported in the terminal for awareness.

---

## ⚙️ Installation
```bash
python -m venv .venv

# Activate the environment:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install pandas numpy matplotlib arch
```

**Optional** `requirements.txt`:
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
The script will:
- Output performance metrics in the terminal.
- Display plots for price action, breaches, trades, and equity curves.

---

## 📈 Example Output
*Example performance plot from a sample run:*

![Example Equity Curve](example_equity_curve.png)

---

## 📝 Notes
- All thresholds (e.g., TP = 17%, SL = 3%) are defined in `egarch.py` and can be changed.
- No transaction costs, slippage, or liquidity constraints are modeled.
- For **research and educational purposes only**.

---

## ⚠️ Disclaimer
This project is **not** financial advice. Use at your own risk.
=======
# EGARCH-BTC-USDT-
Backtests a BTC/USDT volatility strategy. Loads BTCUSDTmergeddataset.csv, computes log returns, fits EGARCH(1,1), flags variance breaches, buys on breach, exits at +17% TP or −3% SL. Compares equity to buy-and-hold, reports CAGR/Sharpe/max drawdown/win rate, and plots price, breaches, trades, equity. Req: pandas, numpy, matplotlib, arch.
>>>>>>> 816e9ab1780f2479832faa84dcf6c1be5d8d5a99
