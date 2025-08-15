# EGARCH-BTC-USDT-
Backtests a BTC/USDT volatility strategy. Loads BTCUSDTmergeddataset.csv, computes log returns, fits EGARCH(1,1), flags variance breaches, buys on breach, exits at +17% TP or −3% SL. Compares equity to buy-and-hold, reports CAGR/Sharpe/max drawdown/win rate, and plots price, breaches, trades, equity. Req: pandas, numpy, matplotlib, arch.
