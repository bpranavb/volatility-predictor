# volatility-predictor
# 📊 Stock Volatility Prediction with Machine Learning

This project predicts 20-day rolling volatility of a stock using historical price data, technical indicators, and macroeconomic signals. It includes model building, evaluation, and simulated trading strategy performance.

## Features Used

- Moving Averages (MA20, MA50, MA100)
- RSI, Bollinger Bands, MACD, ATR
- Volume spike flags
- Macro data (VIX index, SPY returns)
- Lagged features (previous day's returns and volatility)

## 🤖 Model

- `RandomForestRegressor` (baseline)
- `XGBoostRegressor` (planned upgrade)
- Evaluation: MAE, R², and visual comparison of predicted vs actual volatility

## 📈 Backtesting Strategy (Coming Soon)

- Trade when volatility breaks thresholds
- Metrics: Sharpe ratio, drawdown, total return

## 🌐 Future Plans

- Add GARCH-based volatility model
- Switch from level prediction → delta volatility
- Build a Streamlit dashboard to visualize everything
- Optimize hyperparameters and add feature importance charts

---

Created by [@bpranavb](https://github.com/bpranavb)

