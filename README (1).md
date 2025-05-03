
# 📈 Volatility ML Trading Strategy

This project builds a **machine learning + GARCH-powered system** to forecast 20-day rolling volatility and simulate a trading strategy based on volatility trends, changes, and momentum.

It combines:
- 🧠 **XGBoost** for volatility prediction
- 🔄 **ΔVolatility (volatility-of-volatility)** model for risk regime shifts
- 📉 **GARCH(1,1)** for benchmark volatility modeling
- 🛠️ **Threshold-tuned strategy logic** to simulate buy signals
- 📊 **Pine Script on TradingView** to visualize backtested trades

---

## 🗂️ Project Structure

```
volatility_trading_strategy/
├── volatility_strategy.ipynb         ← Full XGBoost + GARCH pipeline
├── final_xgboost_garch_strategy.ipynb← Final model with tuned strategy logic
├── tradingview_script.pine           ← Pine Script implementation of strategy
├── project_report.pdf                ← (Coming soon)
├── requirements.txt                  ← Python dependencies
├── screenshots/                      ← PnL, Sharpe ratio, backtest charts
└── streamlit_app/                    ← (Optional) Streamlit dashboard
```

---

## 📊 Features Used

### 🧠 Technical Indicators
- Moving Averages: MA20, MA50, MA100  
- RSI (Relative Strength Index)  
- Bollinger Band Width  
- ATR (Average True Range)  
- MACD and MACD_diff  

### 🌐 Macroeconomic Signals
- VIX (Market Volatility Index)  
- SPY (S&P 500 ETF)

### 🎯 Targets
- 20-day Rolling Volatility  
- 5-day ΔVolatility (% change of rolling std)

---

## ⚙️ Models

- `XGBoostRegressor` for predicting:
  - 20-day rolling volatility
  - 5-day ΔVolatility
- `arch_model` from `arch` library for **GARCH(1,1)** volatility forecast
- `GridSearchCV` used to tune learning rate, estimators, max depth

---

## 🧠 Strategy Logic

A **Buy Signal** is triggered when:

```
|ΔVolatility| > ΔVol Threshold
AND
(GARCH Vol - ML Vol) > GARCH Threshold
AND
3-Day Momentum > 0
```

This filters for moments when:
- Volatility is shifting rapidly
- GARCH signals higher uncertainty than the ML model
- Price action is positive (confirming breakout)

---

## 📈 Backtest Performance (Python Sim)

**2015–2025 Results:**

| Metric              | Value        |
|---------------------|--------------|
| Total Return        | **134.00%**  |
| Annualized Return   | **8.90%**    |
| Volatility          | **13.79%**   |
| Sharpe Ratio        | **4.67**     |
| Max Drawdown        | **14.18%**   |
| Win Rate            | **20.05%**   |

---

## 🧪 TradingView Backtest

📌 Live script: [Volatility ML Strategy on TradingView](https://www.tradingview.com/script/yLgQ6HBi-Volatility-ML-Strategy/)

**TradingView Report Highlights:**

| Metric             | Value           |
|--------------------|------------------|
| Total P&L          | +1.29M USD       |
| Max Drawdown       | 88.81%           |
| Sharpe Ratio       | 0.032            |
| Profit Factor      | 1.072            |
| Trades             | 981              |
| Win Rate           | 43.2%            |

Note: Pine Script logic is a proxy, not a full ML port, but provides reliable **signal-based validation** of strategy mechanics.

---

## 🔮 Forecast Output

Each model outputs:
- 📈 Next-day 20-day rolling volatility prediction
- ⚠️ ΔVolatility signal for volatility change risk
- 🔍 GARCH-based volatility estimate
- 💰 Expected price range (Upper/Lower bound)

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## 🚧 Next Steps

- ✅ Public Pine Script release on TradingView
- ⏳ Add Streamlit dashboard for forecast interaction
- 🛰️ Deploy real-time inference using live stock feeds (Tiingo/YFinance)
- 📑 Release full project report (PDF)

---

## 👤 Author

**Pranav Bollineni**  
🔗 [LinkedIn](https://www.linkedin.com/in/bpranavb/)  
📊 Quant / Data Science @ Northeastern  
💻 GitHub: [@bpranavb](https://github.com/bpranavb)
