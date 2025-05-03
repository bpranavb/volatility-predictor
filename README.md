# Volatility Trading Strategy

This project builds a machine learning and Generalized Autoregressive Conditional Heteroskedasticity(GARCH) - powered system to predict 20-day stock volatility using financial indicators and macroeconomic signals. It then simulates a **volatility-driven trading strategy**, combining:
- An XGBoost model for volatility prediction
- A ΔVolatility model for volatility changes
- GARCH modeling for benchmarking volatility
- A strategy that uses these to generate buy signals with risk-aware thresholds

---

## Project Structure

- `volatility_strategy.ipynb` – Full Jupyter notebook with:
  - Data collection from Tiingo
  - Feature engineering (RSI, MACD, ATR, BB Width, etc.)
  - Volatility prediction using XGBoost
  - GARCH(1,1) volatility modeling
  - Strategy simulation with threshold tuning
  - Performance metrics and charts

- `requirements.txt` – All required libraries (xgboost, arch, ta, yfinance, etc.)

---

## Features Used

- **Technical Indicators:**  
  - MA20, MA50, MA100  
  - RSI  
  - Bollinger Band Width  
  - ATR  
  - MACD and MACD_diff

- **Macro Signals:**  
  - VIX Close  
  - SPY Close

- **Target Variables:**  
  - 20-day Rolling Volatility  
  - 5-day ΔVolatility % Change

---

## Models Used

- `XGBoostRegressor` for 20-day volatility
- `XGBoostRegressor` for 5-day ΔVolatility
- `arch_model` (GARCH(1,1)) for benchmark volatility

GridSearchCV is used for hyperparameter tuning.

---

## Strategy Logic

**Buy Signal** is generated when:
- `|ΔVolatility|` > `ΔVol Threshold` (significant volatility change)
- GARCH Volatility > ML Predicted Volatility by `GARCH Threshold`
- 3-day momentum > 0 (price is rising)

Final returns are calculated using actual price movement the next day.

---

## Performance

Example backtest (2015–2025):

- **Total Return**: 134.00%  
- **Annualized Return**: 8.90%  
- **Volatility**: 13.79%  
- **Sharpe Ratio**: 4.67  
- **Max Drawdown**: 14.18%  
- **Win Rate**: 20.05%

---

## Prediction Output

For the latest trading day, the model outputs:

- 20-day volatility forecast
- ΔVolatility forecast
- GARCH forecast
- Price bounds for tomorrow

---

## Next Steps

- Add live data pipeline for real-time forecasts
- Host a dashboard on **Streamlit**
- Add Pine Script logic to test signals on **TradingView**

---

## Author

**Pranav Bollineni** – [LinkedIn](https://www.linkedin.com/in/bpranavb/) | 

