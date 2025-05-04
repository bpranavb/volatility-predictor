import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from arch import arch_model
import ta
from tiingo import TiingoClient
from itertools import product

st.set_page_config(page_title="Volatility Strategy Dashboard", layout="wide")
st.title("Volatility Prediction & Strategy Simulator")

ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
if st.button("Run Strategy"):
    with st.spinner("Running full pipeline..."):

        config = {'session': True, 'api_key': '3d01421bfd27eb6d30e7c7df881b96ecec93c27f'}
        client = TiingoClient(config)
        end_date = dt.datetime.today().date()
        start_date = end_date - dt.timedelta(days=365 * 11)

        def fetch(symbol):
            df = client.get_dataframe(symbol, startDate=start_date.strftime('%Y-%m-%d'), endDate=end_date.strftime('%Y-%m-%d'))
            df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
            df.index = df.index.tz_localize(None)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        data = fetch(ticker)
        vix = fetch("VIXY")[['Close']].rename(columns={'Close': 'VIX_Close'})
        spy = fetch("SPY")[['Close']].rename(columns={'Close': 'SPY_Close'})
        data = data.merge(vix, left_index=True, right_index=True).merge(spy, left_index=True, right_index=True)
        data.ffill(inplace=True); data.bfill(inplace=True); data.dropna(inplace=True)

        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(20).std()
        data['Volatility_Delta_5d'] = data['Volatility'].pct_change(5)
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], 14).rsi()
        bb = ta.volatility.BollingerBands(data['Close'], 20, 2)
        data['BB_width'] = bb.bollinger_hband() - bb.bollinger_lband()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], 14).average_true_range()
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd(); data['MACD_diff'] = macd.macd_diff()
        data.dropna(inplace=True)

        clean_data = data.iloc[-2520:].copy()
        features = ['MA20','MA50','MA100','RSI','BB_width','ATR','MACD','MACD_diff','VIX_Close','SPY_Close']
        x, y = clean_data[features], clean_data['Volatility']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        xgb = XGBRegressor(random_state=42)
        grid = GridSearchCV(xgb, {'n_estimators':[50], 'learning_rate':[0.1], 'max_depth':[3]}, cv=3, scoring='neg_mean_absolute_error')
        grid.fit(x_train, y_train)
        model = grid.best_estimator_
        preds = model.predict(x_test)

        y_d = clean_data['Volatility_Delta_5d']
        x_tr_d, x_te_d, y_tr_d, y_te_d = train_test_split(x, y_d, test_size=0.2)
        grid.fit(x_tr_d, y_tr_d)
        model_d = grid.best_estimator_
        delta_preds = model_d.predict(x_te_d)

        returns = clean_data['Returns'].dropna() * 100
        garch_result = arch_model(returns, vol='GARCH', p=1, q=1).fit(disp='off')
        garch_series = pd.Series(np.sqrt(garch_result.forecast(start=clean_data.index[0]).variance.values.flatten()) / 100, index=clean_data.index)

        latest = pd.DataFrame([clean_data[features].iloc[-1]], columns=features)
        pred_vol = model.predict(latest)[0]
        pred_delta = model_d.predict(latest)[0]
        last_price = clean_data['Close'].iloc[-1]
        upper = last_price * (1 + pred_vol); lower = last_price * (1 - pred_vol)

        st.subheader("Forecast for Tomorrow")
        st.write(f"**20d Volatility:** `{round(pred_vol,5)}` | **ΔVol (5d):** `{round(pred_delta,5)}` | **GARCH:**     `{round(garch_series.iloc[-1],5)}`")
        st.write(f"📈 **Price Range:** `${round(lower,2)} - ${round(upper,2)}`")

        def calculate_metrics(returns):
            cumulative = (1 + returns).cumprod()
            total = cumulative.iloc[-1] - 1
            annual = (1 + total)**(252/len(returns)) - 1
            vol = returns.std() * np.sqrt(252)
            sharpe = annual / vol if vol > 0 else 0
            drawdown = 1 - cumulative.div(cumulative.cummax()).min()
            winrate = (returns > 0).mean()
            return {
                'Total Return': total,
                'Annualized Return': annual,
                'Volatility': vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': drawdown,
                'Win Rate': winrate
            }

        def simulate(delta_thresh, garch_thresh):
            d = clean_data.copy()
            d['ML'] = model.predict(d[features])
            d['ΔVol'] = model_d.predict(d[features])
            d['GARCH'] = garch_series.reindex(d.index).ffill()
            d['Momentum'] = d['Close'].pct_change(3)
            d['Signal'] = np.where(
                (abs(d['ΔVol']) > delta_thresh) & ((d['GARCH'] - d['ML']) > garch_thresh) & (d['Momentum'] > 0),
                'BUY', 'HOLD'
            )
            d['Next_Close'] = d['Close'].shift(-1)
            d['Return'] = d['Next_Close'] / d['Close'] - 1
            d['Strategy'] = np.where(d['Signal'] == 'BUY', d['Return'], 0)
        
            # Handle case: no trades made
            if (d['Signal'] == 'BUY').sum() == 0:
                d['Strategy'] = 0.0  # Make sure Strategy returns aren't NaN
                metrics = {
                    'Total Return': 0.0,
                    'Annualized Return': 0.0,
                    'Volatility': d['Return'].std() * np.sqrt(252),
                    'Sharpe Ratio': 0.0,
                    'Max Drawdown': 1 - (1 + d['Return']).cumprod().div((1 + d['Return']).cumprod().cummax()).min(),
                    'Win Rate': 0.0
                }
            else:
                metrics = calculate_metrics(d['Strategy'])
        
            d['Cumulative Market'] = (1 + d['Return']).cumprod()
            d['Cumulative Strategy'] = (1 + d['Strategy']).cumprod()
            return metrics, d


        best_delta, best_garch = 0.005, 0.0015
        metrics, final_data = simulate(best_delta, best_garch)

        st.subheader("Strategy vs Market")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(final_data.index, final_data['Cumulative Market'], label='Market')
        ax.plot(final_data.index, final_data['Cumulative Strategy'], label='Strategy')
        ax.legend(); ax.grid(True); ax.set_title("Backtest: Strategy vs Market")
        st.pyplot(fig)

        st.subheader("Final Strategy Metrics")
        for k, v in metrics.items():
            st.markdown(f"**{k}:** `{v:.2%}`")
        st.markdown(f"Trades Executed: {(final_data['Signal'] == 'BUY').sum()}")


        # === Plots ===
        st.subheader("Volatility Prediction")
        fig1, ax1 = plt.subplots()
        ax1.plot(y_test.sort_index(), label="Actual Vol")
        ax1.plot(pd.Series(preds, index=x_test.index).sort_index(), label="Predicted Vol")
        ax1.legend(); ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("ΔVolatility Prediction")
        fig2, ax2 = plt.subplots()
        ax2.plot(y_te_d.sort_index(), label="Actual ΔVol")
        ax2.plot(pd.Series(delta_preds, index=x_te_d.index).sort_index(), label="Predicted ΔVol")
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.legend(); ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Scatter Plot: Volatility")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test[:100], preds[:100], alpha=0.6)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax3.set_xlabel("Actual"); ax3.set_ylabel("Predicted")
        st.pyplot(fig3)

        st.subheader("XGBoost Feature Importance")
        importances = model.feature_importances_
        fig4, ax4 = plt.subplots()
        ax4.bar(range(len(importances)), importances)
        ax4.set_xticks(range(len(features)))
        ax4.set_xticklabels(features, rotation=45)
        st.pyplot(fig4)
