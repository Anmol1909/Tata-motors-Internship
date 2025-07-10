import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# --- Page settings ---
st.set_page_config(page_title="📈 Stock Prediction App", layout="wide")
st.title("📈 Stock Price Prediction (Linear & Random Forest)")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.text_input("Start Date", "2020-01-01")
end_date = st.sidebar.text_input("End Date", "2024-12-31")
selected_day = st.sidebar.slider("Select Days Ahead to Predict", 1, 60, 10)
model_choice = st.sidebar.selectbox("Select Prediction Model", ["Linear Regression", "Random Forest"])
apply = st.sidebar.button("Apply")

# --- Load stock data ---
@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return pd.DataFrame()

#  RSI Calculation 
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

#  Technical Indicators 
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    return df

#  Linear Regression or Random Forest Prediction 
def predict_linear_or_rf(df, model_type, selected_day):
    df = df.reset_index()
    df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    df = add_technical_indicators(df)
    df = df.dropna()

    features = ['Date_Ordinal', 'SMA_20', 'EMA_20', 'RSI']
    X = df[features]
    y = df['Close']

    model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Start from last 30 known values
    close_history = list(df['Close'].iloc[-30:])
    base_date = df['Date'].max()
    predictions = {}

    for i in range(1, selected_day + 1):
        predict_date = pd.bdate_range(start=base_date + timedelta(days=1), periods=i)[-1]
        ordinal = predict_date.toordinal()

        # Calculate technical indicators manually
        sma_20 = np.mean(close_history[-20:])
        ema_20 = pd.Series(close_history).ewm(span=20, adjust=False).mean().iloc[-1]

        delta = np.diff(close_history[-15:])
        gain = np.mean([x for x in delta if x > 0]) if any(delta > 0) else 0
        loss = np.mean([-x for x in delta if x < 0]) if any(delta < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 100

        # Predict using the model
        input_features = np.array([[ordinal, sma_20, ema_20, rsi]])
        prediction = model.predict(input_features).item()

        # Append to history and save
        close_history.append(prediction)
        predictions[predict_date.strftime('%Y-%m-%d')] = round(prediction, 2)

    return predictions

#  Main 
if apply:
    try:
        df = load_data(ticker, start_date, end_date)
        if df.empty:
            st.warning("⚠️ No data found. Check your ticker and dates.")
        else:
            df = add_technical_indicators(df)

            # Plot closing price + indicators
            st.subheader(f"{ticker} Closing Price & Indicators")
            st.line_chart(df[['Close', 'SMA_20', 'EMA_20']])

            # Plot RSI
            st.subheader("📊 RSI Indicator")
            st.line_chart(df['RSI'])

            # Model prediction
            st.subheader(f"📅 Predicted Prices using {model_choice}")
            preds = predict_linear_or_rf(df, model_choice, selected_day)

            for date, price in preds.items():
                currency_symbol = "₹" if ".NS" in ticker.upper() or ".BO" in ticker.upper() else "$"
            st.write(f"📅 **{date}** → {currency_symbol}{price}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Adjust settings and click Apply to run the prediction.")
