import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Real-Time Stock Predictor", layout="wide")

# Constants
API_KEY = '37ed2ada4c144db496b022e22b5c8590'  # Replace with your Twelve Data API Key
BASE_URL = 'https://api.twelvedata.com/time_series'

st.title("📈 Real-Time Stock Price Predictor")

# --- Sidebar ---
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min"], index=1)
outputsize = st.sidebar.slider("Number of Data Points", 50, 500, 100)

if st.sidebar.button("Fetch & Predict"):
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY
    }

    st.info(f"Fetching data for {symbol.upper()}...")

    response = requests.get(BASE_URL, params=params).json()

    if 'values' not in response:
        st.error("Failed to fetch data. Check your API key or symbol.")
    else:
        df = pd.DataFrame(response['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df.set_index('datetime', inplace=True)
        df = df.astype(float)

        # Prepare data
        df['Target'] = df['close'].shift(-1)
        df.dropna(inplace=True)

        X = df[['close']].values
        y = df['Target'].values

        # Train/Test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.success("Prediction completed!")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test, label="Actual")
        ax.plot(predictions, label="Predicted")
        ax.set_title(f"{symbol.upper()} - Price Prediction")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Show table
        st.subheader("Latest Data")
        st.dataframe(df.tail(10))
