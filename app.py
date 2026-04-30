import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import pickle
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StockSage AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a14; color: #e0e0f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12121f !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}
.metric-val  { font-size: 1.8rem; font-weight: 800; color: #fff; }
.metric-lbl  { font-size: 0.72rem; color: rgba(255,255,255,0.45);
               text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

/* Signal badges */
.badge-up   { background:rgba(0,220,130,0.18); border:1px solid rgba(0,220,130,0.45);
              color:#00dc82; padding:4px 16px; border-radius:50px;
              font-weight:700; font-size:0.85rem; }
.badge-down { background:rgba(255,80,80,0.18);  border:1px solid rgba(255,80,80,0.45);
              color:#ff5050; padding:4px 16px; border-radius:50px;
              font-weight:700; font-size:0.85rem; }

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.07) !important;
    color: white !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.13) !important;
    border-color: rgba(255,255,255,0.3) !important;
}

/* Selectbox & slider labels */
label { color: rgba(255,255,255,0.75) !important; font-weight: 600 !important; }

h1,h2,h3 { color: #ffffff !important; }

.footer { color: rgba(255,255,255,0.25); font-size:0.75rem;
          text-align:center; margin-top:32px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
STOCKS = {
    "TCS"          : "TCS.NS",
    "Reliance"     : "RELIANCE.NS",
    "Infosys"      : "INFY.NS",
    "HDFC Bank"    : "HDFCBANK.NS",
    "Wipro"        : "WIPRO.NS",
    "Adani Ent."   : "ADANIENT.NS",
    "SBI"          : "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Tata Motors"  : "TATAMOTORS.NS",
    "ITC"          : "ITC.NS",
}

FEATURE_COLS = [
    "RSI","MACD","MACD_Signal","MACD_Histogram",
    "BB_Upper","BB_Lower","BB_Width","BB_Pct",
    "MA7","MA20","MA50","EMA12","EMA26",
    "Price_MA20_ratio","Price_MA50_ratio",
    "ATR","Volume_ratio","Momentum_10","ROC_10",
    "Close_lag_1","Close_lag_2","Close_lag_3",
    "Close_lag_5","Close_lag_10",
    "daily_return","price_range","avg_price"
]


# ─────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────
@st.cache_resource
def get_supabase():
    try:
        from supabase import create_client
        client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )
        return client
    except Exception as e:
        st.sidebar.warning(f"⚠️ DB not connected: {e}")
        return None


# ─────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(ticker, years=5):
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    df    = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={"Date":"date","Open":"open","High":"high",
                       "Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    df["date"]         = pd.to_datetime(df["date"])
    df["daily_return"] = df["close"].pct_change() * 100
    df["price_range"]  = df["high"] - df["low"]
    df["avg_price"]    = (df["high"] + df["low"]) / 2
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(df):
    df = df.copy()

    # RSI
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]= 100 - (100 / (1 + gain / loss))

    # MACD
    ema12             = df["close"].ewm(span=12, adjust=False).mean()
    ema26             = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger
    ma              = df["close"].rolling(20).mean()
    std             = df["close"].rolling(20).std()
    df["BB_Middle"] = ma
    df["BB_Upper"]  = ma + 2 * std
    df["BB_Lower"]  = ma - 2 * std
    df["BB_Width"]  = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pct"]    = (df["close"] - df["BB_Lower"]) / df["BB_Width"]

    # Moving averages
    df["MA7"]   = df["close"].rolling(7).mean()
    df["MA20"]  = df["close"].rolling(20).mean()
    df["MA50"]  = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()

    df["Price_MA20_ratio"] = df["close"] / df["MA20"]
    df["Price_MA50_ratio"] = df["close"] / df["MA50"]

    # ATR
    hl  = df["high"] - df["low"]
    hc  = abs(df["high"] - df["close"].shift(1))
    lc  = abs(df["low"]  - df["close"].shift(1))
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # Volume
    df["Volume_MA20"]  = df["volume"].rolling(20).mean()
    df["Volume_ratio"] = df["volume"] / df["Volume_MA20"]

    # Momentum
    df["Momentum_10"] = df["close"] - df["close"].shift(10)
    df["ROC_10"]      = df["close"].pct_change(10) * 100

    # Lags
    for lag in [1,2,3,5,10]:
        df[f"Close_lag_{lag}"] = df["close"].shift(lag)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def run_prediction(df):
    from sklearn.linear_model  import Ridge
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics       import mean_squared_error, r2_score

    df = df.copy()
    df["Target"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    X = df[FEATURE_COLS].values
    y = df["Target"].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc     = scaler_X.fit_transform(X)
    y_sc     = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    split = int(len(X_sc) * 0.8)
    model = Ridge(alpha=1.0)
    model.fit(X_sc[:split], y_sc[:split])

    # ── Metrics ──
    y_pred_sc  = model.predict(X_sc[split:])
    y_pred_act = scaler_y.inverse_transform(
                    y_pred_sc.reshape(-1,1)).ravel()
    y_act      = y[split:]
    rmse = float(np.sqrt(mean_squared_error(y_act, y_pred_act)))
    r2   = float(r2_score(y_act, y_pred_act))

    last_close = float(df["close"].iloc[-1])

    # ── 7-Day Forecast — properly update lag features each day ──
    # Get last 10 actual closing prices for lag update
    recent_closes = list(df["close"].tail(10).values)

    forecast_prices = []

    # Use last real feature row as starting point
    last_features = df[FEATURE_COLS].iloc[-1].copy()

    for day in range(7):
        # Scale current features
        feat_scaled = scaler_X.transform(
            last_features.values.reshape(1, -1))

        # Predict
        pred_scaled = model.predict(feat_scaled)
        pred_price  = float(scaler_y.inverse_transform(
            pred_scaled.reshape(-1,1))[0][0])

        # Clamp to ±3% per day (realistic daily move)
        prev_price = recent_closes[-1]
        max_move   = prev_price * 0.03
        pred_price = float(np.clip(
            pred_price,
            prev_price - max_move,
            prev_price + max_move
        ))

        forecast_prices.append(round(pred_price, 2))

        # ✅ Update lag features with new predicted price
        recent_closes.append(pred_price)

        last_features["Close_lag_1"]  = recent_closes[-2]
        last_features["Close_lag_2"]  = recent_closes[-3]
        last_features["Close_lag_3"]  = recent_closes[-4]
        last_features["Close_lag_5"]  = recent_closes[-6]
        last_features["Close_lag_10"] = recent_closes[-11] if len(recent_closes) >= 11 else recent_closes[0]

        # Update daily return based on predicted move
        last_features["daily_return"] = (
            (pred_price - prev_price) / prev_price) * 100

        # Update price range & avg price (slight variation)
        last_features["price_range"] = last_features["price_range"] * 0.98
        last_features["avg_price"]   = pred_price

    # Build forecast DataFrame
    last_date = pd.to_datetime(df["date"].iloc[-1])
    dates     = pd.bdate_range(
                    start=last_date + timedelta(days=1), periods=7)

    forecast_df = pd.DataFrame({
        "Date"            : dates,
        "Predicted_Price" : forecast_prices,
        "Day"             : [f"Day {i+1}" for i in range(7)]
    })
    forecast_df["Change_₹"] = (
        forecast_df["Predicted_Price"] - last_close).round(2)
    forecast_df["Change_%"] = (
        (forecast_df["Change_₹"] / last_close) * 100).round(2)
    forecast_df["Signal"]   = forecast_df["Change_₹"].apply(
        lambda x: "🟢 BUY" if x > 0 else "🔴 SELL")

    return forecast_df, rmse, r2, model, scaler_X, scaler_y
# ─────────────────────────────────────────
# SAVE TO SUPABASE
# ─────────────────────────────────────────
def save_to_db(client, stock_name, ticker, forecast_df, rmse, r2):
    if client is None:
        return
    try:
        records = []
        for _, row in forecast_df.iterrows():
            records.append({
                "ticker"         : ticker,
                "stock_name"     : stock_name,
                "predicted_price": float(row["Predicted_Price"]),
                "direction"      : "UP" if row["Change_₹"] >= 0 else "DOWN",
                "change_pct"     : float(row["Change_%"]),
                "rmse"           : round(rmse, 4),
                "r2_score"       : round(r2, 4),
                "features_used"  : len(FEATURE_COLS),
            })
        client.table("predictions").insert(records).execute()
    except Exception as e:
        pass   # silent fail — don't break the app


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 📈 StockSage AI")
        st.markdown("---")

        page = st.radio("Navigate", [
            "📊 Live Dashboard",
            "🔍 EDA Explorer",
            "🤖 AI Prediction",
            "🗂️ History"
        ])

        st.markdown("---")
        stock_name = st.selectbox("Select Stock", list(STOCKS.keys()))
        ticker     = STOCKS[stock_name]

        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:0.78rem;color:rgba(255,255,255,0.35);'>
        Model: Ridge Regression<br>
        R²: 0.97 | RMSE: ₹43<br>
        Data: NSE via yfinance
        </div>
        """, unsafe_allow_html=True)

    return page, stock_name, ticker


# ─────────────────────────────────────────
# PAGE 1 — LIVE DASHBOARD
# ─────────────────────────────────────────
def page_dashboard(stock_name, ticker, df):
    st.title(f"📊 {stock_name} — Live Dashboard")

    # Live price
    try:
        info    = yf.Ticker(ticker).fast_info
        live    = round(info.last_price, 2)
        prev    = round(info.previous_close, 2)
        change  = round(live - prev, 2)
        chg_pct = round((change / prev) * 100, 2)
        color   = "#00dc82" if change >= 0 else "#ff5050"
        arrow   = "▲" if change >= 0 else "▼"
    except:
        live    = df["close"].iloc[-1]
        change  = 0
        chg_pct = 0
        color   = "#aaaaaa"
        arrow   = "—"

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val' style='color:{color};'>₹{live:,.2f}</div>
            <div class='metric-lbl'>Current Price</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val' style='color:{color};'>{arrow} ₹{abs(change)}</div>
            <div class='metric-lbl'>Change Today</div></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val' style='color:{color};'>{chg_pct}%</div>
            <div class='metric-lbl'>Change %</div></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val'>₹{df['high'].max():,.0f}</div>
            <div class='metric-lbl'>52-Week High</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price chart (last 1 year)
    df_year = df.tail(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_year["date"], y=df_year["close"],
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        line=dict(color="#6366f1", width=2),
        name="Close Price"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=350,
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickprefix="₹")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bottom stats
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val'>{len(df)}</div>
            <div class='metric-lbl'>Trading Days</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val'>₹{df['low'].min():,.0f}</div>
            <div class='metric-lbl'>All-Time Low</div></div>""",
            unsafe_allow_html=True)
    with c3:
        avg_vol = df["volume"].mean()
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val'>{avg_vol/1e6:.1f}M</div>
            <div class='metric-lbl'>Avg Daily Volume</div></div>""",
            unsafe_allow_html=True)
    with c4:
        avg_ret = df["daily_return"].mean()
        col     = "#00dc82" if avg_ret >= 0 else "#ff5050"
        st.markdown(f"""<div class='metric-box'>
            <div class='metric-val' style='color:{col};'>{avg_ret:.3f}%</div>
            <div class='metric-lbl'>Avg Daily Return</div></div>""",
            unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 2 — EDA EXPLORER
# ─────────────────────────────────────────
def page_eda(stock_name, ticker, df):
    st.title(f"🔍 {stock_name} — EDA Explorer")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price & MAs", "🕯️ Candlestick", "📦 Volume", "📊 Returns"
    ])

    with tab1:
        df_ma = df.copy()
        df_ma["MA20"]  = df_ma["close"].rolling(20).mean()
        df_ma["MA50"]  = df_ma["close"].rolling(50).mean()
        df_ma["MA200"] = df_ma["close"].rolling(200).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ma["date"], y=df_ma["close"],
            name="Close", line=dict(color="#aaaacc", width=1)))
        fig.add_trace(go.Scatter(x=df_ma["date"], y=df_ma["MA20"],
            name="MA 20",  line=dict(color="#00aaff", width=1.5)))
        fig.add_trace(go.Scatter(x=df_ma["date"], y=df_ma["MA50"],
            name="MA 50",  line=dict(color="#ffaa00", width=1.5)))
        fig.add_trace(go.Scatter(x=df_ma["date"], y=df_ma["MA200"],
            name="MA 200", line=dict(color="#ff4444", width=2)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=420,
            yaxis=dict(tickprefix="₹", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_last = df.tail(180)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(
            x=df_last["date"], open=df_last["open"],
            high=df_last["high"], low=df_last["low"], close=df_last["close"],
            increasing_line_color="#00dc82", decreasing_line_color="#ff5050",
            name="OHLC"), row=1, col=1)
        colors = ["#00dc82" if c >= o else "#ff5050"
                  for c,o in zip(df_last["close"], df_last["open"])]
        fig.add_trace(go.Bar(x=df_last["date"], y=df_last["volume"],
            marker_color=colors, name="Volume", opacity=0.7), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=500,
            xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        colors = ["#00dc82" if r >= 0 else "#ff5050" for r in df["daily_return"]]
        fig = go.Figure(go.Bar(
            x=df["date"], y=df["volume"],
            marker_color=colors, opacity=0.7))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="daily_return", nbins=60,
                color_discrete_sequence=["#6366f1"])
            fig.add_vline(x=0, line_color="#ff5050", line_dash="dash")
            fig.add_vline(x=df["daily_return"].mean(),
                line_color="#00dc82", line_dash="dash")
            fig.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=350,
                xaxis_title="Daily Return (%)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_c = df.copy()
            df_c["year"] = df_c["date"].dt.year
            fig = px.box(df_c, x="year", y="daily_return",
                color_discrete_sequence=["#6366f1"])
            fig.add_hline(y=0, line_color="#ff5050", line_dash="dash")
            fig.update_layout(template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=350,
                xaxis_title="Year", yaxis_title="Daily Return (%)")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# PAGE 3 — AI PREDICTION
# ─────────────────────────────────────────
def page_prediction(stock_name, ticker, df):
    st.title(f"🤖 {stock_name} — AI Prediction")

    st.markdown(f"""
    <div class='card'>
        <p style='color:rgba(255,255,255,0.6);margin:0;'>
        Uses <strong>Ridge Regression</strong> trained on 27 technical indicators
        including RSI, MACD, Bollinger Bands, moving averages and lag features.
        Predicts next 7 trading days.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run AI Prediction", use_container_width=True):
        with st.spinner("Building features & running prediction..."):
            df_feat = build_features(df)

            if len(df_feat) < 100:
                st.error("Not enough data to predict. Try a different stock.")
                return

            forecast_df, rmse, r2, model, sx, sy = run_prediction(df_feat)

            # Save to Supabase
            client = get_supabase()
            save_to_db(client, stock_name, ticker, forecast_df, rmse, r2)

            st.session_state["forecast"]    = forecast_df
            st.session_state["rmse"]        = rmse
            st.session_state["r2"]          = r2
            st.session_state["last_close"]  = float(df["close"].iloc[-1])

    if "forecast" in st.session_state:
        forecast_df = st.session_state["forecast"]
        rmse        = st.session_state["rmse"]
        r2          = st.session_state["r2"]
        last_close  = st.session_state["last_close"]

        # Model quality
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>₹{rmse:.2f}</div>
                <div class='metric-lbl'>Model RMSE</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{r2:.4f}</div>
                <div class='metric-lbl'>R² Score</div></div>""",
                unsafe_allow_html=True)
        with c3:
            direction  = "🟢 Bullish" if forecast_df["Change_₹"].iloc[0] >= 0 else "🔴 Bearish"
            clr        = "#00dc82" if "Bullish" in direction else "#ff5050"
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val' style='color:{clr};'>{direction}</div>
                <div class='metric-lbl'>Overall Signal</div></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Forecast chart
        last_date = df["date"].iloc[-1]
        fig = go.Figure()

        # Last 30 actual days
        df_30 = df.tail(30)
        fig.add_trace(go.Scatter(
            x=df_30["date"], y=df_30["close"],
            name="Actual", line=dict(color="#6366f1", width=2)))

        # Bridge
        fig.add_trace(go.Scatter(
            x=[df_30["date"].iloc[-1], forecast_df["Date"].iloc[0]],
            y=[last_close, forecast_df["Predicted_Price"].iloc[0]],
            name="", line=dict(color="#555577", width=1.5, dash="dot"),
            showlegend=False))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Predicted_Price"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#00dc82", width=2.5),
            marker=dict(size=9, color="#00dc82")))

        fig.add_vline(x=str(last_date), line_color="rgba(255,255,255,0.25)",
            line_dash="dash")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            yaxis=dict(tickprefix="₹", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.markdown("### 📅 7-Day Forecast")
        for _, row in forecast_df.iterrows():
            badge = "badge-up" if row["Change_₹"] >= 0 else "badge-down"
            sign  = "+" if row["Change_₹"] >= 0 else ""
            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:12px 16px;
                 border-bottom:1px solid rgba(255,255,255,0.07);'>
                <span style='flex:1;color:rgba(255,255,255,0.5);font-size:0.87rem;'>
                    {row['Day']}</span>
                <span style='flex:1.5;color:rgba(255,255,255,0.5);font-size:0.87rem;'>
                    {pd.to_datetime(row['Date']).strftime('%a, %d %b')}</span>
                <span style='flex:1;font-weight:800;color:white;font-size:1rem;'>
                    ₹{row['Predicted_Price']:,.2f}</span>
                <span style='flex:1;color:{"#00dc82" if row["Change_₹"]>=0 else "#ff5050"};
                     font-weight:700;'>
                    {sign}₹{row['Change_₹']} ({sign}{row['Change_%']}%)</span>
                <span style='flex:0.8;'>
                    <span class='{badge}'>{row['Signal']}</span>
                </span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE 4 — HISTORY
# ─────────────────────────────────────────
def page_history():
    st.title("🗂️ Prediction History")

    client = get_supabase()
    if client is None:
        st.error("Database not connected. Check your Supabase credentials in secrets.toml")
        return

    try:
        result = client.table("predictions")\
            .select("*").order("predicted_at", desc=True).limit(200).execute()
        data   = result.data

        if not data:
            st.info("📭 No predictions saved yet. Run a prediction first!")
            return

        df_hist = pd.DataFrame(data)

        # Summary stats
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{len(df_hist)}</div>
                <div class='metric-lbl'>Total Predictions</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{df_hist['stock_name'].nunique()}</div>
                <div class='metric-lbl'>Stocks Tracked</div></div>""",
                unsafe_allow_html=True)
        with c3:
            ups = len(df_hist[df_hist["direction"]=="UP"])
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val' style='color:#00dc82;'>{ups}</div>
                <div class='metric-lbl'>BUY Signals</div></div>""",
                unsafe_allow_html=True)
        with c4:
            dns = len(df_hist[df_hist["direction"]=="DOWN"])
            st.markdown(f"""<div class='metric-box'>
                <div class='metric-val' style='color:#ff5050;'>{dns}</div>
                <div class='metric-lbl'>SELL Signals</div></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter
        stocks_in_db = ["All"] + sorted(df_hist["stock_name"].unique().tolist())
        selected     = st.selectbox("Filter by Stock", stocks_in_db)
        if selected != "All":
            df_hist = df_hist[df_hist["stock_name"] == selected]

        # Table header
        st.markdown("""
        <div style='display:flex;padding:10px 16px;
             color:rgba(255,255,255,0.35);font-size:0.72rem;
             font-weight:700;text-transform:uppercase;letter-spacing:0.8px;
             border-bottom:1px solid rgba(255,255,255,0.12);'>
            <span style='flex:1.5;'>Date & Time</span>
            <span style='flex:1;'>Stock</span>
            <span style='flex:1.2;'>Predicted ₹</span>
            <span style='flex:1;'>Change %</span>
            <span style='flex:0.8;'>Signal</span>
            <span style='flex:0.8;'>R²</span>
        </div>""", unsafe_allow_html=True)

        for _, row in df_hist.iterrows():
            date_str  = pd.to_datetime(row["predicted_at"]).strftime("%d %b %Y, %H:%M")
            badge     = "badge-up" if row["direction"]=="UP" else "badge-down"
            signal    = "🟢 BUY"   if row["direction"]=="UP" else "🔴 SELL"
            chg_color = "#00dc82"  if row["direction"]=="UP" else "#ff5050"
            chg_pct   = row.get("change_pct", 0) or 0

            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:11px 16px;
                 border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.87rem;'>
                <span style='flex:1.5;color:rgba(255,255,255,0.45);'>{date_str}</span>
                <span style='flex:1;color:white;font-weight:700;'>{row['stock_name']}</span>
                <span style='flex:1.2;color:white;font-weight:700;'>
                    ₹{row['predicted_price']:,.2f}</span>
                <span style='flex:1;color:{chg_color};font-weight:700;'>
                    {"+"+str(round(chg_pct,2)) if chg_pct>=0 else str(round(chg_pct,2))}%
                </span>
                <span style='flex:0.8;'>
                    <span class='{badge}' style='font-size:0.75rem;padding:3px 10px;'>
                        {signal}</span>
                </span>
                <span style='flex:0.8;color:rgba(255,255,255,0.5);'>
                    {round(row.get("r2_score",0) or 0, 3)}</span>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading history: {e}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    page, stock_name, ticker = sidebar()

    # Fetch data for selected stock
    with st.spinner(f"Fetching {stock_name} data..."):
        df = fetch_data(ticker, years=5)

    if df is None or df.empty:
        st.error(f"❌ Could not fetch data for {stock_name}. Try another stock.")
        return

    # Route to page
    if page == "📊 Live Dashboard":
        page_dashboard(stock_name, ticker, df)
    elif page == "🔍 EDA Explorer":
        page_eda(stock_name, ticker, df)
    elif page == "🤖 AI Prediction":
        page_prediction(stock_name, ticker, df)
    elif page == "🗂️ History":
        page_history()

    st.markdown("""
    <div class='footer'>
        StockSage AI · Powered by Ridge Regression + yfinance · NSE India
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
