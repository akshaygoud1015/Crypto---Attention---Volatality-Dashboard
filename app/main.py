import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path
import joblib
import requests

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Crypto Attention Dashboard", layout="wide", page_icon="📈")

# ----------------------------------------------------
# Load ML model (robust path)
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "volatility_model.pkl"


@st.cache_data(ttl=60 * 60)  # cache for 1 hour
def fetch_wikipedia_attention(page, start, end):
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/"
        f"per-article/en.wikipedia/all-access/user/{page}/daily/"
        f"{start_str}/{end_str}"
    )

    try:
        # Add User-Agent header to avoid 403
        headers = {
            'User-Agent': 'CryptoAttentionDashboard/1.0 (Educational Project)'
        }
        
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            st.warning(f"📄 Page: {page}")
            st.warning(f"❌ Wikipedia API returned status {r.status_code}")
            st.code(f"URL: {url}")
            return pd.DataFrame()

        data = r.json().get("items", [])
        if not data:
            st.warning("⚠️ No Wikipedia attention data found")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        st.success(f"✅ Fetched {len(df)} days of Wikipedia data for {page}")
        
        df["date"] = pd.to_datetime(df["timestamp"].str[:8])
        df = df.set_index("date")[["views"]]
        df.rename(columns={"views": "attention_raw"}, inplace=True)

        return df
    except Exception as e:
        st.error(f"❌ Error fetching Wikipedia data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------------------------------
# Header + styles
# ----------------------------------------------------
st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#0f172a,#0ea5e9); padding:18px; border-radius:10px}
    .header h1{color:white; margin:0; font-family: 'Helvetica Neue', Arial}
    .card {background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); padding:14px; border-radius:12px}
    .small {color:#94a3b8; font-size:13px}
    </style>
    <div class="header">
      <h1>📊 Crypto Attention–Driven Volatility Dashboard</h1>
    </div>
    <p class="small">Explore how attention spikes (search & social buzz) relate to price moves and volatility.</p>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Sidebar controls
# ----------------------------------------------------
tickers = {
    "Bitcoin (BTC)": "BTC-USD",  # ✅ FIXED
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
}

WIKI_PAGES = {
    "Bitcoin (BTC)": "Bitcoin",
    "Ethereum (ETH)": "Ethereum",
    "Solana (SOL)": "Solana_(blockchain_platform)",
}

with st.sidebar:
    st.header("Controls")
    crypto = st.selectbox("Cryptocurrency", list(tickers.keys()))
    days = st.slider("Days of history", min_value=30, max_value=720, value=180, step=30)
    smoothing = st.slider("Attention smoothing (days)", min_value=1, max_value=30, value=5)
    show_table = st.checkbox("Show raw data table", value=False)

st.write(f"**Showing analysis for {crypto} — last {days} days**")

# ----------------------------------------------------
# Fetch price data
# ----------------------------------------------------
symbol = tickers[crypto]

end = date.today()
start = end - timedelta(days=days)

data = yf.download(symbol, start=start, end=end, progress=False)

# Check if data was downloaded
if data.empty:
    st.error(f"❌ Failed to download price data for {symbol}")
    st.stop()

# Safely extract Close price
try:
    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
except KeyError:
    close_series = data.xs('Close', axis=1, level=0).iloc[:, 0]

df = pd.DataFrame({'price': close_series}, index=data.index)

st.info(f"📊 Loaded {len(df)} days of price data")

# ----------------------------------------------------
# Volatility + returns
# ----------------------------------------------------
df["return_1d"] = df["price"].pct_change().fillna(0)
df["volatility"] = df["return_1d"].rolling(7).std().fillna(0)

# ----------------------------------------------------
# Wikipedia attention (REAL)
# ----------------------------------------------------
wiki_page = WIKI_PAGES[crypto]  # ✅ FIXED - was tickers[crypto]
att_df = fetch_wikipedia_attention(wiki_page, start, end)

# Ensure attention_raw column exists
if not att_df.empty and "attention_raw" in att_df.columns:
    df = df.join(att_df, how="left")
    df["attention_raw"] = df["attention_raw"].ffill().fillna(0)
    st.success(f"✅ Joined Wikipedia attention data")
else:
    st.info("ℹ️ Wikipedia attention data not available. Using synthetic data.")
    df["attention_raw"] = 0.0

# Normalize attention (important!)
max_att = df["attention_raw"].max()
if max_att > 0:
    df["attention_raw"] = df["attention_raw"] / max_att

# Smooth attention
df["attention"] = (
    df["attention_raw"]
    .rolling(smoothing)
    .mean()
    .fillna(df["attention_raw"])
)

# ----------------------------------------------------
# 🔮 STEP 5: Prediction features
# ----------------------------------------------------
df["attention_change"] = df["attention"].diff().fillna(0)
df["vol_rolling_7"] = df["return_1d"].rolling(7).std().fillna(0)

feature_cols = [
    "attention",
    "attention_change",
    "volatility",
    "vol_rolling_7",
    "return_1d",
]

# Debug: Check if we have data


# Make sure we have at least one row
if len(df) == 0:
    st.error("❌ No data available for prediction")
    st.stop()

latest_features = df[feature_cols].iloc[-1:].values



# ----------------------------------------------------
# 🔮 STEP 6: Predict 7-day future volatility
# ----------------------------------------------------
pred_vol_7d = model.predict(latest_features)[0]

# ----------------------------------------------------
# Top metrics
# ----------------------------------------------------
latest = df.iloc[-1]

# Force scalars to float to avoid Series formatting issues
price_now = float(latest["price"])
chg = float((df["price"].iloc[-1] / df["price"].iloc[-2] - 1) * 100) if len(df) > 1 else 0.0
vol_now = float(latest["volatility"])
pred_vol_7d = float(pred_vol_7d)

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='card'><h3>Price</h3><h2>${price_now:,.2f}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><h3>24h Change</h3><h2>{chg:+.2f}%</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><h3>Current Volatility</h3><h2>{vol_now:.4f}</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='card'><h3>Predicted 7-Day Vol</h3><h2>{pred_vol_7d:.4f}</h2></div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 🔮 STEP 7: Risk interpretation
# ----------------------------------------------------
low_q = df["volatility"].quantile(0.33)
high_q = df["volatility"].quantile(0.66)

risk_level = (
    "Low" if pred_vol_7d < low_q
    else "High" if pred_vol_7d > high_q
    else "Medium"
)

st.subheader("🔮 7-Day Volatility Forecast")
st.metric(
    label="Risk Regime",
    value=risk_level,
    delta="Attention-driven signal"
)

st.caption(
    "Prediction is generated using recent attention dynamics, returns, and rolling volatility patterns."
)

# ----------------------------------------------------
# Chart
# ----------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", yaxis="y1"))
fig.add_trace(go.Bar(x=df.index, y=df["attention"], name="Attention", yaxis="y2", opacity=0.6))

fig.update_layout(
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(title="Attention", overlaying="y", side="right", showgrid=False),
    template="plotly_dark",
    margin=dict(t=30, b=30),
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Optional table + download
# ----------------------------------------------------
if show_table:
    st.dataframe(df.reset_index().rename(columns={"index": "date"}), height=300)

csv = df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv")

st.markdown("---")
st.caption("⚠ Attention data from Wikipedia Pageviews API. More search/social streams will be plugged in.")