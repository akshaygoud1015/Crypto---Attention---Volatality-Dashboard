import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path
import joblib

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Crypto Attention Dashboard", layout="wide", page_icon="📈")

# ----------------------------------------------------
# Load ML model (robust path)
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "volatility_model.pkl"

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
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
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

df = yf.download(symbol, start=start, end=end)
df = df[["Close"]].rename(columns={"Close": "price"})

# ----------------------------------------------------
# Volatility + returns
# ----------------------------------------------------
df["return_1d"] = df["price"].pct_change().fillna(0)
df["volatility"] = df["return_1d"].rolling(7).std().fillna(0)

# ----------------------------------------------------
# Synthetic attention (placeholder)
# ----------------------------------------------------
rng = np.random.default_rng(seed=42)
base_attention = (df["volatility"] / df["volatility"].max()).fillna(0)
noise = rng.normal(scale=0.1, size=len(base_attention))
attention = (base_attention + noise).clip(0)

for _ in range(max(1, days // 60)):
    idx = rng.integers(0, len(attention))
    attention[idx:min(idx + 3, len(attention))] += rng.uniform(1.5, 3.0)

df["attention_raw"] = attention
df["attention"] = df["attention_raw"].rolling(smoothing).mean().fillna(df["attention_raw"])

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
pred_vol_7d = float(pred_vol_7d)  # already a scalar from model.predict

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
st.caption("⚠ Attention data is currently synthetic. Real search/social streams can be plugged in without changing the model interface.")
