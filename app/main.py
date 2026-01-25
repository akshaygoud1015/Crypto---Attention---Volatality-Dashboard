import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="Crypto Attention Dashboard", layout="wide", page_icon="📈")

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

# Fetch price data
symbol = tickers[crypto]
end = date.today()
start = end - timedelta(days=days)
try:
    df = yf.download(symbol, start=start, end=end)
    df = df["Close"].rename("price").to_frame()
except Exception:
    dates = pd.date_range(start=start, end=end)
    price = np.cumsum(np.random.randn(len(dates)) * 2) + 100
    df = pd.DataFrame({"price": price}, index=dates)

# Compute returns & volatility
df["returns"] = df["price"].pct_change().fillna(0)
df["volatility"] = df["returns"].rolling(7).std().fillna(0)

# Synthetic attention series (peaks aligned with volatility)
rng = np.random.default_rng(seed=42)
base_attention = (df["volatility"] / df["volatility"].max()).fillna(0)
noise = rng.normal(scale=0.1, size=len(base_attention))
attention = (base_attention + noise).clip(0)
# inject occasional spikes
for _ in range(max(1, days // 60)):
    idx = rng.integers(0, len(attention))
    attention[idx: min(idx+3, len(attention))] += rng.uniform(1.5, 3.0)

df["attention_raw"] = attention
df["attention"] = df["attention_raw"].rolling(smoothing).mean().fillna(df["attention_raw"])

# Top metrics
latest = df.iloc[-1]
price_now = latest["price"]
chg = (df["price"].iloc[-1] / df["price"].iloc[-2] - 1) * 100 if len(df) > 1 else 0
avg_attention = df["attention"].mean()
avg_vol = df["volatility"].mean()

col1, col2, col3, col4 = st.columns([2,2,2,2])
col1.markdown("<div class='card'><h3 style='margin:0'>Price</h3><h2 style='margin:0'>${:,.2f}</h2></div>".format(price_now), unsafe_allow_html=True)
col2.markdown("<div class='card'><h3 style='margin:0'>24h Change</h3><h2 style='margin:0'>{:+.2f}%</h2></div>".format(chg), unsafe_allow_html=True)
col3.markdown("<div class='card'><h3 style='margin:0'>Avg Attention</h3><h2 style='margin:0'>{:.2f}</h2></div>".format(avg_attention), unsafe_allow_html=True)
col4.markdown("<div class='card'><h3 style='margin:0'>Avg Volatility</h3><h2 style='margin:0'>{:.4f}</h2></div>".format(avg_vol), unsafe_allow_html=True)

# Build figure with dual axis
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", mode="lines", line=dict(color="#0ea5e9", width=2), yaxis="y1", hovertemplate="%{x}<br>Price: %{y:$,.2f}<extra></extra>"))
fig.add_trace(go.Bar(x=df.index, y=df["attention"], name="Attention", marker_color="#f97316", opacity=0.6, yaxis="y2", hovertemplate="%{x}<br>Attention: %{y:.2f}<extra></extra>"))
fig.update_layout(
    yaxis=dict(title="Price (USD)", side="left", rangemode="tozero"),
    yaxis2=dict(title="Attention (scaled)", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=20, l=40, r=40),
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)

if show_table:
    st.subheader("Raw Data")
    st.dataframe(df.reset_index().rename(columns={"index": "date"}), height=300)

# Download
csv = df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name=f"{symbol}_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Tips: try adjusting smoothing and history length in the sidebar. This demo synthesizes attention from volatility when real attention data isn't available.")
