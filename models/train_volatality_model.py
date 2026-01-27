import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Download historical data
# -----------------------------
symbol = "BTC-USD"
end = date.today()
start = end - timedelta(days=1200)  # ~3+ years

df = yf.download(symbol, start=start, end=end)
# ensure we get a single-column DataFrame named 'price' — be robust if 'Close' is a DataFrame
if "Close" in df.columns:
    series = df["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df = series.to_frame(name="price")
else:
    raise RuntimeError("Downloaded data does not contain 'Close' column")

# -----------------------------
# Feature engineering
# -----------------------------
df["return_1d"] = df["price"].pct_change()
df["volatility"] = df["return_1d"].rolling(7).std()

# Synthetic attention (same logic as app)
rng = np.random.default_rng(seed=42)
base_attention = (df["volatility"] / df["volatility"].max())
noise = rng.normal(scale=0.1, size=len(base_attention))
df["attention"] = (base_attention + noise).clip(0)

df["attention_change"] = df["attention"].diff()
df["vol_rolling_7"] = df["return_1d"].rolling(7).std()

# -----------------------------
# Target: future 7-day volatility
# -----------------------------
df["vol_future_7d"] = (
    df["return_1d"]
    .rolling(7)
    .std()
    .shift(-7)
)

df = df.dropna()

features = [
    "attention",
    "attention_change",
    "volatility",
    "vol_rolling_7",
    "return_1d",
]

X = df[features]
y = df["vol_future_7d"]

# Time-aware split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"Model trained successfully")
print(f"MAE: {mae:.6f}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "volatility_model.pkl")
print("Saved volatility_model.pkl")
