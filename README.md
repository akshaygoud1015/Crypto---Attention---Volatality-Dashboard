# Crypto Attention–Driven Volatility

A compact Streamlit dashboard and training script that explores how public attention (searches, pageviews) correlates with short-term crypto price volatility. The project demonstrates data fetching, simple feature engineering, a RandomForest model for 7-day volatility forecasting, and a polished Streamlit UI.

## Features
- Streamlit dashboard with responsive layout and Plotly visualizations
- Live price data from Yahoo Finance (`yfinance`)
- Attention signals from Wikipedia Pageviews API (cached) with safe fallbacks
- Trained RandomForest model to predict 7-day future volatility (`models/volatility_model.pkl`)
- CSV download, metrics cards, and risk interpretation

## Repo structure

- `app/main.py` — Streamlit dashboard and inference
- `models/train_volatality_model.py` — data prep and model training script
- `models/volatility_model.pkl` — trained model (binary)
- `data/attention_cache/` — cached attention CSVs (created at runtime)
- `requirements.txt` — Python dependencies
- `Dockerfile` — Docker image for running the Streamlit app

## Quick start

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:

```bash
streamlit run app/main.py
```

4. Train or re-train the model (optional):

```bash
python models/train_volatality_model.py
```

### Docker

A `Dockerfile` is included at the repository root for building a containerized image of the Streamlit app.

Build the image locally:

```bash
docker build -t crypto-attention-volatility .
```

Run the container (maps Streamlit default port `8501`):

```bash
docker run --rm -p 8501:8501 crypto-attention-volatility
```

If you prefer to run the app without Docker, follow the virtual environment steps above.

## Notes & behavior
- The app attempts to fetch attention data (Wikipedia pageviews). If the external call fails or returns empty results, a neutral fallback is used and a warning appears in the UI.
- Price data is fetched from Yahoo Finance via `yfinance` — network failures show an error and prevent prediction.
- Cached attention files are stored under `data/attention_cache/` to reduce repeated API calls.



## License
This project is provided as-is for educational purposes. Add a license file if you plan to open-source it.

----
If you'd like, I can also:
- Add a short CONTRIBUTING.md with development steps, or
- Create a basic unit test that validates the model file loads and predicts on a dummy row.

