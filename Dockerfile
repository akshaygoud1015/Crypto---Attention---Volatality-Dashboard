FROM python:3.13.9

WORKDIR /app

copy requirements.txt .

run pip install --no-cache-dir -r requirements.txt

copy app ./app

copy models ./models

expose 8501

cmd ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

