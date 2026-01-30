from pytrends.request import TrendReq
import pandas as pd

def fetch_google_trends(keyword, days=30):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], timeframe=f'today {days}-d')
    df = pytrends.interest_over_time()

    if df.empty:
        return None

    df = df.drop(columns=["isPartial"], errors="ignore")
    df.columns = ["attention"]
    df.index.name = "date"
    return df
