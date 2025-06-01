import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_finnhub_news(symbol, from_date, to_date, api_key):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": from_date,
        "to": to_date,
        "token": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        news_data = response.json()
        return pd.DataFrame(news_data)
    else:
        print(f"Error fetching news for {symbol}: {response.status_code}")
        return pd.DataFrame()

# 示例用法
api_key = "d0rdf7pr01qn4tji1c70d0rdf7pr01qn4tji1c7g"  # 替换为您的 Finnhub API 密钥
from_date = "2020-01-01"
to_date = "2024-01-05"
symbols = ["AAPL", "MSFT", "GOOGL"]

for symbol in symbols:
    news_df = fetch_finnhub_news(symbol, from_date, to_date, api_key)
    news_df.to_csv(f"{symbol}_news.csv", index=False)
