import requests
import pandas as pd
from datetime import datetime

import os

# Set the proxy URL and port
proxy_url = 'http://127.0.0.1'
proxy_port = '7890' # !!!please replace it with your own port

# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'


def get_stocktwits_messages(symbol="AAPL", limit=30):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    messages = data.get("messages", [])[:limit]

    result = []
    for msg in messages:
        sentiment_info = msg.get("entities", {}).get("sentiment")
        sentiment = sentiment_info.get("basic") if sentiment_info else "None"

        result.append({
            "id": msg.get("id"),
            "created_at": msg.get("created_at"),
            "user": msg["user"]["username"],
            "body": msg.get("body"),
            "likes": msg.get("likes", {}).get("total", 0),
            "sentiment": sentiment
        })

    df = pd.DataFrame(result)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

# 示例调用
df_aapl = get_stocktwits_messages("AAPL", limit=1000)
df_aapl.to_csv("aapl_stocktwits.csv", index=False)
print("获取成功，共", len(df_aapl), "条消息")