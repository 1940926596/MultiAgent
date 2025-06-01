import snscrape.modules.twitter as sntwitter
import pandas as pd

import os

# Set the proxy URL and port
proxy_url = 'http://127.0.0.1'
proxy_port = '7890' # !!!please replace it with your own port

# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'


query = "AAPL stock OR Apple Inc. since:2024-05-01 until:2025-05-29"
tweets = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= 100:  # 限制数量（比如最多100条）
        break
    tweets.append([tweet.date, tweet.user.username, tweet.content, tweet.url])

df = pd.DataFrame(tweets, columns=["date", "user", "content", "url"])
df.to_csv("aapl_twitter_news.csv", index=False)
print("Done! Saved to aapl_twitter_news.csv")
