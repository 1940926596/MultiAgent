import praw
import pandas as pd
import datetime as dt

import os

# Set the proxy URL and port
proxy_url = 'http://127.0.0.1'
proxy_port = '7890' # !!!please replace it with your own port

# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

reddit = praw.Reddit(
    client_id="QgVI-XaNzPPTbWd1Bw2RlQ",
    client_secret="_o54jJmQFm3fW8hhjELcsi5fmJLm6A",
    user_agent="aapl-agent"
)

subreddit = reddit.subreddit("stocks")

# 搜索最近100条提到 AAPL 的帖子
posts = []
for submission in subreddit.search("AAPL", sort="new", time_filter="week", limit=100):
    posts.append({
        "title": submission.title,
        "text": submission.selftext,
        "score": submission.score,
        "created": dt.datetime.fromtimestamp(submission.created_utc),
        "url": submission.url
    })

df = pd.DataFrame(posts)
df.to_csv("aapl_reddit_recent.csv", index=False)
print("抓取完成：", len(df), "条帖子")


# 获取
# learnpython
# 版块中的热门帖子
subreddit = reddit.subreddit('learnpython')
for post in subreddit.hot(limit=5):
    print(f"Title: {post.title}, Score: {post.score}")