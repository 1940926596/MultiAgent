import tweepy
 
# Twitter API 密钥配置
api_key = '你的API_KEY'
api_secret_key = '你的API_SECRET_KEY'
access_token = '你的ACCESS_TOKEN'
access_token_secret = '你的ACCESS_TOKEN_SECRET'
 
# 认证并连接到 Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key=api_key,
                                consumer_secret=api_secret_key,
                                access_token=access_token,
                                access_token_secret=access_token_secret)
api = tweepy.API(auth)
 
# 检查连接是否成功
try:
    api.verify_credentials()
    print("成功连接到 Twitter API！")
except tweepy.TweepError as e:
    print(f"连接失败: {e}")