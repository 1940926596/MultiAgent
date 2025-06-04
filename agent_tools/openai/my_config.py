import os
import openai

# 推荐设置为环境变量
# openai.api_key = os.getenv("OPENAI_API_KEY")
proxy_url = 'http://127.0.0.1'
proxy_port = '7890'
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

api_key = "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"

TRADE_AMOUNT = 10000