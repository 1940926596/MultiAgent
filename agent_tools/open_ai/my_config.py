import os
import openai

# 推荐设置为环境变量
# openai.api_key = os.getenv("OPENAI_API_KEY")
proxy_url = 'http://127.0.0.1'
proxy_port = '7890'
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

# api_key = "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"
api_key = "sk-c9faecfe9d114fcb8f1e8fd871115529"

TRADE_AMOUNT = 10000