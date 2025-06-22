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