import os
import openai

# 推荐设置为环境变量
# openai.api_key = os.getenv("OPENAI_API_KEY")
proxy_url = 'http://127.0.0.1'
proxy_port = '7890'
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

api_key = ""   #openai   
api_key = ""   #deepseek_api

TRADE_AMOUNT = 10000