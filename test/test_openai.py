from openai import OpenAI
import os

# Set the proxy URL and port
proxy_url = 'http://127.0.0.1'
proxy_port = '7890' # !!!please replace it with your own port

# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

client = OpenAI(api_key = "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")
response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with unstructured data, and your task is to parse it into CSV format."
    },
    {
      "role": "user",
      "content": "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them."
    }
  ],
  temperature=0.7,
  max_tokens=64,
  top_p=1
)

print(response.choices[0].message.content)
