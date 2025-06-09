import os
import sys
import pandas as pd
import re
import ast

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent

# 你需要先定义一个针对新闻情绪总结的 function schema，比如 news_sentiment_summary_function
from agent_tools.data_handler.news_sentiment.news_sentiment_summary_function import news_sentiment_summary_function  


class NewsSentimentAgent(BaseFinanceAgent):
    def __init__(self, name="NewsSentimentAgent", role="Summarize daily news and sentiment trends.",
                 model="gpt-4", memory_size=8):
        super().__init__(name=name, role=role, model=model,
                         function_schema=[news_sentiment_summary_function],
                         memory_size=memory_size)
        self.interested_fields = ["date", "news_texts", "sentiments"]

def parse_news_and_sentiments(news_field_str):
    """
    解析你的原始news字段，拆出news_texts列表和sentiments字典列表
    """
    news_texts = []
    sentiments = []
    pattern = r"- (\[\{.*?\}\]) (.+?)(?=\n- |\Z)"
    matches = re.findall(pattern, news_field_str, flags=re.DOTALL)

    for sent_str, text in matches:
        try:
            sentiment_list = ast.literal_eval(sent_str)
            sentiment_dict = sentiment_list[0] if isinstance(sentiment_list, list) and sentiment_list else {}
        except:
            sentiment_dict = {}
        news_texts.append(text.strip())
        sentiments.append(sentiment_dict)

    return news_texts, sentiments

# 读原始csv
csv_path = '../../../datasets/processed/financial_final.csv'
df = pd.read_csv(csv_path)

# 按日期合并所有news字段
grouped = df.groupby('date').agg({
    'news': lambda x: ' '.join(x.dropna().astype(str))
}).reset_index()

# 解析拆分
grouped['news_texts'] = None
grouped['sentiments'] = None
for idx, row in grouped.iterrows():
    news_texts, sentiments = parse_news_and_sentiments(row['news'])
    grouped.at[idx, 'news_texts'] = news_texts
    grouped.at[idx, 'sentiments'] = sentiments

news_agent = NewsSentimentAgent()

results = []

for idx, row in grouped.iterrows():
    if not row['news_texts']:
        print(f"Skipping date {row['date']} due to no news.")
        continue

    # 把新闻和情绪内容合成prompt上下文
    news_and_sentiment_strs = []
    for text, sent in zip(row['news_texts'], row['sentiments']):
        sent_summary = f"polarity: {sent.get('polarity', 'N/A')}, pos: {sent.get('pos', 'N/A')}, neg: {sent.get('neg', 'N/A')}, neu: {sent.get('neu', 'N/A')}"
        news_and_sentiment_strs.append(f"News: {text}\nSentiment: {sent_summary}")

        print(text)
        print(sent)

    full_context = "\n\n".join(news_and_sentiment_strs)
    print(full_context)

    prompt = (
        f"Date: {row['date']}\n"
        "You are provided with a list of news headlines and their sentiment scores for this day.\n"
        "Please summarize the main points, overall sentiment trend, and any notable insights.\n"
        "News and sentiments:\n"
        f"{full_context}\n"
        "Respond by calling the 'news_sentiment_summary' function according to the schema."
    )

    try:
        # 传prompt调用模型
        result = news_agent.ask_model(prompt)

        results.append({
            "date": row["date"],
            "news_summary": result.get("news_summary"),
            "overall_sentiment": result.get("overall_sentiment"),
            "key_points": result.get("key_points")
        })
    except Exception as e:
        print(f"[ERROR] Failed at date {row['date']}: {e}")

# 保存结果
result_df = pd.DataFrame(results)
result_df.to_csv('daily_news_sentiment_summary.csv', index=False)
print("Saved daily news sentiment summaries to daily_news_sentiment_summary.csv")
