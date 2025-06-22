# import os
# import sys
# import pandas as pd
# import ast
# import re

# # 添加项目路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent
# from agent_tools.data_handler.news_sentiment.news_sentiment_summary_function import news_sentiment_summary_function


# class NewsSentimentAgent(BaseFinanceAgent):
#     def __init__(self, name="NewsSentimentAgent", role="Summarize daily news and sentiment trends.",
#                 memory_size=8):
#         super().__init__(name=name, role=role,
#                          function_schema=[news_sentiment_summary_function],
#                          memory_size=memory_size)


# def parse_news_and_sentiments(news_field_str):
#     news_texts, sentiments = [], []
#     pattern = r"- (\[\{.*?\}\]) (.+?)(?=\n- |\Z)"
#     matches = re.findall(pattern, str(news_field_str), flags=re.DOTALL)
#     for sent_str, text in matches:
#         try:
#             sentiment_list = ast.literal_eval(sent_str)
#             sentiment_dict = sentiment_list[0] if isinstance(sentiment_list, list) and sentiment_list else {}
#         except:
#             sentiment_dict = {}
#         news_texts.append(text.strip())
#         sentiments.append(sentiment_dict)
#     return news_texts, sentiments


# def process_sentiment_for_stock(stock_dir):
#     merged_input_path = os.path.join(stock_dir, "financial_with_macro_summary.csv")
#     if not os.path.exists(merged_input_path):
#         print(f"[SKIP] {stock_dir}: No financial_with_macro_summary.csv found.")
#         return

#     print(f"[INFO] Processing sentiment for {stock_dir}")
#     df = pd.read_csv(merged_input_path)
#     df['date'] = pd.to_datetime(df['date'])

#     # 聚合新闻内容
#     grouped = df.groupby('date').agg({
#         'news': lambda x: ' '.join(x.dropna().astype(str))
#     }).reset_index()

#     grouped['news_texts'], grouped['sentiments'] = None, None
#     for idx, row in grouped.iterrows():
#         news_texts, sentiments = parse_news_and_sentiments(row['news'])
#         grouped.at[idx, 'news_texts'] = news_texts
#         grouped.at[idx, 'sentiments'] = sentiments

#     # 分析新闻情绪
#     agent = NewsSentimentAgent()
#     results = []
#     for idx, row in grouped.iterrows():
#         if not row['news_texts']:
#             continue
#         try:
#             context_blocks = []
#             for text, sent in zip(row['news_texts'], row['sentiments']):
#                 sent_summary = f"polarity: {sent.get('polarity', 'N/A')}, pos: {sent.get('pos', 'N/A')}, " \
#                                f"neg: {sent.get('neg', 'N/A')}, neu: {sent.get('neu', 'N/A')}"
#                 context_blocks.append(f"News: {text}\nSentiment: {sent_summary}")
#             full_context = "\n\n".join(context_blocks)
#             prompt = (
#                 f"Date: {row['date']}\n"
#                 "You are provided with news headlines and sentiment scores. "
#                 "Summarize key points, overall sentiment trend, and any notable insights.\n"
#                 f"{full_context}\n"
#                 "Respond using the 'news_sentiment_summary' function schema."
#             )
#             result = agent.ask_model(prompt)
#             results.append({
#                 "date": row["date"],
#                 "news_summary": result.get("news_summary"),
#                 "overall_sentiment": result.get("overall_sentiment"),
#                 "key_points": result.get("key_points")
#             })
#         except Exception as e:
#             print(f"[ERROR] Sentiment failed at {row['date']} in {stock_dir}: {e}")

#         print(results)

#     sentiment_df = pd.DataFrame(results)
#     sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
#     sentiment_csv = os.path.join(stock_dir, "daily_news_sentiment_summary.csv")
#     sentiment_df.to_csv(sentiment_csv, index=False)
#     print(f"[OK] Saved: {sentiment_csv}")

#     # 合并到最终文件
#     try:
#         df_merged = df.merge(sentiment_df, on="date", how="left")
#         for col in ['news_summary', 'overall_sentiment', 'key_points']:
#             df_merged[col] = df_merged[col].fillna('')
#         final_path = os.path.join(stock_dir, "financial_with_news_macro_summary.csv")
#         df_merged.to_csv(final_path, index=False)
#         print(f"[✅ DONE] Merged sentiment saved to: {final_path}")
#     except Exception as e:
#         print(f"[ERROR] Final merge failed in {stock_dir}: {e}")


# # === 批量入口 ===
# if __name__ == '__main__':
#     base_path = "../../../datasets/processed"
#     for folder in os.listdir(base_path):
#         if folder!="GOOG" and folder!="AAPL" and folder!="COIN" and folder!="TSLA" and folder!="NFLX":
#             folder_path = os.path.join(base_path, folder)
#             if os.path.isdir(folder_path):
#                 process_sentiment_for_stock(folder_path)


import os
import sys
import pandas as pd
import ast
import re

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent
from agent_tools.data_handler.news_sentiment.news_sentiment_summary_function import news_sentiment_summary_function


class NewsSentimentAgent(BaseFinanceAgent):
    def __init__(self, name="NewsSentimentAgent", role="Summarize daily news and sentiment trends.",
                 memory_size=1):
        super().__init__(name=name, role=role,
                         function_schema=[news_sentiment_summary_function],
                         memory_size=memory_size)


def parse_news_and_sentiments(news_field_str):
    news_texts, sentiments = [], []
    pattern = r"- (\[\{.*?\}\]) (.+?)(?=\n- |\Z)"
    matches = re.findall(pattern, str(news_field_str), flags=re.DOTALL)
    for sent_str, text in matches:
        try:
            sentiment_list = ast.literal_eval(sent_str)
            sentiment_dict = sentiment_list[0] if isinstance(sentiment_list, list) and sentiment_list else {}
        except Exception:
            sentiment_dict = {}
        news_texts.append(text.strip())
        sentiments.append(sentiment_dict)
    return news_texts, sentiments


def process_sentiment_for_stock(stock_dir):
    merged_input_path = os.path.join(stock_dir, "financial_with_macro_summary.csv")
    if not os.path.exists(merged_input_path):
        print(f"[SKIP] {stock_dir}: No financial_with_macro_summary.csv found.")
        return

    print(f"[INFO] Processing sentiment for {stock_dir}")
    df = pd.read_csv(merged_input_path)
    df['date'] = pd.to_datetime(df['date'])

    # 聚合新闻内容
    grouped = df.groupby('date').agg({
        'news': lambda x: ' '.join(x.dropna().astype(str))
    }).reset_index()

    grouped['news_texts'], grouped['sentiments'] = None, None
    for idx, row in grouped.iterrows():
        news_texts, sentiments = parse_news_and_sentiments(row['news'])
        grouped.at[idx, 'news_texts'] = news_texts
        grouped.at[idx, 'sentiments'] = sentiments

    agent = NewsSentimentAgent()

    sentiment_csv = os.path.join(stock_dir, "daily_news_sentiment_summary.csv")
    # 如果文件已存在，先删除，保证从头开始写
    if os.path.exists(sentiment_csv):
        os.remove(sentiment_csv)

    for idx, row in grouped.iterrows():
        if not row['news_texts']:
            continue
        try:
            context_blocks = []
            for text, sent in zip(row['news_texts'], row['sentiments']):
                sent_summary = f"polarity: {sent.get('polarity', 'N/A')}, positive: {sent.get('pos', 'N/A')}, " \
                               f"negative: {sent.get('neg', 'N/A')}, neural: {sent.get('neu', 'N/A')}"
                context_blocks.append(f"News: {text}\nSentiment: {sent_summary}")
            full_context = "\n\n".join(context_blocks)
            print(full_context)
            prompt = (
                f"Date: {row['date'].strftime('%Y-%m-%d')}\n"
                "You are an expert financial assistant. Analyze the following news headlines along with their sentiment scores. \n"
                "Your tasks are:\n"
                "1. Summarize the key points covered by the news.\n"
                "2. Determine the **overall sentiment trend** (choose from: positive, negative, or neutral).\n"
                "3. Highlight any **notable risks, market concerns, or impactful events**.\n\n"
                "Each news item is accompanied by sentiment scores: polarity, pos, neg, neu.\n"
                "Consider negative signals seriously (e.g., earnings miss, layoffs, lawsuits, fraud, macro fears).\n\n"
                f"{full_context}\n\n"
                "Respond using the 'news_sentiment_summary' function schema."
                "{\n"
                '  "news_summary": "<a brief summary of main topics>",\n'
                '  "overall_sentiment": "<positive | neutral | negative>",\n'
                '  "key_points": "<concise list of notable risks, concerns, or key takeaways>"\n'
                "}"
            )
            result = agent.ask_model(prompt)

            day_result = {
                "date": row["date"],
                "news_summary": result.get("news_summary", ""),
                "overall_sentiment": result.get("overall_sentiment", ""),
                "key_points": result.get("key_points", "")
            }
            day_df = pd.DataFrame([day_result])

            # 如果文件不存在，写入带表头；否则追加不带表头
            if not os.path.exists(sentiment_csv):
                day_df.to_csv(sentiment_csv, index=False, mode='w', encoding='utf-8-sig')
            else:
                day_df.to_csv(sentiment_csv, index=False, mode='a', header=False, encoding='utf-8-sig')

            print(f"[INFO] Saved sentiment for {row['date'].strftime('%Y-%m-%d')}")

        except Exception as e:
            print(f"[ERROR] Sentiment failed at {row['date']} in {stock_dir}: {e}")

    # 这里可以合并 sentiment_csv 与原始数据，生成最终文件
    try:
        sentiment_df = pd.read_csv(sentiment_csv)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        df_merged = df.merge(sentiment_df, on="date", how="left")
        for col in ['news_summary', 'overall_sentiment', 'key_points']:
            df_merged[col] = df_merged[col].fillna('')
        final_path = os.path.join(stock_dir, "financial_with_news_macro_summary.csv")
        df_merged.to_csv(final_path, index=False)
        print(f"[✅ DONE] Merged sentiment saved to: {final_path}")
    except Exception as e:
        print(f"[ERROR] Final merge failed in {stock_dir}: {e}")


# === 批量入口 ===
if __name__ == '__main__':
    base_path = "../../../datasets/processed"
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            process_sentiment_for_stock(folder_path)
