# 合并
import pandas as pd

# 读两个表
financial_df = pd.read_csv('../macro/financial_with_macro_summary.csv')
news_df = pd.read_csv('daily_news_sentiment_summary.csv')

# 确保日期列是同一类型（日期格式）
financial_df['date'] = pd.to_datetime(financial_df['date'])
news_df['date'] = pd.to_datetime(news_df['date'])

# 按 date 左连接 news_df（新闻摘要）到 financial_df
merged_df = financial_df.merge(news_df, on='date', how='left')

# 对于新闻摘要缺失的行，summary、overall_sentiment、key_points 自动为 NaN，可以改为空字符串（如果你想）
merged_df['news_summary'] = merged_df['news_summary'].fillna('')
merged_df['overall_sentiment'] = merged_df['overall_sentiment'].fillna('')
merged_df['key_points'] = merged_df['key_points'].fillna('')

# 保存合并结果
merged_df.to_csv('../../../datasets/processed/financial_with_news_macro_summary.csv', index=False)
# merged_df.to_csv('financial_with_news_macro_summary.csv', index=False)
print("Saved merged financial and news summary to financial_with_news_summary.csv")