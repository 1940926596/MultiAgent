# handle data
# Load dataset
import os
import sys
import pandas as pd
import re

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent

original_csv_path = '../../../datasets/processed/financial_with_news_macro_summary.csv'
df_original = pd.read_csv(original_csv_path)

agent_csv_path= '../../open_ai/cio_analysis_results.csv'
df_agent=pd.read_csv(agent_csv_path)

required_fields=["date","reasoning"]
df_agent=df_agent[required_fields].copy()
print(df_agent.head())

df_original['date'] = pd.to_datetime(df_original['date']).dt.date
df_agent['date'] = pd.to_datetime(df_agent['date']).dt.date

df_merged = pd.merge(df_original, df_agent,  on="date", how="inner")

new_rows = []
for i, row in df_merged.iterrows():
    data = row.to_dict()
    text = data["reasoning"]
    pattern = re.compile(
        r"\[(?P<role>.*?) Analyst\]: \(action: (?P<action>\w+)\) (?P<reasoning>.*?)\(Confidence: (?P<confidence>[\d.]+)\)",
        re.DOTALL,
    )

    results = pattern.findall(text)

    # 构造字段字典
    fields = {}
    for role, action, reasoning, confidence in results:
        key_base = role.strip().lower().replace(" ", "_")  # e.g. technical_analyst
        fields[f"{key_base}_action"] = action.strip()
        fields[f"{key_base}_confidence"] = float(confidence.strip())
        fields[f"{key_base}_reasoning"] = reasoning.strip()

    combined = {**data, **fields}
    new_rows.append(combined)

df_merged_with_results = pd.DataFrame(new_rows)
print(df_merged_with_results.head())
print(df_merged_with_results.columns.tolist())

def build_rag_text(row):
    return f"""
Date: {row['date']} | Ticker: {row.get('tic', 'N/A')}

[Technical Analyst Reasoning]: {row.get('technical_reasoning', '')}
[Sentiment Analyst Reasoning]: {row.get('sentiment_reasoning', '')}
[Macro Analyst Reasoning]: {row.get('macro_reasoning', '')}
[Risk Analyst Reasoning]: {row.get('risk_reasoning', '')}

[Market Info]:
- Open: {row.get('open', '')}, High: {row.get('high', '')}, Low: {row.get('low', '')}, Close: {row.get('close', '')}
- Volume: {row.get('volume', '')}, VIX: {row.get('vix', '')}, Turbulence: {row.get('turbulence', '')}
- Sentiment Summary: {row.get('news_summary', '')}
- Macro Summary: {row.get('macro_summary', '')}
- Risk Tag: {row.get('risk_tag', '')}
    """.strip()

df_merged_with_results["rag_text"] = df_merged_with_results.apply(build_rag_text, axis=1)
df_merged_with_results.to_csv("./rag_text.csv",index=False)

