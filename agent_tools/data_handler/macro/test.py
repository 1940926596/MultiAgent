import os
import sys
import pandas as pd
import json
from pandas.tseries.offsets import Week

# 项目导入路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent
from agent_tools.data_handler.macro.macro_summary import macro_summary_function


class MacroFinanceAgent(BaseFinanceAgent):
    def __init__(self, name="MacroAgent", role="Analyze quarterly reports and summarize macro-level financial health.",
                 model="gpt-4", memory_size=8):
        super().__init__(name=name, role=role, model=model, function_schema=[macro_summary_function], memory_size=memory_size)
        self.interested_fields = ["form_type", "all_fields", "date", "tic"]


def process_stock_file(stock_dir):
    input_csv = os.path.join(stock_dir, "financial_final.csv")
    output_with_macro = os.path.join(stock_dir, "financial_final_with_macro.csv")
    output_summary = os.path.join(stock_dir, "macro_summary_results.csv")
    final_merged = os.path.join(stock_dir, "financial_with_macro_summary.csv")

    if not os.path.exists(input_csv):
        print(f"[SKIP] No financial_final.csv in {stock_dir}")
        return

    print(f"[INFO] Processing: {input_csv}")
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["date", "tic", "form_type", "all_fields"])
    df['date'] = pd.to_datetime(df['date'])

    macro_agent = MacroFinanceAgent()

    df['macro_summary'] = None
    df['risk_tags'] = None
    df['macro_score'] = None
    results = []

    for idx, row in df.iterrows():
        if not row["all_fields"] or str(row["all_fields"]).strip() == "":
            continue
        try:
            prompt = (
                f"Date: {row['date']}\n"
                f"Form Type: {row['form_type']}\n"
                "Please summarize the operational status, potential risks, "
                "and provide a macro score for this quarter based on the financial report.\n"
                "Please respond by calling the 'macro_summary' function according to the specified schema."
            )
            result = macro_agent.ask_model(prompt)

            df.at[idx, 'macro_summary'] = result.get('macro_summary', '')
            df.at[idx, 'risk_tags'] = ', '.join(result.get('risk_tag', []))
            df.at[idx, 'macro_score'] = result.get('macro_score', None)

            results.append({
                "date": row["date"],
                "tic": row["tic"],
                "macro_summary": result.get("macro_summary"),
                "risk_tag": result.get("risk_tag"),
                "macro_score": result.get("macro_score")
            })
        except Exception as e:
            print(f"[ERROR] Failed at index {idx} in {stock_dir}: {e}")

    df.to_csv(output_with_macro, index=False)
    pd.DataFrame(results).to_csv(output_summary, index=False)
    print(f"[DONE] Saved macro results to:\n  {output_with_macro}\n  {output_summary}")

    # ========== 合并 macro_summary 到 financial_final ==========
    try:
        financial_df = pd.read_csv(input_csv)
        news_df = pd.read_csv(output_summary)

        financial_df['date'] = pd.to_datetime(financial_df['date'])
        news_df['date'] = pd.to_datetime(news_df['date'])

        news_df = news_df.drop(columns=['tic'], errors='ignore')  # 防止没有该列时报错
        merged_df = financial_df.merge(news_df, on='date', how='left')

        merged_df['macro_summary'] = merged_df['macro_summary'].fillna('')
        merged_df['risk_tag'] = merged_df['risk_tag'].fillna('')
        merged_df['macro_score'] = merged_df['macro_score'].fillna('')

        merged_df.to_csv(final_merged, index=False)
        print(f"[MERGED] Saved merged financial + macro to: {final_merged}")

    except Exception as e:
        print(f"[ERROR] Failed to merge macro summary in {stock_dir}: {e}")


# === 批量执行入口 ===
base_dir = "../../../datasets/processed"
for stock_folder in os.listdir(base_dir):
    stock_path = os.path.join(base_dir, stock_folder)
    if os.path.isdir(stock_path):
        process_stock_file(stock_path)
