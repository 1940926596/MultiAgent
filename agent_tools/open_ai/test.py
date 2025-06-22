# File: cio_batch_analysis.py
import sys
import os
# === Add project root to path ==
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import pandas as pd
from agent_tools.open_ai.agent_roles_openai import (
    TechnicalAnalystAgent,
    SentimentAnalystAgent,
    MacroAnalystAgent,
    RiskAnalystAgent,
    CIOAgent
)

def analyze_company(ticker_dir):
    data_path = os.path.join(ticker_dir, "financial_with_news_macro_summary.csv")
    output_path = os.path.join(ticker_dir, "cio_analysis_results.csv")

    if not os.path.exists(data_path):
        print(f"[SKIP] Missing data for {ticker_dir}")
        return

    df = pd.read_csv(data_path)
    if df.empty:
        print(f"[SKIP] Empty dataframe for {ticker_dir}")
        return

    # 初始化各分析智能体
    tech = TechnicalAnalystAgent()
    sent = SentimentAnalystAgent()
    macro = MacroAnalystAgent()
    risk = RiskAnalystAgent()
    cio = CIOAgent(advisors=[tech, sent, macro, risk])

    results = []
    for i, row in df.iterrows():
        data = row.to_dict()
        try:
            result = cio.analyze(data)
            result["date"] = data["date"]
            result["tic"] = data.get("tic", "UNKNOWN")
            results.append(result)
            print(result)
        except Exception as e:
            print(f"[ERROR] Failed at row {i} in {ticker_dir}: {e}")
            continue

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"[DONE] Saved results to {output_path}")
    else:
        print(f"[WARN] No results generated for {ticker_dir}")


if __name__ == "__main__":
    base_dir = "../../datasets/processed"
    tickers = ["AAPL", "GOOG", "TSLA", "COIN", "NFLX", "NIO", "AMZN", "MSFT"]
    tickers = [tickers[7]]
    for ticker in tickers:
        print(f"\n=== Analyzing {ticker} ===")
        ticker_dir = os.path.join(base_dir, ticker)
        analyze_company(ticker_dir)
