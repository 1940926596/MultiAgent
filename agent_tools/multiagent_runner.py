import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from agent_roles import agent_list  # 请确保你已有这个模块，包含多个 FinanceAgent 实例

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).head(2)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    print(df.head())

    # 缺失值处理（可根据字段调整）
    df = df.fillna({
        'macd': 0.0, 'rsi_30': 50.0, 'close': 0.0,
        'all_fields': '', 'news_text': '', 'sentiment': 0.0,
        'vix': 20.0, 'turbulence': 100.0
    })
    print(df.head())

    return df

def truncate_text(text: str, max_len: int = 1024) -> str:
    return text[:max_len] if isinstance(text, str) else ""

def run_agents_on_dataframe(df: pd.DataFrame, agents: List) -> pd.DataFrame:
    # df: 一个 pandas 的 DataFrame，包含你之前准备好的金融数据（每天每只股票的技术指标、文本信息等）
    # agents: 一个智能体列表，包含多个基于大模型构建的 FinanceAgent 实例，比如宏观分析师、风险控制专家等
    
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        data_point = row.to_dict()
        # 把 row 转成字典格式，便于传给智能体。

        # 🚫 限制输入文本长度以防显存爆炸
        data_point["all_fields"] = truncate_text(data_point.get("all_fields", ""), 1024)
        data_point["news_text"] = truncate_text(data_point.get("news_text", ""), 1024)

        daily_result = {"date": row["date"], "tic": row["tic"]}

        for agent in agents:
            agent_output = agent.analyze(data_point)
            daily_result[f"{agent.name}_action"] = agent_output.get("action", "N/A")
            daily_result[f"{agent.name}_reason"] = agent_output.get("reasoning", "")

        results.append(daily_result)

    return pd.DataFrame(results)

def print_sample_results(df: pd.DataFrame, interval: int = 10):
    print("\n🧪 部分分析结果预览：")
    for i in range(0, len(df), interval):
        print(f"\n📅 日期: {df.iloc[i]['date']} / 股票: {df.iloc[i]['tic']}")
        for col in df.columns:
            if col.endswith("_action") or col.endswith("_reason"):
                print(f"{col}: {df.iloc[i][col]}")

if __name__ == "__main__":
    # 路径请根据你的环境修改
    data_path = "../datasets/processed/financial_final.csv"
    save_path = "multiagent_analysis_result.csv"

    print("📥 正在加载并预处理数据...")
    df = load_and_preprocess_data(data_path)

    print("🤖 正在执行多智能体分析...")
    result_df = run_agents_on_dataframe(df, agent_list)

    print("📊 正在保存分析结果...")
    result_df.to_csv(save_path, index=False)

    print("✅ 多日分析完成，结果已保存至:", save_path)
    print_sample_results(result_df, interval=20)
