import json
import pandas as pd
import numpy as np
import faiss
import time
from tqdm import tqdm
from openai import OpenAI
import my_config
from tools import function_schema

# Load rag_text dataset
df = pd.read_csv("./rag_text_with_embedding.csv")

# Initialize OpenAI client
client = OpenAI(api_key=my_config.api_key)

# Embedding function (compatible with openai>=1.0.0)
def get_embedding(text: str):
    for _ in range(3):  # Retry mechanism
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            time.sleep(1)
    return None

# Generate embeddings if not already present
if "embedding" not in df.columns:
    tqdm.pandas(desc="Generating embeddings")
    df["embedding"] = df["rag_text"].progress_apply(get_embedding)
    df.dropna(subset=["embedding"], inplace=True)
    df.to_csv("./rag_text_with_embedding.csv", index=False)
else:
    df["embedding"] = df["embedding"].apply(eval)  # Convert stringified list back to list

# Build FAISS index
embedding_dim = len(df["embedding"].iloc[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.vstack(df["embedding"].values))

# Suggestion generator using FAISS + GPT
# RagAnalystAgent
def generate_suggestion_by_event_range(target_date: str, start_date: str, end_date: str, top_k=5):
    # 转换为 datetime 对象
    target_date_dt = pd.to_datetime(target_date)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 获取目标行
    target_row_df = df[df["date"] == target_date]
    if target_row_df.empty:
        raise ValueError(f"Target date '{target_date}' not found in dataset.")
    target_row = target_row_df.iloc[0]
    query_vec = np.array(target_row["embedding"]).astype("float32").reshape(1, -1)

    # 筛选出指定事件区间内的数据（不包括 target_date 本身）
    df_range = df[
        (pd.to_datetime(df["date"]) >= start_dt) &
        (pd.to_datetime(df["date"]) <= end_dt) &
        (df["date"] != target_date)
    ].copy()

    if df_range.empty:
        raise ValueError("No data found in the specified event range.")

    # 构建临时 FAISS index
    temp_index = faiss.IndexFlatL2(len(df_range["embedding"].iloc[0]))
    temp_index.add(np.vstack(df_range["embedding"].values).astype("float32"))

    # 查询相似项
    distances, indices = temp_index.search(query_vec, top_k)
    similar_contexts = "\n\n".join(df_range.iloc[idx]["rag_text"] for idx in indices[0])
    current_context = target_row["rag_text"]

    # 构造 Prompt
    prompt = (
        f"You are acting as a Chief Investment Officer (CIO) at a top-tier financial institution.\n"
        f"Your role is to make informed and professional investment decisions based on current market data and historical analogs.\n\n"
        f"**Current Date**: {target_row['date']}\n\n"
        f"**Current Market Context Summary (from RAG text)**:\n"
        f"{current_context}\n\n"
        f"**Most Relevant Historical Market Cases (from FAISS-matched RAG texts between {start_date} and {end_date})**:\n"
        f"{similar_contexts}\n\n"
        f"Please follow these steps:\n"
        f"1. Carefully analyze the **Current Market Context**, extracting key financial signals, trends, risks, or sentiments.\n"
        f"2. Compare these elements with the **Historical Cases** provided.\n"
        f"3. Identify patterns, analogies, and lessons from history that can help assess the current situation.\n"
        f"4. Formulate a reasoned investment decision based on your analysis.\n\n"
        f"Output Format:\n"
        f"- `action`: one of ['buy', 'sell', 'hold']\n"
        f"- `confidence`: float between 0 and 1, indicating how strongly you support this action\n"
        f"- `reason`: a concise but professional justification based on evidence from both current and historical contexts\n\n"
        f"\"Respond by calling the 'stock_decision' function according to the schema.\""
    )

    print(prompt)

    # 调用模型
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional financial analyst. Your task is to generate clear, actionable investment advice based on historical and current data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        functions=function_schema,
        function_call={"name": "stock_decision"}
    )

    args = response.choices[0].message.function_call.arguments
    parsed = json.loads(args)
    return parsed

if __name__ == "__main__":
    suggestion = generate_suggestion_by_event_range(
        target_date="2024-11-12",
        start_date="2024-08-01",
        end_date="2024-12-31",
        top_k=5
    )
    print(suggestion)
