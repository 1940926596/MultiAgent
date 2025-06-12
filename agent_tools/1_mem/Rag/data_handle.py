import pandas as pd
import numpy as np
import faiss
import openai
import time
from tqdm import tqdm

openai.api_key = "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"


# 读取 rag_text
df = pd.read_csv("./rag_text.csv")

# 获取 OpenAI 嵌入向量
def get_embedding(text, retries=3):
    for _ in range(retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            print("获取嵌入失败，重试中...", e)
            time.sleep(1)
    return None

# 如果你还没生成 embedding：
if "embedding" not in df.columns:
    df["embedding"] = df["rag_text"].apply(get_embedding)
    df.dropna(subset=["embedding"], inplace=True)
    df.to_csv("./rag_text_with_embedding.csv", index=False)
else:
    df["embedding"] = df["embedding"].apply(eval)  # 若已存储为字符串，转为 list

# 构建 FAISS 向量索引
embedding_dim = len(df["embedding"].iloc[0])
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.vstack(df["embedding"].values))


def generate_suggestion_by_date(date_str, top_k=5):
    target_row = df[df["date"] == date_str].iloc[0]
    query_vec = np.array(target_row["embedding"]).astype("float32").reshape(1, -1)

    # FAISS 检索最相似的 top_k 个文本
    distances, indices = index.search(query_vec, top_k)
    similar_contexts = "\n\n".join(df.iloc[idx]["rag_text"] for idx in indices[0])

    # 构造 Prompt
    prompt = f"""
以下是历史上与当前市场相似的情况，请参考分析建议：

{similar_contexts}

当前日期: {target_row['date']}
请你作为一名专业金融CIO，根据以上历史上下文和当前信息，输出操作建议 (buy/sell/hold)，并给出信心值（0-1）和简洁有逻辑的理由。
"""

    # 调用 GPT
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是专业的金融分析师，需根据历史上下文与当前情况生成明确建议"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    suggestion = response["choices"][0]["message"]["content"]
    return suggestion


suggestion = generate_suggestion_by_date("2024-11-12")
print(suggestion)
