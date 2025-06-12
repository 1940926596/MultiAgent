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
def generate_suggestion_by_date(date_str, top_k=5):
    # Locate row for the given date
    target_row = df[df["date"] == date_str].iloc[0]
    query_vec = np.array(target_row["embedding"]).astype("float32").reshape(1, -1)

    # Search for top-k similar entries
    distances, indices = index.search(query_vec, top_k)
    similar_contexts = "\n\n".join(df.iloc[idx]["rag_text"] for idx in indices[0])
    current_context = target_row["rag_text"]

    # Construct the RAG prompt
    prompt = (
        f"You are acting as a Chief Investment Officer (CIO) at a top-tier financial institution.\n"
        f"Your role is to make informed and professional investment decisions based on current market data and historical analogs.\n\n"
        f"**Current Date**: {target_row['date']}\n\n"
        f"**Current Market Context Summary (from RAG text)**:\n"
        f"{current_context}\n\n"
        f"**Most Relevant Historical Market Cases (from FAISS-matched RAG texts)**:\n"
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
    # Query OpenAI (GPT-4o or gpt-4-turbo)
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


suggestion = generate_suggestion_by_date("2024-11-12")
print(suggestion)
