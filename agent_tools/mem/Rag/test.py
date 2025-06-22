# File: rag_inference_all.py
import os
import json
import faiss
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from openai import OpenAI
import my_config
from tools import function_schema

client = OpenAI(
    api_key=my_config.api_key,
    base_url="https://api.deepseek.com"
    )

base_path = "../../../datasets/processed"
tickers = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NFLX", "NIO", "COIN"]
tickers=[tickers[7]]
def generate_suggestion(target_row, df_range, top_k=5):
    query_vec = np.array(eval(target_row["embedding"])).astype("float32").reshape(1, -1)
    df_range["embedding"] = df_range["embedding"].apply(eval)

    index = faiss.IndexFlatL2(len(df_range["embedding"].iloc[0]))
    index.add(np.vstack(df_range["embedding"].values).astype("float32"))

    distances, indices = index.search(query_vec, top_k)
    similar_contexts = "\n\n".join(df_range.iloc[i]["rag_text"] for i in indices[0])
    current_context = target_row["rag_text"]
    start_date = df_range["date"].min().strftime("%Y-%m-%d")
    end_date = df_range["date"].max().strftime("%Y-%m-%d")

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


    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a financial CIO giving investment decisions."},
            {"role": "user", "content": prompt}
        ],
        tools=[{
            "type": "function",
            "function": function_schema[0]
        }] if function_schema else [],

        tool_choice={"type": "function", "function": {"name": function_schema[0]["name"]}} if function_schema else "auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        args = tool_call.function.arguments
        parsed = json.loads(args)
        return parsed
    else:
        print("[⚠️ tool_calls 为空，未触发函数调用]")
        return {
            "action": "hold",
            "confidence": 0.5,
            "reasoning": "Model did not return tool call"
        }

for ticker in tickers:
    print(f"\n[🧠] RAG inference for {ticker}")
    df_path = os.path.join(base_path, ticker, "rag_text_with_embedding.csv")
    output_path = os.path.join(base_path, ticker, "rag_agent_suggestions.csv")

    if not os.path.exists(df_path):
        print(f"[❌] Missing embedding file: {df_path}")
        continue

    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    all_dates = sorted(df["date"].unique())

    train_start = pd.to_datetime("2022-01-03")
    train_end = pd.to_datetime("2022-10-04")

    results = []

    for current_date in tqdm(all_dates):
        row = df[df["date"] == current_date]
        if row.empty:
            continue
        row = row.iloc[0]

        if current_date <= train_end:
            df_range = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        else:
            df_range = df[(df["date"] >= train_start) & (df["date"] < current_date)]

        if df_range.empty:
            print(f"[⏭️] No context for {current_date.date()}")
            continue

        try:
            suggestion = generate_suggestion(row, df_range)
            results.append({
                "date": current_date.strftime("%Y-%m-%d"),
                **suggestion
            })
        except Exception as e:
            print(f"[❌] Error on {current_date.date()}: {e}")
            continue

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[✅] Saved: {output_path}")
