import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from datetime import datetime, timedelta
from rag import generate_suggestion_by_event_range

# Load the dataset
df = pd.read_csv("./rag_text_with_embedding.csv")
df["embedding"] = df["embedding"].apply(eval)
df["date"] = pd.to_datetime(df["date"])

# Define training time window
train_start = pd.to_datetime("2024-08-01")
train_end = pd.to_datetime("2024-12-31")

# Get all available dates (both training and test)
all_dates = df["date"].sort_values().unique()

results = []

for current_date in tqdm(all_dates):
    # Skip if there is no data for the current date
    current_rows = df[df["date"] == current_date]
    if current_rows.empty:
        continue

    # Determine the context range
    if current_date <= train_end:
        start_event = train_start.strftime("%Y-%m-%d")
        end_event = train_end.strftime("%Y-%m-%d")
    else:
        start_event = train_start.strftime("%Y-%m-%d")
        end_event = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        suggestion = generate_suggestion_by_event_range(
            start_date=start_event,
            end_date=end_event,
            target_date=current_date.strftime("%Y-%m-%d")
        )
    except ValueError as e:
        print(f"[Skipped] {current_date.strftime('%Y-%m-%d')}: {e}")
        continue

    results.append({
        "date": current_date.strftime("%Y-%m-%d"),
        **suggestion
    })

# Save results to CSV
pd.DataFrame(results).to_csv("./rag_agent_suggestions.csv", index=False)
print("rag_agent_suggestions.csv has been saved.")
