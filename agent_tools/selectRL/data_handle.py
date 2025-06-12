# === Handle Data ===
# Load dataset
import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path to enable relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Load short-term agents' output (technical, sentiment, macro, risk) ===
agent_csv_path = '../mem/Rag/rag_text.csv'
df_agent_tsmr = pd.read_csv(agent_csv_path)
required_fields = [
    "date", "technical_action", "technical_confidence", "technical_reasoning",
    "sentiment_action", "sentiment_confidence", "sentiment_reasoning",
    "macro_action", "macro_confidence", "macro_reasoning",
    "risk_action", "risk_confidence", "risk_reasoning"
]
df_agent_tsmr = df_agent_tsmr[required_fields].copy()
print(df_agent_tsmr.head())

# === Load long-term RAG agent output ===
agent_csv_path1 = '../mem/Rag/rag_agent_suggestions.csv'
df_agent_rag = pd.read_csv(agent_csv_path1)
required_fields = ["date", "action", "confidence", "reasoning"]
df_agent_rag = df_agent_rag[required_fields].copy()
print(df_agent_rag.head())

# === Convert action strings to integers ===
# Mapping: buy=0, hold=1, sell=2
action_map = {"buy": 0, "hold": 1, "sell": 2}

def encode_action_confidence(df, prefix):
    actions = df[f"{prefix}_action"].str.lower().map(action_map).fillna(1).astype(int)  # default to hold
    confidences = df[f"{prefix}_confidence"].fillna(0.5).astype(float)
    return np.stack([actions, confidences], axis=1)  # shape (N, 2)

# Encode the 4 short-term agents
tech = encode_action_confidence(df_agent_tsmr, "technical")
sent = encode_action_confidence(df_agent_tsmr, "sentiment")
macro = encode_action_confidence(df_agent_tsmr, "macro")
risk = encode_action_confidence(df_agent_tsmr, "risk")

# Combine into shape (N, 4, 2)
short_agents = np.stack([tech, sent, macro, risk], axis=1)

# Encode the RAG agent
df_agent_rag["action"] = df_agent_rag["action"].str.lower().map(action_map).fillna(1).astype(int)
df_agent_rag["confidence"] = df_agent_rag["confidence"].fillna(0.5).astype(float)
rag = df_agent_rag[["action", "confidence"]].values  # shape (N, 2)

def one_hot_action_confidence(arr):
    # arr shape (N, 2): column 0 = action (0/1/2), column 1 = confidence
    N = arr.shape[0]
    onehot = np.zeros((N, 3), dtype=np.float32)
    onehot[np.arange(N), arr[:, 0].astype(int)] = 1.0
    confidence = arr[:, 1].reshape(-1, 1)
    return np.concatenate([onehot, confidence], axis=1)  # shape (N, 4)

# Encode each agent (4 short-term + 1 long-term) to (N, 4) then combine
agent_feats = []
for i in range(4):
    agent_feats.append(one_hot_action_confidence(short_agents[:, i, :]))
agent_feats.append(one_hot_action_confidence(rag))

# Combine into shape (N, 5 agents, 4 features) -> reshape to (N, 20)
agent_feats = np.stack(agent_feats, axis=1)
cached_obs = agent_feats.reshape(agent_feats.shape[0], -1)
print(cached_obs)

# === Load price data ===
price_data = pd.read_csv("../../datasets/processed/financial_final.csv")  # must contain ['date', 'close']
price_data = price_data.sort_values("date").reset_index(drop=True)
price_data["close_next"] = price_data["close"].shift(-1).ffill()

# === Align dates with agent data ===
df_dates = df_agent_tsmr[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
data = pd.merge(df_dates, price_data, on="date", how="left").ffill()

required_fields = ["date", "close", "close_next"]
data = data[required_fields].copy()
print(data.tail())
