# File: train_meta_agent_all.py

import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# === Add project root to path ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Observation Builder from CSVs ===
def precompute_observations_from_files(summary_df, rag_suggestions_path):
    rag_df = pd.read_csv(rag_suggestions_path)
    rag_df["date"] = pd.to_datetime(rag_df["date"]).dt.date
    rag_lookup = rag_df.set_index("date").to_dict(orient="index")

    extra_features = [
        "macd", "rsi_30", "cci_30", "vix", "turbulence", "close_30_sma", "close_60_sma"
    ]
    cached_obs = []

    for _, row in summary_df.iterrows():
        obs = []

        # 4 base agents: technical, sentiment, macro, risk
        for key in ["technical", "sentiment", "macro", "risk"]:
            action = row.get(f"{key}_action", "hold")
            confidence = float(row.get(f"{key}_confidence", 0.0))
            for act in ["buy", "hold", "sell"]:
                obs.append(confidence if act == action else 0.0)

        # rag agent (from rag_agent_suggestions.csv)
        date_key = pd.to_datetime(row["date"]).date()
        rag_result = rag_lookup.get(date_key, None)
        if rag_result:
            rag_action = rag_result.get("action", "hold")
            rag_conf = float(rag_result.get("confidence", 0.0))
        else:
            rag_action, rag_conf = "hold", 0.0

        for act in ["buy", "hold", "sell"]:
            obs.append(rag_conf if act == rag_action else 0.0)

        # holding flag
        obs.append(0.0)

        # extra features
        for feat in extra_features:
            obs.append(row.get(feat, 0.0))

        cached_obs.append(np.array(obs, dtype=np.float32))

    return cached_obs


# === MetaCIO Environment ===
class MetaCIOEnv(gym.Env):
    def __init__(self, data, cached_obs):
        super(MetaCIOEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.cached_obs = cached_obs
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0

        num_agents = 5  # 4 base + 1 rag
        obs_dim = num_agents * 3 + 1 + 7  # 3 actions per agent + holding + 7 extra features

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0-buy, 1-hold, 2-sell

    def _get_obs(self):
        obs = self.cached_obs[self.current_step].copy()
        obs[15] = 1.0 if self.holding else 0.0  # overwrite holding flag at index 15
        return obs

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0
        return self._get_obs()

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row["close"]

        reward = 0.0
        if action == 0 and not self.holding:
            self.holding = True
            self.entry_price = price
            self.holding_days = 0
        elif action == 2 and self.holding:
            pnl = (price - self.entry_price) / self.entry_price
            reward += pnl
            self.holding = False
            self.entry_price = 0.0
            self.holding_days = 0
        elif self.holding:
            self.holding_days += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Holding: {self.holding}")


# === Multi-Ticker Training Entry ===
if __name__ == "__main__":
    base_path = "../../datasets/processed"
    tickers = ["AAPL", "GOOG", "TSLA", "COIN", "NFLX", "NIO", "AMZN", "MSFT"]

    for ticker in tickers:
        print(f"\n=== Training model for {ticker} ===")
        ticker_dir = os.path.join(base_path, ticker)
        data_path = os.path.join(ticker_dir, "financial_with_news_macro_summary.csv")
        rag_path = os.path.join(ticker_dir, "rag_agent_suggestions.csv")
        cache_path = os.path.join(ticker_dir, "cached_observations.csv")
        model_path = os.path.join(ticker_dir, f"meta_cio_rl_{ticker}.zip")

        if not os.path.exists(data_path) or not os.path.exists(rag_path):
            print(f"[WARN] Skipping {ticker}: Missing data or suggestions.")
            continue

        df = pd.read_csv(data_path)

        if os.path.exists(cache_path):
            print(f"[INFO] Loading cached observations from {cache_path} ...")
            cached_obs = pd.read_csv(cache_path).values.astype(np.float32)
        else:
            print(f"[INFO] Precomputing observations for {ticker} ...")
            cached_obs = precompute_observations_from_files(df, rag_path)
            pd.DataFrame(cached_obs).to_csv(cache_path, index=False)
            print(f"[INFO] Saved observations to {cache_path}")

        env = MetaCIOEnv(df, cached_obs)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/", device="cpu")
        model.learn(total_timesteps=50000)
        model.save(model_path)
        print(f"[✅ DONE] Model saved to {model_path}")
