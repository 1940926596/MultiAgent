# File: train_meta_agent_all.py

import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# === Add project root to path ==
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.agent_roles_openai import (
    TechnicalAnalystAgent, SentimentAnalystAgent, MacroAnalystAgent, RiskAnalystAgent
)


def precompute_observations(data, advisors):
    extra_features = ["macd", "rsi_30", "cci_30", "vix", "turbulence", "close_30_sma", "close_60_sma"]
    cached_obs = []

    for i in range(len(data)):
        row = data.iloc[i].to_dict()
        obs = []

        for name, agent in advisors.items():
            result = agent.analyze(row)
            action = result["action"]
            confidence = result["confidence"]
            for act in ["buy", "hold", "sell"]:
                obs.append(confidence if act == action else 0.0)

        obs.append(0.0)  # initial holding=False

        for feat in extra_features:
            obs.append(row.get(feat, 0.0))

        cached_obs.append(np.array(obs, dtype=np.float32))

    return cached_obs


class MetaCIOEnv(gym.Env):
    def __init__(self, data, advisors, cached_obs=None):
        super(MetaCIOEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.advisors = advisors
        self.cached_obs = cached_obs
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0

        self.extra_features = ["macd", "rsi_30", "cci_30", "vix", "turbulence", "close_30_sma", "close_60_sma"]
        obs_dim = len(advisors) * 3 + 1 + len(self.extra_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.action_space = spaces.Discrete(3)  # 0-buy, 1-hold, 2-sell

    def _get_obs(self):
        if self.cached_obs is not None:
            obs = self.cached_obs[self.current_step].copy()
            holding_index = len(self.advisors) * 3
            obs[holding_index] = 1.0 if self.holding else 0.0
            return obs
        else:
            row = self.data.iloc[self.current_step]
            obs = []
            for name, agent in self.advisors.items():
                result = agent.analyze(row.to_dict())
                action = result["action"]
                confidence = result["confidence"]
                for act in ["buy", "hold", "sell"]:
                    obs.append(confidence if act == action else 0.0)

            obs.append(1.0 if self.holding else 0.0)
            for feat in self.extra_features:
                obs.append(row.get(feat, 0.0))
            return np.array(obs, dtype=np.float32)

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
            reward -= 0.0005
            self.holding_days += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs()

        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Holding: {self.holding}")


# === 多公司训练主入口 ===
if __name__ == "__main__":
    base_path = "../../datasets/processed"
    tickers = ["AAPL", "GOOG", "TSLA", "COIN", "NFLX", "NVDA", "AMZN", "MSFT"]  # 可扩展
    tickers = [tickers[0]]
    for ticker in tickers:
        print(f"\n=== Training model for {ticker} ===")
        ticker_dir = os.path.join(base_path, ticker)
        data_path = os.path.join(ticker_dir, "financial_with_news_macro_summary.csv")
        cache_path = os.path.join(ticker_dir, "cached_observations.csv")
        model_path = os.path.join(ticker_dir, f"meta_cio_rl_{ticker}.zip")

        if not os.path.exists(data_path):
            print(f"[WARN] {ticker} skipped: missing file.")
            continue

        df = pd.read_csv(data_path)

        advisors = {
            "tech": TechnicalAnalystAgent(),
            "sent": SentimentAnalystAgent(),
            "macro": MacroAnalystAgent(),
            "risk": RiskAnalystAgent()
        }

        # 处理 observation 缓存
        if os.path.exists(cache_path):
            print(f"[INFO] Loading cached observations from {cache_path} ...")
            df_cached = pd.read_csv(cache_path)
            cached_obs = df_cached.values.astype(np.float32)
        else:
            print(f"[INFO] Generating cached observations for {ticker} ...")
            cached_obs = precompute_observations(df, advisors)
            cached_obs_array = np.stack(cached_obs)
            pd.DataFrame(cached_obs_array).to_csv(cache_path, index=False)
            print(f"[INFO] Saved to {cache_path}")

        # 初始化环境并训练模型
        env = MetaCIOEnv(df, advisors, cached_obs=cached_obs)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/", device="cpu")
        model.learn(total_timesteps=50000)
        model.save(model_path)
        print(f"[✅ DONE] Model saved to {model_path}")
