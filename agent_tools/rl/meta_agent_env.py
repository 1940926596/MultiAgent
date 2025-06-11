# File: meta_agent_env.py
import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces

# === Add project root to path ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class MetaCIOEnv(gym.Env):
    def __init__(self, data, advisors, cached_obs):
        super(MetaCIOEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.advisors = advisors
        self.cached_obs = cached_obs
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0

        self.extra_features = ["macd", "rsi_30", "cci_30", "vix", "turbulence", "close_30_sma", "close_60_sma"]
        obs_dim = len(advisors) * 3 + 1 + len(self.extra_features)  # advisor output + holding state + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0-buy, 1-hold, 2-sell

    def _get_obs(self):
        return self.cached_obs[self.current_step]

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0
        return self._get_obs()

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row["close"]
        next_price = row.get("close_next", price)

        reward = 0.0
        transaction_cost = 0.001  # 0.1%

        if action == 0 and not self.holding:
            self.holding = True
            self.entry_price = price
            self.holding_days = 0
            reward -= price * transaction_cost
        elif action == 2 and self.holding:
            pnl = (price - self.entry_price) / self.entry_price
            reward += pnl
            reward -= price * transaction_cost
            self.holding = False
            self.entry_price = 0.0
            self.holding_days = 0
        elif self.holding:
            reward += 0.0
            reward -= 0.0005
            self.holding_days += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Holding: {self.holding}")

# === Precompute advisor observations ===
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

        obs.append(0.0)  # holding 状态初始化为 False

        for feat in extra_features:
            obs.append(row.get(feat, 0.0))

        cached_obs.append(np.array(obs, dtype=np.float32))

    return cached_obs

# === Training Entry Point ===
from agent_tools.open_ai.agent_roles_openai import (
    TechnicalAnalystAgent, SentimentAnalystAgent, FundamentalAnalystAgent,
    CIOAgent, MacroAnalystAgent, RiskAnalystAgent
)

df = pd.read_csv("../../datasets/processed/financial_with_news_macro_summary.csv")
df["close_next"] = df["close"].shift(-1)

advisors = {
    "tech": TechnicalAnalystAgent(),
    "sent": SentimentAnalystAgent(),
    "macro": MacroAnalystAgent(),
    "risk": RiskAnalystAgent()
}

print("⏳ Running advisors to precompute observations...")
cached_obs = precompute_observations(df, advisors)
print("✅ Advisor outputs cached.")

env = MetaCIOEnv(df, advisors, cached_obs)

from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./tensorboard_logs/", device="cpu")
model.learn(total_timesteps=2)
model.save("meta_cio_rl")


# # === Testing Phase ===
# env = MetaCIOEnv(df.tail(1).reset_index(drop=True), advisors)
# obs = env.reset()
# done = False
# total_reward = 0

# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
#     total_reward += reward
#     env.render()

# print("Total reward:", total_reward)


# # === Visualization ===
# import matplotlib.pyplot as plt

# env = MetaCIOEnv(df.tail(1).reset_index(drop=True), advisors)
# obs = env.reset()
# done = False
# total_reward = 0
# rewards = []
# portfolio_values = []
# prices = []
# steps = []

# cash = 1.0  # initial capital
# holding = 0
# entry_price = 0

# while not done:
#     row = env.data.iloc[env.current_step]
#     price = row["close"]

#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)

#     # Update portfolio state
#     if action == 0 and not env.holding:
#         entry_price = price
#         holding = 1
#         cash -= price

#     elif action == 2 and env.holding:
#         cash += price
#         holding = 0

#     current_value = cash + (price if holding else 0)

#     rewards.append(reward)
#     portfolio_values.append(current_value)
#     prices.append(price)
#     steps.append(env.current_step)

#     env.render()

# print("Total reward:", total_reward)

# # === Plot ===
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(steps, portfolio_values, label="Portfolio Value")
# plt.title("Portfolio Value Over Time")
# plt.xlabel("Step")
# plt.ylabel("Value")
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(steps, rewards, label="Daily Reward", color="orange")
# plt.title("Daily Reward")
# plt.xlabel("Step")
# plt.ylabel("Reward")
# plt.legend()

# plt.tight_layout()
# plt.show()
