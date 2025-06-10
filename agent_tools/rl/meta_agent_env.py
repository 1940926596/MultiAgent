# 文件名：meta_agent_env.py
import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.rl.test import MetaAgentRL

class MetaCIOEnv(gym.Env):
    def __init__(self, data, advisors):
        super(MetaCIOEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.advisors = advisors
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0

        # === 构造状态空间 ===
        self.extra_features = ["macd", "rsi_30", "cci_30", "vix", "turbulence", "close_30_sma", "close_60_sma"]
        obs_dim = len(advisors) * 3 + 1 + len(self.extra_features)  # advisor输出 + 持仓状态 + 技术指标
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.action_space = spaces.Discrete(3)  # 0-buy, 1-hold, 2-sell

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        obs = []

        # === advisor 输出 action/confidence，只有选中那项为confidence，其它为0 ===
        for name, agent in self.advisors.items():
            result = agent.analyze(row.to_dict())
            action = result["action"]
            confidence = result["confidence"]
            for act in ["buy", "hold", "sell"]:
                obs.append(confidence if act == action else 0.0)

        # === 当前是否持仓 ===
        obs.append(1.0 if self.holding else 0.0)

        # === 加入额外特征 ===
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
        next_price = row.get("close_next", price)

        reward = 0.0
        transaction_cost = 0.001  # 千分之一

        # === buy ===
        if action == 0 and not self.holding:
            self.holding = True
            self.entry_price = price
            self.holding_days = 0
            reward -= price * transaction_cost

        # === sell ===
        elif action == 2 and self.holding:
            pnl = (price - self.entry_price) / self.entry_price
            reward += pnl
            reward -= price * transaction_cost
            self.holding = False
            self.entry_price = 0.0
            self.holding_days = 0

        # === still holding ===
        elif self.holding:
            pnl = (price - self.entry_price) / self.entry_price
            reward += 0.0  # 不立即计入浮盈
            reward -= 0.0005  # 时间惩罚（每持有一天扣一点）

            self.holding_days += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs()

        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Holding: {self.holding}")


# === 以下是训练入口 ===

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

env = MetaCIOEnv(df.head(200), advisors)

from stable_baselines3 import PPO

# 加入 tensorboard 日志目录
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

model.learn(total_timesteps=10_000)

model.save("meta_cio_rl")

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print("Total reward:", total_reward)
