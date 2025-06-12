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

from gym import Env
from gym.spaces import Box, Discrete

class AgentSelectEnv(gym.Env):
    def __init__(self, data, cached_obs):
        super(AgentSelectEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.cached_obs = cached_obs
        self.current_step = 0

        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0

        # 观察空间：agent特征 + 持仓状态(1维)
        obs_dim = cached_obs.shape[1] + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 动作空间：5个agent的权重向量，每个权重在0~1之间
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

    def _get_obs(self):
        holding_flag = np.array([1.0 if self.holding else 0.0])
        return np.concatenate([self.cached_obs[self.current_step], holding_flag], axis=0)

    def reset(self):
        self.current_step = 0
        self.holding = False
        self.entry_price = 0.0
        self.holding_days = 0
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        if np.sum(action) > 0:
            action = action / np.sum(action)  # 归一化权重       
             
        row = self.data.iloc[self.current_step]
        price = row["close"]
        next_price = self.data.iloc[self.current_step + 1]["close"] if self.current_step + 1 < len(self.data) else price
        market_up = next_price > price

        # === Reward from agent predictions ===
        obs_vec = self.cached_obs[self.current_step]  # shape: (20,)
        reward = 0.0
        for i in range(5):
            onehot = obs_vec[i * 4: i * 4 + 3]
            confidence = obs_vec[i * 4 + 3]
            agent_action = np.argmax(onehot)

            if market_up and agent_action in [0, 1]:  # 预测涨
                reward += confidence * action[i]
            elif not market_up and agent_action == 2:  # 预测跌
                reward += confidence * action[i]
            else:
                reward -= confidence * action[i]

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Holding: {self.holding}, Entry Price: {self.entry_price}")


# === RL Training ===
from stable_baselines3 import PPO
from data_handle import data, cached_obs

env = AgentSelectEnv(data, cached_obs)
model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./tensorboard_logs/", device="cpu")
model.learn(total_timesteps=50000)

