from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

class AgentWeightEnv(Env):
    def __init__(self, data, agent_outputs, market_labels):
        self.data = data  # 市场特征
        self.agent_outputs = agent_outputs  # 各 agent 的 action+confidence
        self.market_labels = market_labels  # 实际涨跌或收益 label
        self.index = 0

        # 状态空间：假设 agent_outputs 是 5×4 向量，market 特征是 15维
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
        # 动作空间：5个 agent 权重（连续），我们用 Box，再归一化为 softmax
        self.action_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self):
        self.index = 0
        return self._get_state()

    def _get_state(self):
        agent_vec = self.agent_outputs[self.index].flatten()
        market_vec = self.data[self.index]
        return np.concatenate([agent_vec, market_vec])

    def step(self, action):
        action = np.clip(action, 0, 1)
        weights = action / (np.sum(action) + 1e-8)

        agent_actions = self.agent_outputs[self.index][:, 0:3]  # 取 one-hot 行为
        agent_confidence = self.agent_outputs[self.index][:, 3]

        weighted_decision = np.sum(agent_actions.T * weights * agent_confidence, axis=1)  # 3维向量
        final_action = np.argmax(weighted_decision)

        # reward: 与真实 market 方向比对
        market_move = self.market_labels[self.index]  # 0/1/2 = buy/sell/hold
        reward = 1.0 if final_action == market_move else -0.5

        self.index += 1
        done = self.index >= len(self.data)

        return self._get_state() if not done else None, reward, done, {}


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

env = DummyVecEnv([lambda: AgentWeightEnv(market_states, agent_outputs, true_labels)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)