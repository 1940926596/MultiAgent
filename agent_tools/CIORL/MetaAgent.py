import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 环境模拟（Dummy，返回随机收益）
class DummyMarketEnv:
    def __init__(self):
        self.t = 0
        self.max_t = 1000

    def reset(self):
        self.t = 0
        state = np.random.randn(10)  # 假设状态是10维向量
        return state

    def step(self, action):
        # action: 0=Buy,1=Hold,2=Sell
        reward = np.random.randn() * (1 if action==0 else -1 if action==2 else 0.1)
        self.t += 1
        done = (self.t >= self.max_t)
        next_state = np.random.randn(10)
        return next_state, reward, done, {}

# 简单策略网络，输入状态，输出3个动作概率
class PolicyNet(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# REINFORCE训练过程
def train():
    env = DummyMarketEnv()
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    gamma = 0.99  # 折扣因子

    for episode in range(500):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # 计算折扣奖励
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # 计算损失
        loss = 0
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss -= log_prob * R  # REINFORCE目标是最大化期望回报

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, loss: {loss.item():.3f}, total_reward: {sum(rewards):.3f}")

if __name__ == "__main__":
    train()
