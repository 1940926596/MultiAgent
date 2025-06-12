import os
import sys
import numpy as np
import pandas as pd

class MetaAgent:
    def __init__(self, advisors, weights=None, use_softmax=True):
        self.advisors = advisors
        self.n = len(advisors)
        if weights is None:
            self.weights = np.ones(self.n) / self.n
        else:
            self.weights = np.array(weights)
        self.use_softmax = use_softmax

    def normalize_weights(self):
        if self.use_softmax:
            exp_weights = np.exp(self.weights)
            return exp_weights / np.sum(exp_weights)
        else:
            return self.weights / np.sum(self.weights)

    def analyze(self, data):
        action_space = ["buy", "hold", "sell"]
        action_scores = np.zeros(len(action_space))
        confidence_scores = np.zeros(len(action_space))
        explanations = []

        weights = self.normalize_weights()

        for i, agent in enumerate(self.advisors):
            result = agent.analyze(data)
            action = result["action"]
            conf = result["confidence"]
            reasoning = result["reasoning"]

            # 记录每个advisor的last_action，方便后续权重更新
            agent.last_action = action

            explanations.append(f"[{agent.__class__.__name__}] {action} ({conf}): {reasoning}")

            if action in action_space:
                idx = action_space.index(action)
                action_scores[idx] += weights[i] * conf
                confidence_scores[idx] += weights[i]

        final_scores = np.divide(
            action_scores,
            confidence_scores,
            out=np.zeros_like(action_scores),
            where=confidence_scores != 0
        )

        final_idx = np.argmax(final_scores)
        final_action = action_space[final_idx]
        final_confidence = final_scores[final_idx]

        return {
            "final_action": final_action,
            "final_confidence": round(float(final_confidence), 3),
            "explanations": "\n".join(explanations)
        }

class MetaAgentRL(MetaAgent):
    def __init__(self, advisors, weights=None, use_softmax=True, learning_rate=0.1):
        super().__init__(advisors, weights, use_softmax)
        self.learning_rate = learning_rate

    def update_weights(self, final_action, reward):
        action_space = ["buy", "hold", "sell"]

        for i, agent in enumerate(self.advisors):
            advisor_action = agent.last_action
            if advisor_action == final_action:
                self.weights[i] += self.learning_rate * reward
            else:
                self.weights[i] -= self.learning_rate * reward

        # 保证权重不小于0.01，防止消失
        self.weights = np.clip(self.weights, 0.01, None)
        # 归一化
        self.weights /= np.sum(self.weights)

    def analyze_and_learn(self, data, reward):
        result = self.analyze(data)
        final_action = result["final_action"]
        self.update_weights(final_action, reward)
        return result

# 简单回测奖励函数
def compute_reward(data, action):
    close_today = data["close"]
    close_next = data.get("close_next", close_today)
    price_diff = close_next - close_today

    if action == "buy":
        return 1 if price_diff > 0 else -1
    elif action == "sell":
        return 1 if price_diff < 0 else -1
    else:
        return 0

# --- 下面部分为你的实际使用示例 ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.agent_roles_openai import TechnicalAnalystAgent, SentimentAnalystAgent, FundamentalAnalystAgent, CIOAgent ,MacroAnalystAgent,RiskAnalystAgent

df = pd.read_csv("../../datasets/processed/financial_with_news_macro_summary.csv")

# 增加下一天close列，方便计算reward
df["close_next"] = df["close"].shift(-1)

tech = TechnicalAnalystAgent()
sent = SentimentAnalystAgent()
macro = MacroAnalystAgent()
risk = RiskAnalystAgent()

meta_rl = MetaAgentRL(advisors=[tech, sent, macro, risk], weights=[0.25, 0.25, 0.25, 0.25], learning_rate=0.05)

results = []
weights_history = []

for i, row in df.head(3).iterrows():
    data = row.to_dict()
    # 先根据当前权重做分析，获得动作
    result = meta_rl.analyze(data)
    # 根据动作和实际价格走势计算reward
    reward = compute_reward(data, result["final_action"])
    # 用reward更新权重
    meta_rl.update_weights(result["final_action"], reward)

    # 记录结果
    result["date"] = data["date"]
    result["tic"] = data["tic"]
    result["reward"] = reward
    results.append(result)
    weights_history.append(meta_rl.weights.copy())

results_df = pd.DataFrame(results)
weights_df = pd.DataFrame(weights_history, columns=["tech", "sent", "macro", "risk"])

results_df.to_csv("meta_agent_rl_results.csv", index=False)
weights_df.to_csv("meta_agent_rl_weights.csv", index=False)

print("强化学习 MetaAgentRL 训练完成，结果和权重历史已保存。")
