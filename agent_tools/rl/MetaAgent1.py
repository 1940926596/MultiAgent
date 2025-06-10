import os
import sys
import numpy as np
import pandas as pd

import numpy as np

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


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.agent_roles_openai import TechnicalAnalystAgent, SentimentAnalystAgent, FundamentalAnalystAgent, CIOAgent ,MacroAnalystAgent,RiskAnalystAgent

# Load data
df = pd.read_csv("../../datasets/processed/financial_with_news_macro_summary.csv")

# Initialize advisors
tech = TechnicalAnalystAgent()
sent = SentimentAnalystAgent()
macro = MacroAnalystAgent()
risk = RiskAnalystAgent()

# 替换原来的 CIOAgent
meta = MetaAgent(advisors=[tech, sent, macro, risk], weights=[0.25, 0.25, 0.25, 0.25])  # 权重可自定义

results = []
for i, row in df.head(3).iterrows():  # 这里可以改成你想分析的日期范围
    data = row.to_dict()
    result = meta.analyze(data)
    result["date"] = data["date"]
    result["tic"] = data["tic"]
    results.append(result)

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("meta_agent_analysis_results.csv", index=False)
print("MetaAgent 多智能体分析完成，结果已保存至: meta_agent_analysis_results.csv")