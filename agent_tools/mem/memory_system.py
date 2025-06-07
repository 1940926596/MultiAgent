from collections import deque, defaultdict
import numpy as np

class MemorySystem:
    def __init__(self, short_term_days=30, long_term_days=90):
        self.market_memory = deque(maxlen=short_term_days)  # 最近30天市场状态
        self.agent_memory = defaultdict(lambda: deque(maxlen=long_term_days))  # 每个agent 90天记录
        self.agent_summary = {}  # 汇总每个agent统计信息：胜率、平均收益、置信度

    ### ===== 市场记忆模块 =====
    def log_market_state(self, date, market_data: dict):
        """
        market_data = {
            'vix': 17.2, 'turbulence': 56.1,
            'avg_news_sentiment': 0.42,
            'macd': 0.01, 'rsi_30': 65,
            ...
        }
        """
        self.market_memory.append({
            "date": date,
            "data": market_data
        })

    def get_market_volatility_trend(self, window=7):
        values = [x["data"]["turbulence"] for x in list(self.market_memory)[-window:] if "turbulence" in x["data"]]
        return np.mean(values) if values else 0

    ### ===== Agent记忆模块 =====
    def log_agent_performance(self, agent_name, date, action, reward, confidence):
        self.agent_memory[agent_name].append({
            "date": date,
            "action": action,
            "reward": reward,
            "confidence": confidence
        })

    def get_agent_recent_accuracy(self, agent_name, window=10):
        history = list(self.agent_memory[agent_name])[-window:]
        correct = sum(1 for h in history if h["reward"] > 0)
        return correct / window if window and len(history) >= window else 0

    def get_agent_avg_confidence(self, agent_name, window=10):
        history = list(self.agent_memory[agent_name])[-window:]
        confs = [h["confidence"] for h in history if "confidence" in h]
        return np.mean(confs) if confs else 0

    def get_top_k_agents(self, k=3, metric="accuracy"):
        scores = {}
        for agent in self.agent_memory.keys():
            if metric == "accuracy":
                scores[agent] = self.get_agent_recent_accuracy(agent)
            elif metric == "confidence":
                scores[agent] = self.get_agent_avg_confidence(agent)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
