from collections import deque, defaultdict
import datetime
import numpy as np


# 短期记忆（STM, Short-Term Memory）：存储最近几天（如30天）的市场状态和 Agent 输出，用于当下决策。
# 长期记忆（LTM, Long-Term Memory）：积累历史表现（如90天或更长），便于总结、趋势判断和未来任务参考。

class MemorySystem:
    def __init__(self, short_term_days=30, long_term_days=90):
        # 保存每日市场状态和 agent 行为轨迹
        self.market_memory = {}  # {date: market_state}
        self.agent_memory = {}   # {agent: [ {date, market_state, action, reward, confidence}, ... ]}

        self.short_term_days = short_term_days
        self.long_term_days = long_term_days

    def log_market_state(self, date, market_data):
        self.market_memory[date] = market_data
        # 记录每一天的市场特征，如波动率（turbulence）、新闻情绪、技术指标等。

    def log_agent_performance(self, agent, date, action, reward, confidence):
        record = {
            "date": date,
            "market_state": self.market_memory.get(date, {}),
            "action": action,
            "reward": reward,
            "confidence": confidence
        }
        self.agent_memory.setdefault(agent, []).append(record)
        # 为指定 Agent 记录其当天表现：执行的操作（buy/sell/hold）、实际收益（reward）、模型信心值等。


    def get_agent_memory(self, agent, days=None, since_date=None):
        """
        获取某个 agent 的短期或长期记忆
        """
        records = self.agent_memory.get(agent, [])
        if since_date:
            records = [r for r in records if r['date'] >= since_date]
        elif days:
            cutoff = datetime.strptime(records[-1]['date'], "%Y-%m-%d") - datetime.timedelta(days=days)
            records = [r for r in records if datetime.strptime(r['date'], "%Y-%m-%d") >= cutoff]
        return records
 

    def get_agent_recent_accuracy(self, agent, days=10):
        memory = self.get_agent_memory(agent, days=days)
        if not memory:
            return 0.0
        wins = [r for r in memory if r["reward"] > 0]
        return len(wins) / len(memory)

    def get_agent_avg_confidence(self, agent, days=10):
        memory = self.get_agent_memory(agent, days=days)
        if not memory:
            return 0.0
        return np.mean([r["confidence"] for r in memory])

    def get_top_k_agents(self, k=2, metric="accuracy", days=10):
        stats = []
        for agent in self.agent_memory:
            if metric == "accuracy":
                value = self.get_agent_recent_accuracy(agent, days)
            elif metric == "confidence":
                value = self.get_agent_avg_confidence(agent, days)
            else:
                continue
            stats.append((agent, value))
        stats.sort(key=lambda x: x[1], reverse=True)
        return stats[:k]

    def get_market_volatility_trend(self, window=7):
        dates = sorted(self.market_memory.keys())[-window:]
        values = [self.market_memory[d]["turbulence"] for d in dates if "turbulence" in self.market_memory[d]]
        return np.mean(values) if values else 0.0
