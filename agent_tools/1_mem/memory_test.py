from datetime import datetime, timedelta
from memory_system import MemorySystem
import numpy as np

memory = MemorySystem(short_term_days=30, long_term_days=90)


agent_names = ['TechAnalyst', 'FundamentalAnalyst', 'SentimentAnalyst']
start_date = datetime.strptime("2025-01-01", "%Y-%m-%d")

# 模拟 10 天数据
for i in range(10):
    current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    
    # 市场状态记录
    market_data = {
        "vix": np.random.uniform(15, 25),
        "turbulence": np.random.uniform(30, 80),
        "avg_news_sentiment": np.random.uniform(-1, 1),
        "macd": np.random.uniform(-0.05, 0.05),
        "rsi_30": np.random.uniform(30, 70)
    }
    memory.log_market_state(current_date, market_data)

    # 每个 agent 的表现记录
    for agent in agent_names:
        action = np.random.choice(['buy', 'sell', 'hold'])
        reward = np.random.normal(0.01, 0.02)  # 模拟收益（可以为正为负）
        confidence = np.random.uniform(0.5, 1.0)
        memory.log_agent_performance(agent, current_date, action, reward, confidence)


# 市场波动趋势
print("📈 最近7日turbulence均值：", memory.get_market_volatility_trend(window=7))

# 每个 agent 最近10天的胜率和置信度
for agent in agent_names:
    print(f"\n📊 {agent} 最近表现：")
    print("✅ 胜率：", memory.get_agent_recent_accuracy(agent))
    print("🤝 平均置信度：", memory.get_agent_avg_confidence(agent))

# top-K agents
print("\n🏆 胜率最高的前2名 agents：")
print(memory.get_top_k_agents(k=2, metric="accuracy"))

print("\n💡 置信度最高的前2名 agents：")
print(memory.get_top_k_agents(k=2, metric="confidence"))



class MetaController:
    def __init__(self, memory_system, min_accuracy=0.6, min_confidence=0.7):
        self.memory = memory_system
        self.min_accuracy = min_accuracy
        self.min_confidence = min_confidence

    def select_active_agents(self, current_date, available_agents):
        """
        基于当前市场状态和agent历史表现，返回今日应激活的Agent集合
        """
        active_agents = []

        market_trend = self.memory.get_market_volatility_trend()
        print(f"[MetaController] 当前7日市场turbulence均值: {market_trend:.2f}")

        for agent in available_agents:
            acc = self.memory.get_agent_recent_accuracy(agent)
            conf = self.memory.get_agent_avg_confidence(agent)
            print(f"🧠 {agent} - 胜率: {acc:.2f} / 信心: {conf:.2f}")

            if acc >= self.min_accuracy and conf >= self.min_confidence:
                active_agents.append(agent)

        if not active_agents:
            # 保底机制，至少保留1个 Agent
            print("⚠️ 无Agent满足条件，选择置信度最高的一个")
            top_agent = self.memory.get_top_k_agents(k=1, metric="confidence")
            if top_agent:
                active_agents = [top_agent[0][0]]

        return active_agents




meta_controller = MetaController(memory)

# 假设 1 月 10 日进行控制决策
selected_agents = meta_controller.select_active_agents("2025-01-10", agent_names)


print(f"\n✅ 今日激活的智能体: {selected_agents}")
