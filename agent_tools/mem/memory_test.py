from datetime import datetime, timedelta
from memory_system import MemorySystem
import numpy as np

# Initialize memory system
memory = MemorySystem(short_term_days=30, long_term_days=90)

# Define agent names
agent_names = ['TechAnalyst', 'FundamentalAnalyst', 'SentimentAnalyst']
start_date = datetime.strptime("2025-01-01", "%Y-%m-%d")

# Simulate 10 days of data
for i in range(10):
    current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    
    # Log market state for the day
    market_data = {
        "vix": np.random.uniform(15, 25),
        "turbulence": np.random.uniform(30, 80),
        "avg_news_sentiment": np.random.uniform(-1, 1),
        "macd": np.random.uniform(-0.05, 0.05),
        "rsi_30": np.random.uniform(30, 70)
    }
    memory.log_market_state(current_date, market_data)

    # Log performance of each agent
    for agent in agent_names:
        action = np.random.choice(['buy', 'sell', 'hold'])
        reward = np.random.normal(0.01, 0.02)  # Simulated reward (can be positive or negative)
        confidence = np.random.uniform(0.5, 1.0)
        memory.log_agent_performance(agent, current_date, action, reward, confidence)

# Display market volatility trend
print("7-day average turbulence:", memory.get_market_volatility_trend(window=7))

# Show recent performance stats for each agent
for agent in agent_names:
    print(f"\n{agent} recent performance:")
    print("Win rate:", memory.get_agent_recent_accuracy(agent))
    print("Average confidence:", memory.get_agent_avg_confidence(agent))

# Top-K agents by different metrics
print("\nTop 2 agents by win rate:")
print(memory.get_top_k_agents(k=2, metric="accuracy"))

print("\nTop 2 agents by confidence:")
print(memory.get_top_k_agents(k=2, metric="confidence"))


class MetaController:
    def __init__(self, memory_system, min_accuracy=0.6, min_confidence=0.7):
        self.memory = memory_system
        self.min_accuracy = min_accuracy
        self.min_confidence = min_confidence

    def select_active_agents(self, current_date, available_agents):
        """
        Select a set of active agents for today based on market state and historical performance.
        """
        active_agents = []

        market_trend = self.memory.get_market_volatility_trend()
        print(f"[MetaController] Current 7-day average turbulence: {market_trend:.2f}")

        for agent in available_agents:
            acc = self.memory.get_agent_recent_accuracy(agent)
            conf = self.memory.get_agent_avg_confidence(agent)
            print(f"{agent} - Accuracy: {acc:.2f} / Confidence: {conf:.2f}")

            if acc >= self.min_accuracy and conf >= self.min_confidence:
                active_agents.append(agent)

        if not active_agents:
            # Fallback: ensure at least one agent is selected
            print("No agent meets the criteria, selecting the one with highest confidence.")
            top_agent = self.memory.get_top_k_agents(k=1, metric="confidence")
            if top_agent:
                active_agents = [top_agent[0][0]]

        return active_agents


# Instantiate the MetaController
meta_controller = MetaController(memory)

# Simulate control decision on January 10
selected_agents = meta_controller.select_active_agents("2025-01-10", agent_names)

print(f"\nSelected agents for today: {selected_agents}")
