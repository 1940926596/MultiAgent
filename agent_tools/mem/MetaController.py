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


