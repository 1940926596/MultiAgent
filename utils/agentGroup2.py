from agent import FinanceAgent
from datetime import datetime


class AgentGroup:
    def __init__(self, goal: str, agents: list[FinanceAgent]):
        self.goal = goal
        self.agents = agents
        self.history = []

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def broadcast(self, message: str):
        """所有 agent 同步接收信息，返回各自的响应。"""
        responses = []
        for agent in self.agents:
            response = agent.chat(f"系统目标：{self.goal}\n你收到一条信息：{message}")
            responses.append((agent.role, response))
        self.history.append({
            "type": "broadcast",
            "timestamp": self._now(),
            "message": message,
            "responses": responses
        })
        return responses

    def delegate(self, task: str, agent_role: str):
        """为特定 agent 分配任务。"""
        for agent in self.agents:
            if agent.role == agent_role:
                response = agent.chat(f"你被指派了一个任务：{task}")
                self.history.append({
                    "type": "delegate",
                    "timestamp": self._now(),
                    "agent": agent_role,
                    "task": task,
                    "response": response
                })
                return response
        return f"Agent {agent_role} not found."

    def summarize(self):
        """调用总经理 agent 总结当前状态并决策下一步。"""
        manager = next((a for a in self.agents if "经理" in a.role), None)
        if manager is None:
            return "未找到总经理 Agent"

        # 构造完整上下文
        context_lines = []
        context_lines.append(f"系统目标：{self.goal}")
        for h in self.history:
            if h["type"] == "broadcast":
                context_lines.append(f"[{h['timestamp']}] 📢 广播：{h['message']}")
                for role, resp in h["responses"]:
                    context_lines.append(f"    ↳ {role} 回复：{resp}")
            elif h["type"] == "delegate":
                context_lines.append(f"[{h['timestamp']}] 📌 指派任务给 {h['agent']}：{h['task']}")
                context_lines.append(f"    ↳ {h['agent']} 回复：{h['response']}")

        context = "\n".join(context_lines)

        # 提交给总经理人做总结
        decision = manager.chat(
            f"以下是当前系统中各专家的交流记录：\n{context}\n\n请你作为总经理人，总结目前信息并提出下一步行动建议。",
            reset_history=True
        )
        return decision





from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../models/Qwen-4B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)



# 创建测试 Agent
macro_analyst = FinanceAgent(role="宏观分析师", system_prompt="你是一位宏观经济分析师，擅长分析经济趋势、货币政策、地缘政治事件对市场的影响。趋势、货币政策、地缘政治事件对市场的影响。", model=qwen_model,tokenizer=tokenizer)
risk_controller = FinanceAgent(role="风险控制专家", system_prompt="你是一位风险控制专家，擅长评估金融风险与潜在危机", model=qwen_model, tokenizer=tokenizer)
asset_advisor = FinanceAgent(role="资产配置顾问", system_prompt="你是一位资产配置顾问，擅长根据市场形势调整投资组合", model=qwen_model, tokenizer=tokenizer)
manager = FinanceAgent(role="总经理人", system_prompt="你是一位总经理人，擅长协调各专家并做出最终决策", model=qwen_model, tokenizer=tokenizer)

# 放进 AgentGroup 中
agents = [macro_analyst, risk_controller, asset_advisor, manager]
# agents = [asset_advisor]
group = AgentGroup(goal="面对市场波动，给出最优投资建议", agents=agents)


# 所有 Agent 同步接收一条市场新闻
responses = group.broadcast("美联储宣布加息 25 个基点，全球市场震荡。")
for role, response in responses:
    print(f"\n{role} 的回复：\n{response}")


# 单独给资产配置顾问分配一个任务
response = group.delegate("请评估当前形势下最优资产配置策略。", agent_role="资产配置顾问")
print("\n资产配置顾问的回复：\n", response)


# 让总经理人基于前面记录的交流，进行总结与下一步建议
decision = group.summarize()
print("\n总经理人总结与决策：\n", decision)
