from agent import FinanceAgent
from datetime import datetime


class AgentGroup:
    def __init__(self, goal: str, agents: list[FinanceAgent]):
        self.goal = goal
        self.agents = agents
        self.history = []
        self.group_context = f"系统目标：{self.goal}"  # 初始共享上下文

    def iterate(self, n_rounds: int = 3):
        """多轮循环交流，模拟 agent 协同处理复杂任务"""
        for i in range(n_rounds):
            print(f"\n🌀 第 {i+1} 轮对话")
            round_responses = []

            for agent in self.agents:
                message = f"{self.group_context}\n请根据目前情况发表你的看法。"
                response = agent.chat(message)
                round_responses.append((agent.role, response))

                print(f"\n{agent.role} 回复：\n{response}")

            # 将这一轮对话添加到历史
            self.history.append({
                "type": "round",
                "round": i + 1,
                "responses": round_responses
            })

            # 汇总这轮所有 agent 的观点，更新 group_context
            summaries = [f"{role}：{resp}" for role, resp in round_responses]
            self.group_context += "\n" + "\n".join(summaries)

    def summarize(self):
        """由总经理人总结多轮对话后的结论"""
        manager = next((a for a in self.agents if "经理" in a.role), None)
        if manager is None:
            return "未找到总经理 Agent"

        decision = manager.chat(f"以下是本系统的多轮讨论内容：\n{self.group_context}\n\n请给出最终总结与建议。")
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
group = AgentGroup(goal="面对市场波动，给出最优投资建议", agents=agents)

group.iterate(n_rounds=3)

final_decision = group.summarize()
print("\n💼 最终总结：\n", final_decision)
