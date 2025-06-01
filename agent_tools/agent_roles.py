# agent_roles.py
from base_agent import BaseFinanceAgent
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../models/Qwen-4B"  # 替换成你的实际路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

### 技术分析 Agent
class TechnicalAnalystAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        macd = data.get("macd")
        rsi = data.get("rsi_30")
        close = data.get("close")

        # 也可以用 prompt 调大模型
        prompt = f"当前 MACD 值为 {macd}，RSI 为 {rsi}，股价为 {close}。请根据技术指标分析后市趋势，并给出建议（买入、卖出、观望）及理由。"
        answer = self.llm_chat(prompt)

        return {"action": "llm", "confidence": 1.0, "reasoning": answer}

### 基本面分析 Agent
class FundamentalAnalystAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        fields = data.get("all_fields", "")
        prompt = f"以下是公司的部分财报内容：{fields}\n请你判断公司基本面是否良好，并给出投资建议和理由。"
        answer = self.llm_chat(prompt)

        return {"action": "llm", "confidence": 1.0, "reasoning": answer}

### 情绪分析 Agent
class SentimentAnalystAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        sentiment = data.get("sentiment", 0)
        news = data.get("news_text", "")
        prompt = f"新闻内容如下：{news}\n新闻情绪打分为 {sentiment}。请判断市场情绪，并给出投资建议和理由。"
        answer = self.llm_chat(prompt)

        return {"action": "llm", "confidence": 1.0, "reasoning": answer}

### 风控 Agent
class RiskControlAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        vix = data.get("vix")
        turbulence = data.get("turbulence")
        prompt = f"当前 VIX 为 {vix}，市场动荡指标为 {turbulence}。请评估市场风险并给出操作建议。"
        answer = self.llm_chat(prompt)

        return {"action": "llm", "confidence": 1.0, "reasoning": answer}

### CIO 决策 Agent
class CIOAgent(BaseFinanceAgent):
    def __init__(self, name, role, advisors):
        super().__init__(name, role, model, tokenizer)
        self.advisors = advisors

    def analyze(self, data: dict) -> dict:
        results = []
        for advisor in self.advisors:
            results.append(advisor.analyze(data))

        # 简单投票决策
        votes = {"buy": 0, "sell": 0, "hold": 0, "llm": 0}
        for res in results:
            votes[res["action"]] += 1

        final_action = max(votes, key=votes.get)
        reasoning = "\n".join(
            [f"{a.name}: {res['reasoning']}" for a, res in zip(self.advisors, results)]
        )

        return {
            "action": final_action,
            "confidence": 1.0,
            "reasoning": reasoning
        }

# 实例化
tech = TechnicalAnalystAgent("技术分析师", "负责技术指标分析", model, tokenizer)
fund = FundamentalAnalystAgent("基本面分析师", "负责财报分析", model, tokenizer)
sent = SentimentAnalystAgent("情绪分析师", "负责舆情分析", model, tokenizer)
risk = RiskControlAgent("风控专家", "负责风险评估", model, tokenizer)

cio = CIOAgent("CIO", "总决策者", [tech, fund, sent, risk])

agent_list = [tech, fund, sent, risk, cio]

