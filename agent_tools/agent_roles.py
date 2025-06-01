# agent_roles.py

from base_agent import BaseFinanceAgent

# 技术分析师
class TechnicalAnalystAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        close = data.get("close")
        macd = data.get("macd")
        rsi = data.get("rsi_30")
        # 这里可以加策略逻辑或模型调用

        # 示例逻辑
        if macd > 0 and rsi < 70:
            action = "buy"
            confidence = 0.8
            reason = "MACD 上穿 0 且 RSI 未超买，短期看涨"
        else:
            action = "hold"
            confidence = 0.5
            reason = "指标未明显趋势"

        return {"action": action, "confidence": confidence, "reasoning": reason}


# 基本面分析师
class FundamentalAnalystAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        fields = data.get("all_fields", "")
        # TODO: 用 NLP 提取财报关键词（如收入/ROE 增长等）

        action = "hold"
        confidence = 0.6
        reason = "尚未解析详细财报字段"

        return {"action": action, "confidence": confidence, "reasoning": reason}


# 情绪分析 Agent
class SentimentAnalystAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        sentiment = data.get("sentiment", 0)
        news = data.get("news_text", "")

        if sentiment > 0.2:
            action = "buy"
            confidence = 0.75
            reason = "新闻情绪积极，情绪驱动上涨概率大"
        elif sentiment < -0.2:
            action = "sell"
            confidence = 0.75
            reason = "新闻情绪极度负面，存在下跌风险"
        else:
            action = "hold"
            confidence = 0.5
            reason = "情绪中性，影响不明显"

        return {"action": action, "confidence": confidence, "reasoning": reason}


# 风控 Agent
class RiskControlAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        vix = data.get("vix")
        turbulence = data.get("turbulence")

        if vix > 25 or turbulence > 120:
            action = "sell"
            confidence = 0.9
            reason = "市场波动率和系统风险过高，应降低风险暴露"
        else:
            action = "hold"
            confidence = 0.6
            reason = "风险可控，可维持仓位"

        return {"action": action, "confidence": confidence, "reasoning": reason}


# 资产配置 Agent（可选）
class AssetAllocatorAgent(BaseFinanceAgent):
    def analyze(self, data: dict) -> dict:
        # 预留，当前为单资产策略可忽略
        return {"action": "hold", "confidence": 0.0, "reasoning": "资产配置功能未启用"}


# 总决策 CIO Agent
class CIOAgent(BaseFinanceAgent):
    def __init__(self, name: str, role: str, advisors: list):
        super().__init__(name, role)
        self.advisors = advisors  # List of other Agent objects

    def analyze(self, data: dict) -> dict:
        decisions = []
        for agent in self.advisors:
            result = agent.analyze(data)
            decisions.append(result)

        # 简单加权（可以替换为更复杂的策略）
        scores = {"buy": 0, "hold": 0, "sell": 0}
        for res in decisions:
            scores[res["action"]] += res["confidence"]

        action = max(scores, key=scores.get)
        reasoning = "\n".join(
            [f"{agent.name}: {res['reasoning']}" for agent, res in zip(self.advisors, decisions)]
        )

        return {
            "action": action,
            "confidence": scores[action] / len(self.advisors),
            "reasoning": reasoning,
        }

