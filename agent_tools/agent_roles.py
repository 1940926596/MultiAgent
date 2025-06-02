from base_agent import BaseFinanceAgent
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_path = "../models/Qwen-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# === 通用输出抽取函数 ===

def extract_action_confidence_reason(text: str):
    """
    从模型输出中抽取 操作建议、理由 和 信心值。
    """
    try:
        action = "hold"
        confidence = 0.5
        reasoning = text.strip()

        # 抽取操作建议
        match_action = re.search(r"操作建议[:：]?\s*(买入|卖出|观望)", text)
        if match_action:
            action = {
                "买入": "buy",
                "卖出": "sell",
                "观望": "hold"
            }.get(match_action.group(1), "hold")

        # 抽取信心值（0~1 小数）
        match_conf = re.search(r"信心值[:：]?\s*(0\.\d+|1\.0+)", text)
        if match_conf:
            confidence = float(match_conf.group(1))
            if not (0 <= confidence <= 1):
                confidence = 0.5

        return action, confidence, reasoning

    except Exception as e:
        return "hold", 0.5, f"解析失败：{str(e)}，原始输出为：{text}"


# === Agent 模板类（建议所有 Agent 都继承用）===

class LLMDrivenAgent(BaseFinanceAgent):
    def ask_model(self, prompt: str) -> dict:
        raw_output = self.llm_chat(prompt)
        action, confidence, reasoning = extract_action_confidence_reason(raw_output)
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning
        }


# === 各类分析师 ===

class TechnicalAnalystAgent(LLMDrivenAgent):
    def analyze(self, data: dict) -> dict:
        macd = data.get("macd")
        rsi = data.get("rsi_30")
        close = data.get("close")

        prompt = (
            f"你是一位专业技术分析师。\n"
            f"当前股票价格为 {close}，MACD 为 {macd}，RSI 为 {rsi}。\n"
            "请你根据技术分析判断当前适合的操作，并回答：\n"
            "操作建议：<买入/卖出/观望>\n"
            "理由：<简洁明了，基于技术指标>\n"
            "信心值：<0-1 之间的小数>"
        )
        return self.ask_model(prompt)


class FundamentalAnalystAgent(LLMDrivenAgent):
    def analyze(self, data: dict) -> dict:
        fields = data.get("all_fields", "").strip()
        if not fields:
            return {
                "action": "hold",
                "confidence": 0.4,
                "reasoning": "未提供财报信息，默认选择观望"
            }

        prompt = (
            "你是一位基本面分析师。\n"
            f"以下是部分财报内容：\n{fields}\n\n"
            "请判断当前公司的基本面状况并给出操作建议：\n"
            "操作建议：<买入/卖出/观望>\n"
            "理由：<简洁说明核心财务指标或趋势>\n"
            "信心值：<0-1 的小数>"
        )
        return self.ask_model(prompt)


class SentimentAnalystAgent(LLMDrivenAgent):
    def analyze(self, data: dict) -> dict:
        sentiment = data.get("sentiment", 0)
        news = data.get("news_text", "")

        prompt = (
            f"你是一位情绪分析师。\n"
            f"当前新闻情绪得分为 {sentiment}。\n新闻内容如下：\n{news}\n\n"
            "请你综合判断市场情绪对股价的影响，并输出：\n"
            "操作建议：<买入/卖出/观望>\n"
            "理由：<情绪影响判断依据>\n"
            "信心值：<0-1 的小数>"
        )
        return self.ask_model(prompt)


class RiskControlAgent(LLMDrivenAgent):
    def analyze(self, data: dict) -> dict:
        vix = data.get("vix")
        turbulence = data.get("turbulence")

        prompt = (
            f"你是一位风险控制专家。\n"
            f"当前 VIX（市场波动率）为 {vix}，Turbulence（系统动荡）为 {turbulence}。\n"
            "请评估当前风险状况并给出建议：\n"
            "操作建议：<买入/卖出/观望>\n"
            "理由：<风险因素及影响>\n"
            "信心值：<0-1 的小数>"
        )
        return self.ask_model(prompt)


# === CIO ===

class CIOAgent(BaseFinanceAgent):
    def __init__(self, name, role, advisors):
        super().__init__(name, role, model, tokenizer)
        self.advisors = advisors

    def analyze(self, data: dict) -> dict:
        votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        reasons = []

        for advisor in self.advisors:
            res = advisor.analyze(data)
            votes[res["action"]] += res["confidence"]
            reasons.append(f"{advisor.name} 的建议：{res['reasoning']}")

        final_action = max(votes, key=votes.get)
        total_conf = sum(votes.values())
        final_conf = round(votes[final_action] / total_conf, 3) if total_conf else 0.5

        return {
            "action": final_action,
            "confidence": final_conf,
            "reasoning": "\n\n".join(reasons)
        }


# === 实例化 Agent ===

tech = TechnicalAnalystAgent("技术分析师", "负责技术分析", model, tokenizer)
fund = FundamentalAnalystAgent("基本面分析师", "负责基本面分析", model, tokenizer)
sent = SentimentAnalystAgent("情绪分析师", "负责舆情分析", model, tokenizer)
risk = RiskControlAgent("风控专家", "负责风险评估", model, tokenizer)

cio = CIOAgent("CIO", "总决策者", [tech, fund, sent, risk])
agent_list = [tech, fund, sent, risk, cio]
