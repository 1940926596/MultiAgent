import json
from base_agent_openai import BaseFinanceAgent
from tools import function_schema

class TechnicalAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Technical Analyst"):
        super().__init__(name=name, role="Skilled at judging market trends through technical indicators", function_schema=function_schema)
        self.interested_fields = ["close", "macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

    def analyze(self, data: dict) -> dict:
        self.update_history(data)
        prompt = (
            f"- Current Price: {data['close']}\n"
            f"- MACD: {data['macd']}\n"
            f"- RSI(30): {data['rsi_30']}\n"
            f"- CCI(30): {data['cci_30']}\n"
            f"- DMI(dx_30): {data['dx_30']}\n"
            f"- 30-day Moving Average: {data['close_30_sma']}\n"
            f"- 60-day Moving Average: {data['close_60_sma']}\n"
            "Please judge based on historical trends: buy, sell, or hold? Output action, confidence, reasoning."
        )
        return self.ask_model(prompt)


class SentimentAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Sentiment Analyst"):
        super().__init__(name=name, role="Focuses on news and public sentiment changes", function_schema=function_schema)
        self.interested_fields = ["sentiment", "news_text"]

    def analyze(self, data: dict) -> dict:
        self.update_history(data)
        prompt = (
            f"Below is a summary of news for {data['tic']} on {data['date']} (each includes sentiment and headline):\n\n"
            f"{data['news']}\n\n"
            "Please analyze the potential impact of these news sentiments on the market, judge the recommended action (buy, sell, hold), "
            "and fill in the following fields: action, confidence, reasoning."
        )
        return self.ask_model(prompt)


class FundamentalAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Fundamental Analyst"):
        super().__init__(name=name, role="Judges company value based on financial reports", function_schema=function_schema)
        self.interested_fields = ["form_type", "all_fields"]

    def analyze(self, data: dict) -> dict:
        self.update_history(data)

        # If there is a financial report on the day, prioritize using the report for analysis
        if data.get("form_type") and data.get("all_fields"):
            all_fields_str = json.dumps(data['all_fields'], ensure_ascii=False)
            summary = all_fields_str[:300] + '...' if len(all_fields_str) > 300 else all_fields_str

            content = (
                f"- Date: {data.get('date', 'N/A')}\n"
                f"- Report Type: {data['form_type']}\n"
                f"- Financial Report Summary: {summary}...\n"
            )
            print(content)
        else:
            # When there is no specific report or macro summary
            content = "- No available financial report or macro summary\n"

        prompt = (
            f"{content}"
            "Please judge the company's value and investment potential based on the above information, "
            "and output action, confidence, reasoning."
        )
        return self.ask_model(prompt)



class CIOAgent(BaseFinanceAgent):
    def __init__(self, name="CIO", advisors=None):
        super().__init__(name=name, role="Responsible for integrating analysts' suggestions to form final trading decisions")
        self.advisors = advisors or []

    def analyze(self, data: dict) -> dict:
        votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        reasons = []

        for advisor in self.advisors:
            result = advisor.analyze(data)
            action = result.get("action", "hold")
            confidence = result.get("confidence", 0.5)

            votes[action] += confidence
            reasons.append(f"[{advisor.name}]: {result.get('reasoning', 'No reasoning')} (Confidence: {confidence})")

        final_action = max(votes, key=votes.get)
        total_conf = sum(votes.values())
        final_conf = round(votes[final_action] / total_conf, 3) if total_conf else 0.5

        return {
            "action": final_action,
            "confidence": final_conf,
            "reasoning": "\n\n".join(reasons),
        }



# run_analysis

import pandas as pd
from agent_roles_openai import TechnicalAnalystAgent, SentimentAnalystAgent, FundamentalAnalystAgent, CIOAgent

# Load data
df = pd.read_csv("/data/postgraduates/2024/chenjiarui/Fin/MultiAgents/datasets/processed/financial_final.csv")

# Initialize agents
tech = TechnicalAnalystAgent()
sent = SentimentAnalystAgent()
fund = FundamentalAnalystAgent()
cio = CIOAgent(advisors=[tech, sent, fund])

# Test on first few rows
results = []
for i, row in df.head(3).iterrows():
    data = row.to_dict()
    result = cio.analyze(data)
    # print(result.__str__)
    result["date"] = data["date"]
    result["tic"] = data["tic"]
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("cio_analysis_results.csv", index=False)
print("Multi-agent analysis complete, results saved to: cio_analysis_results.csv")
