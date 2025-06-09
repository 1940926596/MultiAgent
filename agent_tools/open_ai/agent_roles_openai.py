import json
from base_agent_openai import BaseFinanceAgent
from tools import function_schema
import pandas as pd

class TechnicalAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Technical Analyst"):
        super().__init__(name=name, role="Skilled at judging market trends through technical indicators", function_schema=function_schema, memory_size=10)
        self.interested_fields = ["close", "macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

    def analyze(self, data: dict) -> dict:
        
        prompt = (
            f"Below are the market technical indicators for {data['tic']} on {data['date']} \n\n"
            f"- Current Closing Price: {data['close']}\n"
            f"- MACD (Moving Average Convergence Divergence): {data['macd']}\n"
            f"- RSI (30-day Relative Strength Index): {data['rsi_30']}\n"
            f"- CCI (30-day Commodity Channel Index): {data['cci_30']}\n"
            f"- DMI (Directional Movement Index, 30-day): {data['dx_30']}\n"
            f"- 30-day Simple Moving Average: {data['close_30_sma']}\n"
            f"- 60-day Simple Moving Average: {data['close_60_sma']}\n\n"
            "Please judge based on historical trends: buy, sell, or hold? Output action, confidence, reasoning.\n"
            "Respond by calling the 'stock_decision' function according to the schema."
        )

        result = self.ask_model(prompt)
        self.update_history(data)
        return result

class RiskAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Risk Analyst"):
        super().__init__(name=name, role="Focuses on market risks such as volatility and turbulence", function_schema=function_schema,memory_size=10)
        self.interested_fields = ["vix", "turbulence"]

    def analyze(self, data: dict) -> dict:
        prompt = (
            f"- Date: {data.get('date', 'N/A')}\n"
            f"- VIX (CBOE Volatility Index, reflecting expected market volatility based on S&P 500 options): {data.get('vix', 'N/A')}\n"
            f"- Market Turbulence Index (A composite risk indicator calculated using asset price volatility and correlation among ~20 major U.S. companies): {data.get('turbulence', 'N/A')}\n\n"
            "Based on the above risk indicators, please assess the current risk level of the market.\n"
            "Should we take a conservative approach (e.g., hold or sell), or is it safe to buy?\n"
            "Please output: action, confidence, reasoning.\n"
            "Respond by calling the 'stock_decision' function according to the schema."
        )

        result = self.ask_model(prompt)
        self.update_history(data)
        return result
    

class SentimentAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Sentiment Analyst"):
        super().__init__(name=name,role="Focuses on news and public sentiment changes",function_schema=function_schema, memory_size=10)
        self.interested_fields = ["news_summary", "overall_sentiment", "key_points"]

    def analyze(self, data: dict) -> dict:
        if data.get("news_summary") and data.get("overall_sentiment") and not pd.isna(data.get("news_summary")) and not pd.isna(data.get("overall_sentiment")):
            
            prompt = (
                f"Below is a summary of news for {data['tic']} on {data['date']} \n\n"
                f"- News Summary (A concise summary of the key news points and sentiment trends for the day): {data.get('news_summary', 'No summary')}\n"
                f"- News Summary Key Points (List of key insights or important points extracted from the news): {data.get('key_points', 'No key points')}\n"
                f"- Overall Sentiment (Overall sentiment trend, e.g., Positive, Negative, Neutral): {data.get('overall_sentiment', 'N/A')}\n\n"
                "Please analyze the impact of recent news and sentiment on the stock's short-term movement.\n"
                "Based on the above information, output your investment decision.\n"
                "And fill in the following fields: action, confidence, reasoning.\n"
                "Respond by calling the 'stock_decision' function according to the schema."
            )

            result = self.ask_model(prompt)
            self.update_history(data)
            return result
        
        else:
            prompt = (
                f"- No available news and public sentiment for {data['tic']} on {data['date']}\n"
                "Please make your judgment based on historical information.\n"
                "Respond by calling the 'stock_decision' function according to the schema."
            )
            return self.ask_model(prompt)
        

# Process 10-Q(Quarterly) and 8-K(Irregular)
class MacroAnalystAgent(BaseFinanceAgent):
    def __init__(self, name="Macro Analyst"):
        super().__init__(name=name, role="Analyzes macroeconomic factors and external environment", function_schema=function_schema)
        self.interested_fields = ["form_type","macro_summary","risk_tag","macro_score"]

    def analyze(self, data: dict) -> dict:
        if data.get("macro_summary") and data.get("risk_tag") and not pd.isna(data.get("macro_summary")) and not pd.isna(data.get("risk_tag")):
            
            prompt = (
                f"- Date: {data.get('date', 'N/A')}\n"
                f"- {data.get('form_type','8-K/10-Q')} Macroeconomic Summary (Brief natural language summary of current macroeconomic conditions, based on the quarterly financial report or important events):\n{data.get('macro_summary', 'No macroeconomic summary available.')}\n"
                f"- Identified Risk Tags (Tags from a macroeconomic perspective from {data.get('form_type','8-K/10-Q')} risk sections such as 'High R&D expenditure', 'Slow revenue growth', 'Healthy financial condition', etc.): {data.get('risk_tag', [])}\n"
                f"- Macroeconomic Score (An overall health score of the company's operations for this quarter, ranging from 0 to 1): {data.get('macro_score', 'N/A')}\n\n"
                "Based on the macroeconomic conditions, monetary/fiscal policy signals, and global market factors.\n "
                "please provide an investment recommendation.\n"
                "Please output: action, confidence, reasoning.\n"
                "Respond by calling the 'stock_decision' function according to the schema."
            )

            result = self.ask_model(prompt)
            self.update_history(data)
            return result
            
        else:
            prompt = (
                f"- No available macro_summary and risk_tags for {data['tic']} on {data['date']}\n"
                "Please make your judgment based on historical information.\n"
                "Respond by calling the 'stock_decision' function according to the schema."
            )
            return self.ask_model(prompt)



# Analyze annual financial report and general public opinion
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
            content = "- No available financial report or macro summary today.\n"

        prompt = (
            f"{content}"
            "Please judge the company's value and investment potential based on the above information, "
            "and output action, confidence, reasoning."
            "Respond by calling the 'stock_decision' function according to the schema."
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
            reasons.append(f"[{advisor.name}]: (action: {action}) {result.get('reasoning', 'No reasoning')} (Confidence: {confidence})")

        final_action = max(votes, key=votes.get)
        total_conf = sum(votes.values())
        final_conf = round(votes[final_action] / total_conf, 3) if total_conf else 0.5

        return {
            "action": final_action,
            "confidence": final_conf,
            "reasoning": "\n\n".join(reasons),
        }



# run_analysis
if __name__ == "__main__":
    import pandas as pd
    from agent_roles_openai import TechnicalAnalystAgent, SentimentAnalystAgent, FundamentalAnalystAgent, CIOAgent ,MacroAnalystAgent

    # Load data
    df = pd.read_csv("../../datasets/processed/financial_with_news_macro_summary.csv")

    # Initialize agents
    tech = TechnicalAnalystAgent()
    sent = SentimentAnalystAgent()
    macro = MacroAnalystAgent()
    risk = RiskAnalystAgent()

    cio = CIOAgent(advisors=[tech, sent, macro, risk])

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

    # FundamentalAnalystAgent Deal With


