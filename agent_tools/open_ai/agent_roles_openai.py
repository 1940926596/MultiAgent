import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.web_search import search_wikipedia_summary,search_industry_news
from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent
from agent_tools.open_ai.tools import function_schema,function_schema1
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
        super().__init__(
            name=name,
            role="Analyzes company fundamentals to assess long-term intrinsic value and financial health",
            function_schema=function_schema1
        )
        self.interested_fields = [
            "Total Revenue", "Net Income", "EBITDA", "Operating Income", 
            "R&D", "SG&A", "Total Assets", "Total Liabilities", 
            "Long Term Debt", "Current Liabilities", "Cash and Equivalents", 
            "Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"
        ]

    def analyze(self, data: dict) -> dict:
        tic = data.get("tic", "")
        year = data.get("date", "")[:4]

        # 1. 获取维基百科简介
        wiki_summary = search_wikipedia_summary(tic)

        # 2. 获取行业新闻摘要
        industry_news = search_industry_news(tic, year=year)

        prompt = (
            f"You are a professional financial analyst. Please analyze the fundamental situation of company {tic} for the year {year}.\n"
            f"Use the following financial data:\n"
            f"- Total Revenue: {data.get('Total Revenue', 'N/A')}\n"
            f"- Net Income: {data.get('Net Income', 'N/A')}\n"
            f"- EBITDA: {data.get('EBITDA', 'N/A')}\n"
            f"- Operating Income: {data.get('Operating Income', 'N/A')}\n"
            f"- R&D Spending: {data.get('R&D', 'N/A')}\n"
            f"- SG&A Expenses: {data.get('SG&A', 'N/A')}\n"
            f"- Total Assets: {data.get('Total Assets', 'N/A')}\n"
            f"- Total Liabilities: {data.get('Total Liabilities', 'N/A')}\n"
            f"- Long-Term Debt: {data.get('Long Term Debt', 'N/A')}\n"
            f"- Current Liabilities: {data.get('Current Liabilities', 'N/A')}\n"
            f"- Cash and Equivalents: {data.get('Cash and Equivalents', 'N/A')}\n"
            f"- Operating Cash Flow: {data.get('Operating Cash Flow', 'N/A')}\n"
            f"- Free Cash Flow: {data.get('Free Cash Flow', 'N/A')}\n"
            f"- Capital Expenditure: {data.get('Capital Expenditure', 'N/A')}\n\n"
            f"Additional Information:\n"
            f"- Wikipedia Summary: {wiki_summary}\n"
            f"- Industry Trends & News: {industry_news}\n\n"
            "Now summarize:\n"
            "1. Financial Summary\n"
            "2. Company Profile\n"
            "3. Industry Context\n"
            "4. Risk Assessment\n"
            "5. Confidence (0-1)\n\n"
            "Respond by calling the 'fundamental_analysis_report' function according to the schema."
        )
        result = self.ask_model(prompt)
        # self.update_history(data)
        return result
    


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
    from agent_roles_openai import TechnicalAnalystAgent, SentimentAnalystAgent, FundamentalAnalystAgent, CIOAgent ,MacroAnalystAgent,RiskAnalystAgent

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
    for i, row in df.iterrows():
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



    # # FundamentalAnalystAgent Deal With
    # fun_df = pd.read_csv("../../datasets/fundamentals/AAPL_fundamentals_enhanced.csv")
    # fun=FundamentalAnalystAgent()
    # fun_results = []
    # for i, row in fun_df.iterrows():
    #     data = row.to_dict()
    #     fun_result = fun.analyze(data)
    #     fun_result["date"] = data["date"]
    #     fun_result["tic"] = data["tic"]
    #     fun_results.append(fun_result)

    # results_df = pd.DataFrame(fun_results)
    # results_df = results_df.sort_values(by='date')
    # results_df.to_csv("longterm_analysis_results.csv", index=False)
    # print("results saved to: longterm_analysis_results.csv")
