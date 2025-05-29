import yfinance as yf
import pandas as pd
import os

# Set the proxy URL and port
proxy_url = 'http://127.0.0.1'
proxy_port = '7890'
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'


def get_fundamentals(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)

    income_stmt = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T
    
    start_year=2021
    end_year=2024

    # 筛选时间段，index是Timestamp，转为datetime
    income_stmt = income_stmt[(income_stmt.index.year >= start_year) & (income_stmt.index.year <= end_year)]
    balance_sheet = balance_sheet[(balance_sheet.index.year >= start_year) & (balance_sheet.index.year <= end_year)]
    cash_flow = cash_flow[(cash_flow.index.year >= start_year) & (cash_flow.index.year <= end_year)]


    # 安全提取列
    def safe_get(df, key):
        return df[key] if key in df.columns else pd.Series([None] * len(df), index=df.index)

    # 基础财报字段
    fundamentals = pd.DataFrame({
        "Total Revenue": safe_get(income_stmt, "Total Revenue"),
        "Net Income": safe_get(income_stmt, "Net Income"),
        "EBITDA": safe_get(income_stmt, "EBITDA"),
        "Operating Income": safe_get(income_stmt, "Operating Income"),
        "R&D": safe_get(income_stmt, "Research And Development"),
        "SG&A": safe_get(income_stmt, "Selling General And Administration"),

        "Total Assets": safe_get(balance_sheet, "Total Assets"),
        "Total Liabilities": safe_get(balance_sheet, "Total Liabilities Net Minority Interest"),
        "Long Term Debt": safe_get(balance_sheet, "Long Term Debt"),
        "Current Liabilities": safe_get(balance_sheet, "Current Liabilities"),
        "Cash and Equivalents": safe_get(balance_sheet, "Cash And Cash Equivalents"),
        "Stockholders Equity": safe_get(balance_sheet, "Stockholders Equity"),

        "Operating Cash Flow": safe_get(cash_flow, "Operating Cash Flow"),
        "Free Cash Flow": safe_get(cash_flow, "Free Cash Flow"),
        "Capital Expenditure": safe_get(cash_flow, "Capital Expenditure"),
    })

    # 增强指标计算
    fundamentals["Net Profit Margin"] = fundamentals["Net Income"] / fundamentals["Total Revenue"]
    fundamentals["Operating Margin"] = fundamentals["Operating Income"] / fundamentals["Total Revenue"]
    fundamentals["R&D Intensity"] = fundamentals["R&D"] / fundamentals["Total Revenue"]
    fundamentals["SG&A Ratio"] = fundamentals["SG&A"] / fundamentals["Total Revenue"]
    fundamentals["ROA"] = fundamentals["Net Income"] / fundamentals["Total Assets"]
    fundamentals["ROE"] = fundamentals["Net Income"] / fundamentals["Stockholders Equity"]
    fundamentals["Debt to Equity"] = fundamentals["Total Liabilities"] / fundamentals["Stockholders Equity"]
    fundamentals["Long-term Debt Ratio"] = fundamentals["Long Term Debt"] / fundamentals["Total Assets"]
    fundamentals["Current Ratio"] = fundamentals["Total Assets"] / fundamentals["Current Liabilities"]
    fundamentals["FCF Margin"] = fundamentals["Free Cash Flow"] / fundamentals["Total Revenue"]

    fundamentals.index.name = "date"
    fundamentals.reset_index(inplace=True)
    fundamentals["tic"] = ticker

    return fundamentals


tickers = ["AAPL"]
all_fundamentals = pd.concat([get_fundamentals(tic) for tic in tickers])
all_fundamentals.to_csv("../../datasets/fundamentals/fundamentals_enhanced.csv", index=False)
