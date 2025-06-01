import yfinance as yf
import pandas as pd

import os

# Set the proxy URL and port
proxy_url = 'http://127.0.0.1'
proxy_port = '7890' # !!!please replace it with your own port

# Set the http_proxy and https_proxy environment variables
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'


def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)

    income_stmt = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T

    print("\n[Income Statement Columns]:", income_stmt.columns.tolist())
    print("\n[Balance Sheet Columns]:", balance_sheet.columns.tolist())
    print("\n[Cash Flow Columns]:", cash_flow.columns.tolist())

# get_fundamentals("AAPL")


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

    # 用于容错：字段缺失时返回 NaN 列
    def safe_get(df, key):
        return df[key] if key in df.columns else pd.Series([None] * len(df), index=df.index)

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

        "Operating Cash Flow": safe_get(cash_flow, "Operating Cash Flow"),
        "Free Cash Flow": safe_get(cash_flow, "Free Cash Flow"),
        "Capital Expenditure": safe_get(cash_flow, "Capital Expenditure"),
    })

    fundamentals.index.name = "date"
    fundamentals.reset_index(inplace=True)
    fundamentals["tic"] = ticker

    return fundamentals

tickers = ["AAPL"]
all_fundamentals = pd.concat([get_fundamentals(tic) for tic in tickers])
all_fundamentals.to_csv(f"../../datasets/fundamentals/{tickers[0]}_fundamentals.csv", index=False)
