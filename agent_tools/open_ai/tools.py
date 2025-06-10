# function_schema.py

function_schema = [
    {
        "name": "stock_decision",
        "description": "Make an operation decision for a given stock",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell", "hold"],
                    "description": "Operation recommendation"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level (0 to 1) in the recommended action (buy/sell/hold)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for the operation"
                }
            },
            "required": ["action", "confidence", "reasoning"]
        }
    }
]

function_schema1 = [{
    "name": "fundamental_analysis_report",
    "description": "Generates a comprehensive fundamental analysis report of a company based on financial data and external research sources.",
    "parameters": {
        "type": "object",
        "properties": {
            "financial_summary": {
                "type": "string",
                "description": "Summarizes key points from the financial data: revenue, profitability, cash flow, and debt situation."
            },
            "industry_context": {
                "type": "string",
                "description": "Summarizes recent industry trends, challenges, and opportunities from online sources such as industry reports or Wikipedia."
            },
            "company_profile": {
                "type": "string",
                "description": "Provides a brief overview of the company's core business, competitive advantages, and market position."
            },
            "risk_assessment": {
                "type": "string",
                "description": "Highlights potential long-term risks based on financial metrics and external context (e.g., debt levels, industry headwinds)."
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "A confidence score for the completeness and reliability of the analysis."
            }
        },
        "required": ["financial_summary", "industry_context", "company_profile", "risk_assessment"]
    }
}]