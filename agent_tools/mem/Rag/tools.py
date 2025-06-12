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