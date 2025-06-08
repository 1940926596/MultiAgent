macro_summary_function = {
    "name": "macro_summary",
    "description": "Summarize the operational status, potential risks, and provide a macro score for this quarter based on the financial report.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the quarterly financial report."
            },
            "risk_tag": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags from a macroeconomic perspective, such as 'High R&D expenditure', 'Slow revenue growth', 'Healthy financial condition', etc."
            },
            "macro_score": {
                "type": "number",
                "description": "An overall health score of the company's operations for this quarter, ranging from 0 to 1.",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["summary", "risk_tag", "macro_score"]
    }
}
