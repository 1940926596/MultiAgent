news_sentiment_summary_function = {
    "name": "news_sentiment_summary",
    "description": "Summarize daily news articles and their sentiment scores.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise summary of the key news points and sentiment trends for the day."
            },
            "overall_sentiment": {
                "type": "string",
                "description": "Overall sentiment trend, e.g., Positive, Negative, Neutral."
            },
            "key_points": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of key insights or important points extracted from the news."
            }
        },
        "required": ["summary", "overall_sentiment", "key_points"]
    }
}
