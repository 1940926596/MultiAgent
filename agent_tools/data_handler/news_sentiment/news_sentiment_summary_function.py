news_sentiment_summary_function = {
    "name": "news_sentiment_summary",
    "description": "Analyze daily financial news and summarize key points and sentiment trends.",
    "parameters": {
        "type": "object",
        "properties": {
            "news_summary": {
                "type": "string",
                "description": "A concise summary of the main financial news and sentiment signals for the day."
            },
            "overall_sentiment": {
                "type": "string",
                "description": "The overall market sentiment inferred from the news. Must be one of: Positive, Negative, or Neutral."
            },
            "key_points": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "A list of important insights, such as risks, opportunities, or notable events mentioned in the news."
            }
        },
        "required": ["news_summary", "overall_sentiment", "key_points"]
    }
}
