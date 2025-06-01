from agent_roles import (
    TechnicalAnalystAgent, 
    FundamentalAnalystAgent,
    SentimentAnalystAgent,
    RiskControlAgent,
    CIOAgent
)

# 1) 创建各个分析师实例
tech_agent = TechnicalAnalystAgent(name="技术分析师", role="Technical")
fund_agent = FundamentalAnalystAgent(name="基本面分析师", role="Fundamental")
sent_agent = SentimentAnalystAgent(name="情绪分析师", role="Sentiment")
risk_agent = RiskControlAgent(name="风控分析师", role="Risk")

# 2) CIO 总决策，集合所有顾问
cio_agent = CIOAgent(name="首席投资官", role="CIO", advisors=[tech_agent, fund_agent, sent_agent, risk_agent])

# 3) 模拟输入数据（你可以从你的数据集中抽取一条）
sample_data = {
    "close": 150.0,
    "macd": 0.5,
    "rsi_30": 65,
    "all_fields": '{"revenue": "100B", "roe": "15%"}',
    "sentiment": 0.3,
    "news_text": "Company reports strong earnings growth.",
    "vix": 20,
    "turbulence": 100
}

# 4) 调用各个分析师分析
print("技术分析师决策:", tech_agent.analyze(sample_data))
print("基本面分析师决策:", fund_agent.analyze(sample_data))
print("情绪分析师决策:", sent_agent.analyze(sample_data))
print("风控分析师决策:", risk_agent.analyze(sample_data))

# 5) CIO 汇总决策
final_decision = cio_agent.analyze(sample_data)
print("\nCIO 汇总决策:")
print(final_decision)
