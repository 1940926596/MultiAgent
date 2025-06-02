# 测试运行
from agent_roles import agent_list

test_data = {
    "macd": 0.5,
    "rsi_30": 60,
    "close": 190,
    "all_fields": "净利润同比增长 20%，ROE 提升至 18%，营收稳定增长。",
    "sentiment": 0.3,
    "news_text": "苹果发布新产品，市场反响热烈。",
    "vix": 18,
    "turbulence": 90
}

for agent in agent_list:
    print(f"\n🚀 {agent.name} 正在分析...")
    result = agent.analyze(test_data)
    print(f"📈 动作: {result['action']} | 理由:\n{result['reasoning']}")
