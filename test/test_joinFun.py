# 初始 group_context 内容
group_context = "历史上下文："

# 模拟本轮的 (角色, 回复)
round_responses = [
    ("宏观分析师", "美国CPI高于预期，通胀压力依然存在"),
    ("资产配置顾问", "建议减少股票，增持债券与现金"),
    ("风险控制专家", "提高止损线，警惕市场剧烈波动")
]

# 构造 summaries 列表
summaries = [f"{role}：{resp}" for role, resp in round_responses]
print(summaries)

# 模拟 += 的过程
group_context += "\n" + "\n".join(summaries)

# 打印最终结果
print(group_context)
