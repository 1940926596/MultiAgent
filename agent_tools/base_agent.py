# base_agent.py

from abc import ABC, abstractmethod

class BaseFinanceAgent(ABC):
    def __init__(self, name: str, role: str, model=None):
        self.name = name
        self.role = role
        self.model = model  # 可以是 NLP 模型、预测器、规则系统等

    @abstractmethod
    def analyze(self, data: dict) -> dict:
        """
        子类必须实现的分析方法
        输入：data 是一个 dict，包含结构化/非结构化信息
        输出：一个 dict，如：
        {
            "action": "buy" / "hold" / "sell",
            "confidence": 0.85,
            "reasoning": "基于MACD和RSI趋势，短期上涨概率高"
        }
        """
        pass
