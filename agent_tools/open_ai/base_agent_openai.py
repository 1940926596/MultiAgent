# import os,sys
# import requests

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from openai import OpenAI
# import json
# from agent_tools.open_ai import my_config

# class BaseFinanceAgent:
#     def __init__(self, name: str, role: str, model="gpt-3.5-turbo", function_schema=None, memory_size=5):
#         self.name = name
#         self.role = role
#         self.model = model
#         self.function_schema = function_schema or []
#         self.client = OpenAI(api_key=my_config.api_key)
#         self.memory_size = memory_size
#         self.history = []
#         self.interested_fields = []

#     def update_history(self, data: dict):
#         self.history.append(data)
#         if len(self.history) > self.memory_size:
#             self.history.pop(0)

#     def build_history_prompt(self):
#         if not self.history and not self.interested_fields:
#             return ""

#         prompt = "Relevant historical data:\n"

#         if not self.history:
#             return prompt + "No historical data."
        

#         for h in self.history:
#             line = f"- Date {h.get('date', '')}"
#             for field in self.interested_fields:
#                 val = h.get(field, "NA")
#                 line += f" | {field}: {val}"
#             prompt += line + "\n"
#         return prompt

#     def ask_model(self, current_prompt: str) -> dict:
#         full_prompt = self.build_history_prompt() + "\nCurrent input:\n" + current_prompt

#         messages = [
#             {"role": "system", "content": f"You are a financial analyst. Your role is: {self.role}"},
#             {"role": "user", "content": full_prompt}
#         ]

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             functions=self.function_schema,
#             function_call={"name": self.function_schema[0]["name"]}
#         )

#         try:
#             args = response.choices[0].message.function_call.arguments
#             parsed = json.loads(args)
#             return parsed
        
#         except Exception as e:
#             print(f"[Model function call parsing failed] {e}")

#         return {
#             "action": "hold",
#             "confidence": 0.5,
#             "reasoning": "Model did not return structured output"
#         }
import os
import sys
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from openai import OpenAI
from agent_tools.open_ai import my_config  # 需要包含 api_key 字符串变量


class BaseFinanceAgent:
    def __init__(self, name: str, role: str, model="deepseek-chat", function_schema=None, memory_size=5):
        self.name = name
        self.role = role
        self.model = model
        self.function_schema = function_schema or []
        self.memory_size = memory_size
        self.history = []
        self.interested_fields = []

        self.client = OpenAI(
            api_key=my_config.api_key,
            base_url="https://api.deepseek.com"
        )

    def update_history(self, data: dict):
        self.history.append(data)
        if len(self.history) > self.memory_size:
            self.history.pop(0)

    def build_history_prompt(self):
        if not self.history and not self.interested_fields:
            return ""

        prompt = "Relevant historical data:\n"
        for h in self.history:
            line = f"- Date {h.get('date', '')}"
            for field in self.interested_fields:
                val = h.get(field, "NA")
                line += f" | {field}: {val}"
            prompt += line + "\n"
        return prompt

    def ask_model(self, current_prompt: str) -> dict:
        full_prompt = self.build_history_prompt() + "\nCurrent input:\n" + current_prompt

        messages = [
            {"role": "system", "content": f"You are a financial analyst. Your role is: {self.role}"},
            {"role": "user", "content": full_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": self.function_schema[0]
                }] if self.function_schema else [],
                tool_choice={"type": "function", "function": {"name": self.function_schema[0]["name"]}} if self.function_schema else "auto"
            )

            message = response.choices[0].message

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                args = tool_call.function.arguments
                parsed = json.loads(args)
                return parsed
            else:
                print("[⚠️ tool_calls 为空，未触发函数调用]")
                return {
                    "action": "hold",
                    "confidence": 0.5,
                    "reasoning": "Model did not return tool call"
                }

        except Exception as e:
            print(f"[❌ DeepSeek 调用或解析失败] {e}")
            return {
                "action": "hold",
                "confidence": 0.5,
                "reasoning": "Model exception occurred"
            }

# # 示例 Function Schema，必须符合 DeepSeek 要求
# function_schema = [{
#     "name": "stock_decision",
#     "description": "Decide whether to buy, sell, or hold a stock based on technical indicators",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "action": {
#                 "type": "string",
#                 "enum": ["buy", "sell", "hold"],
#                 "description": "Investment action"
#             },
#             "confidence": {
#                 "type": "number",
#                 "description": "Confidence in the action (0.0 - 1.0)"
#             },
#             "reasoning": {
#                 "type": "string",
#                 "description": "Explanation for the decision"
#             }
#         },
#         "required": ["action", "confidence", "reasoning"]
#     }
# }]


# # 创建一个示例 Agent 实例
# agent = BaseFinanceAgent(
#     name="TestAgent",
#     role="Technical Analyst focusing on trend and momentum indicators",
#     function_schema=function_schema
# )

# # 提供一条测试数据
# data = {
#     "date": "2025-06-20",
#     "close": 189.3,
#     "macd": 1.2,
#     "rsi_30": 70,
#     "cci_30": 120,
#     "dx_30": 20,
#     "close_30_sma": 180,
#     "close_60_sma": 170
# }
# agent.interested_fields = ["close", "macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

# # 构造 prompt 并调用
# prompt = (
#     f"Below are the market technical indicators for a stock on {data['date']} \n\n"
#     f"- Current Closing Price: {data['close']}\n"
#     f"- MACD: {data['macd']}\n"
#     f"- RSI (30-day): {data['rsi_30']}\n"
#     f"- CCI (30-day): {data['cci_30']}\n"
#     f"- DMI (30-day): {data['dx_30']}\n"
#     f"- 30-day SMA: {data['close_30_sma']}\n"
#     f"- 60-day SMA: {data['close_60_sma']}\n\n"
#     "Based on this data, respond by calling 'stock_decision' function with action, confidence, and reasoning."
# )

# result = agent.ask_model(prompt)
# print("[结果解析]:", result)