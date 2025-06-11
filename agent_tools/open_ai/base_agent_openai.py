import os,sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from openai import OpenAI
import json
from agent_tools.open_ai import my_config

class BaseFinanceAgent:
    def __init__(self, name: str, role: str, model="gpt-3.5-turbo", function_schema=None, memory_size=5):
        self.name = name
        self.role = role
        self.model = model
        self.function_schema = function_schema or []
        self.client = OpenAI(api_key=my_config.api_key)
        self.memory_size = memory_size
        self.history = []
        self.interested_fields = []

    def update_history(self, data: dict):
        self.history.append(data)
        if len(self.history) > self.memory_size:
            self.history.pop(0)

    def build_history_prompt(self):
        if not self.history and not self.interested_fields:
            return ""

        prompt = "Relevant historical data:\n"

        if not self.history:
            return prompt + "No historical data."
        

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=self.function_schema,
            function_call={"name": self.function_schema[0]["name"]}
        )

        try:
            args = response.choices[0].message.function_call.arguments
            parsed = json.loads(args)
            return parsed
        
        except Exception as e:
            print(f"[Model function call parsing failed] {e}")

        return {
            "action": "hold",
            "confidence": 0.5,
            "reasoning": "Model did not return structured output"
        }
