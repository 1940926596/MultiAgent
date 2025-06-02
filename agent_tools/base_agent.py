# base_agent.py
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseFinanceAgent:
    def __init__(self, name: str, role: str, model=None, tokenizer=None):
        self.name = name
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = f"你现在是一个金融分析师，角色是：{role}"
        self.history = [{"role": "system", "content": self.system_prompt}]

    def llm_chat(self, prompt, reset_history=False):
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，无法使用 llm_chat")

        if reset_history:
            self.history = [{"role": "system", "content": self.system_prompt}]

        self.history.append({"role": "user", "content": prompt})

        input_text = self._format_prompt()
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        match = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", response, re.DOTALL)
        answer = match[-1].strip() if match else response
        self.history.append({"role": "assistant", "content": answer})
        print(self.history.__str__())
        return answer

    def _format_prompt(self):
        prompt = ""
        for msg in self.history:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
