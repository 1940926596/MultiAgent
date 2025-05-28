# 构造Qwen的输入格式（ChatML 或者 HF格式，取决于模型）
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from datetime import datetime

# 获取当前时间，并格式化为字符串
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 构造带时间戳的文件路径
log_path = f'../log/log_{timestamp}.txt'


class FinanceAgent:
    def __init__(self, role, system_prompt, model, tokenizer):
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.tokenizer = tokenizer
        self.history = []  # 存储上下文历史

        # 初始化时向模型输入角色设定
        self._init_conversation()

    def _init_conversation(self):
        self.history.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_message, reset_history=False):
        if reset_history:
            self.history = [{"role": "system", "content": self.system_prompt}]

        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": h["role"], "content": h["content"]} for h in self.history]
        # print(messages)

        input_text = ""
        for msg in messages:
            input_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        input_text += "<|im_start|>assistant\n"

        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(  
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(response)

        import re
        match = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", response, re.DOTALL)
        if match:
            answer = match[-1].strip()
        else:
            answer = response  # fallback
        

        self.history.append({"role": "assistant", "content": answer})

        with open(log_path, 'w', encoding='utf-8') as f:
            for item in self.history:
                f.write(f"{item['role'].upper()}: {item['content']}\n")

        return answer



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
   
    model_path = "../models/Qwen-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    macro_analyst = FinanceAgent(
        role="宏观分析师",
        system_prompt="你是一位宏观经济分析师，擅长分析经济趋势、货币政策、地缘政治事件对市场的影响。",
        model=model,
        tokenizer=tokenizer,

    )

    response = macro_analyst.chat("当前全球经济放缓会对中国市场有什么影响？请你精炼作答")
    # print(response)


"""
def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": h["role"], "content": h["content"]} for h in self.history]

        input_text = ""
        for msg in messages:
            input_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        input_text += "<|im_start|>assistant\n"

        # print("[DEBUG 输入] \n", input_text)

        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(  
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id= eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 处理 assistant 的回答并去掉 endoftext
        import re

        match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # print("match")
        else:
            answer = response  # fallback


        self.history.append({"role": "assistant", "content": answer})
        
        with open(log_path, 'w', encoding='utf-8') as f:
            for item in self.history:
                f.write(f"{item['role'].upper()}: {item['content']}\n")

        return answer
"""
    