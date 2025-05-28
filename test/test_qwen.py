from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 替换成本地模型路径
model_path = "../models/Qwen-4B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# 编写提示语
prompt = "请你作为一个金融分析师，分析当前市场趋势。"

# 构造输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成输出
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256)

# 打印结果
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🤖 模型回答：\n")
print(response)

with open('./text/test_qwen.txt', 'w', encoding='utf-8') as f:
    f.write(response)