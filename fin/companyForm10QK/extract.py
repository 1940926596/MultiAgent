import os
import re

# 设置路径
input_path = "/data/postgraduates/2024/chenjiarui/Fin/MultiAgents/fin/companyForm10Q/sec-edgar-filings/AAPL/10-Q/0000320193-25-000008/full-submission.txt"
output_path = "AAPL-10Q-2025Q1.html"  # 输出路径可改



with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# 用正则找到所有 <DOCUMENT>...</DOCUMENT> 区段
documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)

# 寻找 TYPE 是 10-Q 的文档
for doc in documents:
    if "<TYPE>10-Q" in doc:
        # 提取该段并保存为 HTML 文件
        match = re.search(r"<FILENAME>(.*?)\n", doc)
        filename = match.group(1).strip() if match else output_path
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(doc)
        print(f"提取成功！保存为：{output_path}")
        break
else:
    print("没有找到 TYPE 为 10-Q 的文档。")


# from bs4 import BeautifulSoup

# with open("AAPL-10Q-2025Q1.html", "r", encoding="utf-8") as f:
#     soup = BeautifulSoup(f, "lxml")

# # 示例：获取所有财务字段
# for tag in soup.find_all("ix:nonfraction"):
#     print(tag.get("name"), ":", tag.text.strip())



from bs4 import BeautifulSoup
import csv

# Step 1: 读取提取好的 HTML 文件
html_file = "AAPL-10Q-2025Q1.html"
with open(html_file, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "lxml")

# Step 2: 提取所有 <ix:nonFraction> 和 <ix:nonNumeric> 节点
elements = soup.find_all(["ix:nonfraction", "ix:nonnumeric"])

# Step 3: 结构化数据
data = []
for tag in elements:
    name = tag.get("name")
    context = tag.get("contextref")
    unit = tag.get("unitref")  # 可能为 None
    value = tag.text.strip().replace(",", "")
    data.append([name, context, unit, value])

# Step 4: 写入 CSV 文件
save_path="../../datasets/companyForm10Q"

csv_file = "AAPL_10Q_Fields.csv"
with open(f"{save_path}/{csv_file}", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "contextRef", "unitRef", "value"])
    writer.writerows(data)

print(f"提取完成！财务字段已保存为：{csv_file}")
