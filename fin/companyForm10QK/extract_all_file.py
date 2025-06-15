import os
import re
from bs4 import BeautifulSoup
import csv

def process_full_submission_txt(txt_path):
    root = os.path.dirname(txt_path)

    # HTML 和 CSV 输出路径
    output_html_path = os.path.join(root, "Fields.html")
    output_csv_path = os.path.join(root, "Fields.csv")

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 提取所有 <DOCUMENT>...</DOCUMENT>
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)

    found = False
    for doc in documents:
        if any(tag in doc for tag in ["<TYPE>8-K", "<TYPE>10-K", "<TYPE>10-Q", "<TYPE>20-F", "<TYPE>6-K"]):
            with open(output_html_path, "w", encoding="utf-8") as out:
                out.write(doc)
            print(f"✅ 提取 HTML 成功：{output_html_path}")
            found = True
            break

    if not found:
        print(f"❌ 未找到有效的 <TYPE> 标签：{txt_path}")
        return

    # 解析 HTML 提取数据
    with open(output_html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    elements = soup.find_all(["ix:nonfraction", "ix:nonnumeric"])
    data = []
    for tag in elements:
        name = tag.get("name")
        context = tag.get("contextref")
        unit = tag.get("unitref")
        value = tag.text.strip().replace(",", "")
        data.append([name, context, unit, value])

    # 写入 CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "contextRef", "unitRef", "value"])
        writer.writerows(data)

    print(f"✅ 提取完成，CSV 文件保存于：{output_csv_path}")


def process_all_sec_filings(root_dir):
    print(f"📁 扫描目录：{root_dir}")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "full-submission.txt":
                txt_path = os.path.join(root, file)
                try:
                    process_full_submission_txt(txt_path)
                except Exception as e:
                    print(f"⚠️ 错误处理文件 {txt_path}：{e}")


# 设置主目录路径
root_sec_path = "./sec-edgar-filings1"
assert os.path.exists(root_sec_path), f"路径不存在：{root_sec_path}"

# 执行提取
process_all_sec_filings(root_sec_path)
