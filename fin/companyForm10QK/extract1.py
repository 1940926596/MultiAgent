import os
import re
from bs4 import BeautifulSoup
import csv

def process_all_md_files(md_root):
    print(md_root)
    for root, _, files in os.walk(md_root):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(root)
                print(file_path)

                output_path = f"{root}/AAPL_8K_Fields.html"
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 用正则找到所有 <DOCUMENT>...</DOCUMENT> 区段
                documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL)

                # 寻找 TYPE 是 8-K 的文档
                for doc in documents:
                    if "<TYPE>8-K" in doc:
                        # 提取该段并保存为 HTML 文件
                        match = re.search(r"<FILENAME>(.*?)\n", doc)
                        filename = match.group(1).strip() if match else output_path
                        with open(output_path, "w", encoding="utf-8") as out:
                            out.write(doc)
                        print(f"提取成功！保存为：{output_path}")
                        break
                else:
                    print("没有找到 TYPE 为 8-K 的文档。")    

                # Step 1: 读取提取好的 HTML 文件
                html_file = output_path
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
                save_path = root

                csv_file = "AAPL_8K_Fields.csv"
                with open(f"{save_path}/{csv_file}", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["name", "contextRef", "unitRef", "value"])
                    writer.writerows(data)

                print(f"提取完成！财务字段已保存为：{csv_file}")                                

    


# 修改成你的实际路径
md_root_dir = '../sec-edgar-filings/AAPL/8-K'
# md_root_dir = '../sec-edgar-filings/AAPL/10-Q'
print(os.path.exists(md_root_dir))  # 应该输出 True

process_all_md_files(md_root_dir)