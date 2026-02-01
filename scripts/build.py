import os
import json
import csv
from argparse import ArgumentParser

# Initialize ArgumentParser
#让你运行脚本时可以通过 --dataset 指定数据集名字。
parser = ArgumentParser(description='Process dataset name.')
#parser.add_argument('--dataset', metavar='dataset', type=str, help='Name of the dataset')
parser.add_argument('--dataset', required=True, type=str, help='Name of the dataset')
args = parser.parse_args()
dataset = args.dataset

# Define CSV file name and field names
csv_filename = f'log/{dataset}_extracted.csv'  # 修改保存路径 输出文件路径：例如 log/CLUTRR_extracted.csv
fieldnames = ['id', 'student_correctness', 'guide_correctness', 'question', 'answer', 'teacher_log_content', 'student_log_content_1', 'student_log_content_3']
#fieldnames：CSV 表头有哪些列


# Count the number of files in the directory
#os.listdir("log/CLUTRR") 会返回这个目录下所有文件/子目录名字的列表
#len(...) 得到数量
#file_count = len(os.listdir(f'log/{dataset}'))
json_files = sorted([f for f in os.listdir(f'log/{dataset}') if f.endswith('.json')])
file_count = len(json_files)



# Open the CSV file for writing
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile: #打开（创建）CSV 文件
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames) #创建 CSV 字典写入器
    writer.writeheader() #把 fieldnames 写成 CSV 的第一行列名

    # Define the range of file names to process
    for i in range(file_count):
        file_path = f'log/{dataset}/{dataset}_{i}.json'  # Adjust the file path pattern as needed
        if os.path.exists(file_path): #用来判断文件是否存在
            try:
                with open(file_path, 'r', encoding='utf-8') as jsonfile:
                    data = json.load(jsonfile) #把 json 文件读成 Python 字典 data

                    # Extract the required fields
                    #从 data 里提取你要的字段
                    extracted_data = {
                        'id': data['id'],
                        'student_correctness': data['student_correctness'],
                        'guide_correctness': data['guide_correctness'],
                        'question': data['question'],
                        'answer': data['answer'],
                        'teacher_log_content': data['teacher_log'][1]['content'] ,# Extract the content field from teacher_log
                        'student_log_content_1': data['student_log'][1]['content'],  # Extract the first content field from student_log
                        'student_log_content_3': data['student_log'][3]['content']   # Extract the third content field from student_log
                    }

                    # Write to the CSV file
                    writer.writerow(extracted_data)
            except Exception as e: #异常处理和结束提示
                print("Error processing file:", file_path)
                print("Error message:", str(e))

print("Extraction complete and saved to", csv_filename)


# python build.py --dataset CLUTRR