# import csv
# import numpy as np
# from argparse import ArgumentParser #ArgumentParser：解析命令行参数，比如 --dataset CLUTRR
#
# parser = ArgumentParser() #创建命令行参数解析器
# parser.add_argument('--dataset', required=True) #声明必须提供 --dataset 参数，不提供会报错并提示用法。
# args = parser.parse_args()
#
# dataset = args.dataset #把 dataset 参数取出来保存到变量。
#
# #学生第一次回答正确率
# #学生第二次（guided）回答正确率
# correctness_columns = ['student_correctness', 'guide_correctness']#指定要评测的列名
#
# def evaluate(file_name):
#     with open(file_name, 'r', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile) #DictReader 会把每一行读成一个字典（dict）
#         data = list(reader)
#
#     for column in correctness_columns:#依次计算 student/guide 两列
#         column_accuracy = calculate_accuracy(data, column)#调用下面的函数得到该列的准确率（百分比）
#         print(f"{file_name} {column} Accuracy: {column_accuracy:.2f}%")#保留两位小数打印
#
# def calculate_accuracy(data, column):#真正计算 Accuracy
#     correctness_values = [row[column] for row in data] #取出这一列所有的值
#     accuracy = 100 * np.mean([1 if correctness == 'True' else 0 for correctness in correctness_values]) #1.把 True/False 转成 1/0  2.求平均再乘 100 np.mean([1,0,1,...]) 得到一个 0~1 的小数
#     return accuracy
#
#
# #评测文件名
# file_name1 = (f'log/date_new.csv')
#
# evaluate(file_name1)
#
#
# # python evaluate.py --dataset CLUTRR


import os, json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

log_dir = os.path.join("log", args.dataset)
if not os.path.isdir(log_dir):
    raise FileNotFoundError(f"Log dir not found: {log_dir}")

files = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
if not files:
    raise FileNotFoundError(f"No .json logs found in {log_dir}")

def to_bool(x):
    # 兼容 True/False 或 'True'/'False'
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() == "true"

student_ok = 0
guide_ok = 0
n = 0

for fn in files:
    path = os.path.join(log_dir, fn)
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"[WARN] skip {fn}: {e}")
        continue

    student_ok += 1 if to_bool(d.get("student_correctness", False)) else 0
    guide_ok += 1 if to_bool(d.get("guide_correctness", False)) else 0
    n += 1

if n == 0:
    raise RuntimeError("No valid json logs to evaluate.")

print(f"Dataset: {args.dataset}")
print(f"Student Accuracy: {student_ok/n*100:.2f}% ({student_ok}/{n})")
print(f"Guide   Accuracy: {guide_ok/n*100:.2f}% ({guide_ok}/{n})")


#python scripts/evaluate.py --dataset CLUTRR
