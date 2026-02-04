import os

import sys
import json
import csv
import time

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from src.llm_client import chat
from src.utils.dataset_io import resolve_data_path, validate_openai_api_key

#MODEL = "gpt-3.5-turbo-1106"
MODEL = "deepseek-chat"  # 这是DeepSeek的主要对话模型

import re

def extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

def add_message(role, content, history):
    history.append({"role": role, "content": content})#往对话历史里追加消息

def ai_request(history, t=0.2):  # 发请求的函数：ai_request
    print(f"[DEBUG] 进入ai_request，历史长度: {len(history)}")
    return chat(history, model=MODEL, temperature=t)


def baseline(dataset, start_index=0, data_path=None):  # 主流程函数：baseline(dataset, start_index=0)
    validate_openai_api_key(mock_llm=False)
    print(f"[DEBUG] 进入baseline函数，数据集: {dataset}")  # baseline 函数是否进入
    print(f"Running baseline for dataset: {dataset}")  # 当前正在跑哪个数据集
    dataset_key = dataset.lower()
    # Create directory for logs if it doesn't exist
    #创建日志目录
    log_dir = f'log/{dataset}'#log/CLUTRR 这种目录，不存在就创建。
    os.makedirs(log_dir, exist_ok=True)#exist_ok=True：如果目录已存在也不报错。

    # Open the JSONL file 读取数据集文件（jsonl）
    try:
        data_path = resolve_data_path(dataset_key, data_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    with open(data_path, 'r', encoding='utf-8') as file:
        # Read all lines from the file
        lines = file.readlines() #每一行就是一个 JSON 对象。 把所有样本行读进内存（lines 列表）

        for i, line in tqdm(enumerate(lines[start_index:], start=start_index)): #循环处理每一条样本
            # Parse the JSON data
            data = json.loads(line)

            # Extract data from the JSON object
            #取出字段：id、question、answer
            id = data['id']
            question = data['question']
            answers = data['answer'].split("#### ")[-1]

            #先让“老师模型”读题并输出指导步骤，然后把这段指导保存起来
            #组装一个prompt（老师要做什么 + 题目）
            prompt0 = f"""You are a knowledgeable and patient professor whose role is to guide students in solving problems correctly.
Here is a question:
{question}
Note: Since your responsibility is to guide students in answering the question, your analysis should think step by step, 
Please note that your role is to guide them step by step through the problem, so please don't give them the final result
"""
            teacher_history = [] #老师的对话上下文（这里只有一轮）
            add_message('user', prompt0, teacher_history)#把 prompt 作为 user 消息塞进去
            teacher_output = ai_request(teacher_history)#调模型生成老师回复 teacher_output
            add_message('assistant', teacher_output, teacher_history)#把老师回复也塞回 history（形成完整对话记录）
            time.sleep(1)#暂停 1 秒防止请求过快

            #Student 第一次作答
            #给学生的作业要求+格式模板
            student1 = f"""To solve the problem, Please think and reason step by step, then answer.
Question:
{question}  
Generation Format:
Reasoning process:
Answer:
"""

            student_history = []#初始化学生的对话历史
            add_message('user', student1, student_history)#把学生 prompt 当成 user 消息加入 history
            student_output1 = ai_request(student_history)#向模型发送请求，得到学生第一次回答
            add_message('assistant', student_output1, student_history)#把学生的回答也存进历史
            time.sleep(1)#暂停 1 秒（节流）
            pred = extract_answer(student_output1).lower()
            gold = answers.strip().lower()
            student_correctness = 'True' if pred == gold else 'False'
            #student_correctness = 'True' if answers.lower() in student_output1.lower() else 'False'#判断学生第一次是否答对（字符串包含） 转小写


            #Student 第二次作答 （参考 teacher_output 做校验与修正）
            #构造一个新 prompt 并把 teacher_output 当作“可信分析”给它参考
            student2 = f"""Task:
The question contains a large set of semi-synthetic stories involving hypothetical families. 
The task is to infer the relationship between two family members, whose relationship is not explicitly mentioned in the given story.
This is an credible analysis of this question:
{teacher_output}
Please verify your reasoning process for errors based on this analysis,
then refine your reasoning process and answer.
For question: How is [A] related to [B], your answer should be [A] is [B]'s [relationship].
Generation Format:
inference process:
Answer: 
"""
            add_message('user', student2, student_history)#把这个 prompt 追加到 同一个 student_history（所以模型能看到自己第一轮回答过什么）。
            student_output2 = ai_request(student_history)#再请求一次模型输出 student_output2（希望它修正之前的推理并给更正确的答案）。
            add_message('assistant', student_output2, student_history)
            #guide_correctness = 'True' if any(answer.lower() in student_output2.lower() for answer in answers) else 'False'
            pred = extract_answer(student_output2).lower()
            gold = answers.strip().lower()
            guide_correctness = 'True' if pred == gold else 'False'
            #guide_correctness = 'True' if answers.lower() in student_output2.lower() else 'False'
            time.sleep(1)

            # Save log
            #每做完一道题，就把这道题的输入、标准答案、模型输出过程都保存成一个 JSON 文件，方便你后面统计、复盘、找错误样例
            log_filename = f'{dataset}_{i}.json' #每个样本一个日志文件，文件名能看出属于哪个数据集、是第几个样本
            log_path = os.path.join(log_dir, log_filename) #拼出完整路径
            with open(log_path, 'w') as log_file: #以写入模式打开文件
                json.dump({     #把一个 Python 字典写成 JSON
                    "id": id,
                    "question": question,
                    'answer': answers,
                    "student_correctness": student_correctness, #学生第一次答题是否正确（通常是 'True'/'False' 或 bool，看你怎么定义）。
                    "guide_correctness": guide_correctness,#学生第二次参考 teacher_output 后是否正确。
                    "teacher_log": teacher_history,
                    "student_log": student_history,
                }, log_file, indent=4) #indent=4 会把 JSON 格式化成更易读的样子（缩进 4 个空格）


if __name__ == "__main__":  #入口判断
    parser = ArgumentParser(description="Baseline script with dataset and start index arguments")#创建命令行参数解析器
    parser.add_argument("--dataset", help="Dataset name") #--dataset：决定跑哪个数据集目录
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")#--start_index：支持从某个样本编号开始，方便续跑。
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()

    baseline(args.dataset, args.start_index, data_path=args.data_path)


baseline(args.dataset, args.start_index, data_path=args.data_path)