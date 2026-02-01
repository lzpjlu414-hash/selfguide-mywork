import json
import csv
import time
import openai
import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser
import re
import ast


def judge_correctness(dataset_key: str, gold: str, pred: str) -> str:
    """
    返回 'True' / 'False'（保持你原代码风格）
    dataset_key: 已经 lower() 的 dataset 名称
    """
    gold_s = str(gold).strip().lower()
    pred_s = str(pred).strip().lower()

    if dataset_key == "mmlu":
        # gold 通常是 A/B/C/D
        # 从模型输出里抓一个最像选项的字母（优先 A-D）
        m = re.search(r"\b([a-d])\b", pred_s)
        pred_choice = m.group(1) if m else (pred_s[:1] if pred_s else "")
        return "True" if pred_choice == gold_s[:1] else "False"

    if dataset_key == "sqa":
        # 只取第一个词判断 yes/no
        first = pred_s.split()[0] if pred_s.split() else ""
        return "True" if first in ("yes", "no") and first == gold_s else "False"

    # date / CLUTRR：沿用你原来的“子串包含”规则
    return "True" if gold_s in pred_s else "False"


# 从 log_guideline/{dataset}.jsonl 逐行读样本（每行一个 JSON：id/question/answer/guideline/...）。
#
# 第 1 轮：把问题发给模型，让模型生成一个回答（output0）。
#
# 第 2 轮：再把 问题 + “另一个 agent 的解答（其实就是 output0）” 发给模型，让它基于这个“建议”给出更新后的最终答案（output）。
#
# 用非常简单的字符串匹配评估对错（answers 是否出现在 output 中）。
#
# 把完整对话 history 和对错写入 log/{method}/{dataset}/{dataset}_{i}.json

openai.api_key = ""
openai.api_base = ""

MODEL = "gpt-3.5-turbo-1106"


def add_message(role, content, history):
    history.append({"role": role, "content": content})
    # 把一条消息追加到 history 列表里
    # role 通常是 'user' / 'assistant' / 'system'
    # history 的结构是 OpenAI ChatCompletion 的 messages 格式
    #content 就是一段字符串，通常就是你准备发给模型的 prompt，或者模型返回的回答文本。


#把 history（一组对话 messages）发给模型 → 拿回模型的回复文本 → 返回这段文本
def ai_request(history, t=0.2):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=history,
        temperature=t,
    )
    output = response["choices"][0]["message"]["content"] # # ChatCompletion 返回 choices 列表；取第一个候选的 message.content 作为输出文本
    return output


def baseline(dataset, method, start_index=0):
    # 打印你正在跑哪个数据集
    print(f"Running baseline for dataset: {dataset}")
    # Create directory for logs if it doesn't exist
    # 输出日志目录：例如 log/sd_debate/date 或 log/cot_debate/date
    log_dir = f'log/{method}/{dataset}'
    # 如果目录不存在就创建；exist_ok=True 表示存在也不报错
    os.makedirs(log_dir, exist_ok=True)

    # Open the JSONL file
    with open(f"log_guideline/{dataset}.jsonl", 'r', encoding='utf-8') as file:
        # 打开输入数据文件：每行一个 JSON（jsonl 格式）
        # Read all lines from the file
        # 一次性把所有行读入内存（数据大时可能占内存；但写起来简单）
        lines = file.readlines()

        for i, line in tqdm(enumerate(lines[start_index:], start=start_index)):
            # 从 start_index 开始处理
            # enumerate(lines[start_index:], start=start_index) 保证 i 还是原始行号
            # tqdm 包装后显示进度条

            # Parse the JSON data
            data = json.loads(line)

            # Extract data from the JSON object
            id = data['id']
            question = data['question']
            answers = data['answer']
            guideline = data['guideline']

            # 这几段是给不同数据集准备的prompt模板，因为不同任务的输入格式 / 输出格式不一样
            # mmlu
            # choice = data['choices']
            # options = '\n'.join([f"{key}: {value}" for key, value in choice.items()])
            # mcq = f"Question: {question}\n{options}"
            # prompt0 = mcq + '\n' + "Choice: "

            # CLUTRR
            # prompt0 = question + '\n' + "Answer: "

            # # sqa
            # prompt0 = "Question: " + question + '\n' + "Your answer should be Yes or No. \nAnswer: "

            #date 数据集：简单直接问答提示
            prompt_sd = "Question: " + question + '\n' + "Answer: "
            # 例子：
            # Question: ...
            # Answer:

            prompt_cot = f"""To solve the problem, Please think and reason step by step, then answer.
            Question: {question}  
            """

            if method == "sd_debate":
                prompt0 = prompt_sd
            elif method == "cot_debate":
                prompt0 = prompt_cot
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

            history0 = []
            add_message('user', prompt0, history0)
            output0 = ai_request(history0)
            add_message('assistant', output0, history0)
            time.sleep(1)

            # date
            prompt1 = f"""Question: {question}
These are the solution to the problem from another agent:
{output0}
Using the reasoning from another agent as additional advice, can you give an updated answer? 
Examine the solution to the problem from another agent. 
Put your final answer in the form of MM/DD/YYYY.
"""

# #             # CLUTRR
# #             prompt1 = f"""{question}
# # These are the solution to the problem from another agent:
# # {output0}
# # Using the reasoning from another agent as additional advice, can you give an updated answer?
# # Examine the solution to the problem from another agent.
# # Please give your final answer.
# # """
# #
# #             # sqa
# #             prompt1 = f"""Question: {question}
# # These are the solution to the problem from another agent:
# # {output0}
# # Using the reasoning from another agent as additional advice, can you give an updated answer?
# # Examine the solution to the problem from another agent.
# # Put your final answer in the form of Yes or No.
# # """
# #             # mmlu
# #             prompt1 = f"""{mcq}
# # These are the solution to the problem from another agent:
# # {output0}
# # Using the reasoning from another agent as additional advice, can you give an updated answer?
# # Examine the solution to the problem from another agent.
# # Please give your final answer.
# # Choice:
# # """
#
#
#             history1 = []
#             add_message('user', prompt1, history1)
#             output = ai_request(history1)
#             add_message('assistant', output, history1)
#             time.sleep(1)
#
#             # mmlu
#             # ground_truth = answers.lower()
#             # theoutput = str(output).split()
#             # text_str = str(theoutput[0]).lower()
#             # print(ground_truth)
#             # print(text_str)
#             #
#             # correctness = 'True' if ground_truth in text_str else 'False'
#
#             # sqa
#             # first_words = ' '.join(output.split()[:1])
#             # corrected_result = first_words.lower()  # 小写化结果
#             # if 'yes' in corrected_result:
#             #     correctness = True
#             # elif 'no' in corrected_result:
#             #     correctness = False
#             # else:
#             #     correctness = None  # 如果既不是'yes'也不是'no'，则将 correctness 设为 None
#
#             # date/CLUTRR
#             correctness = 'True' if answers.lower() in output.lower() else 'False'

            dataset_key = dataset.lower()
            method_key = method.lower()

            # ---------- 构造第一轮 prompt0（按 dataset + method 自动切换） ----------
            if dataset_key == "mmlu":
                choice = data["choices"]  # dict: {"A": "...", "B": "...", ...}
                options = "\n".join([f"{k}: {v}" for k, v in choice.items()])
                stem = f"Question: {question}\n{options}\nChoice: "

                if method_key == "cot_debate":
                    prompt0 = "Please think step by step, then answer with a single letter (A/B/C/D).\n" + stem
                elif method_key == "sd_debate":
                    prompt0 = stem
                else:
                    raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

            elif dataset_key == "clutrr":
                stem = question + "\nAnswer: "
                if method_key == "cot_debate":
                    prompt0 = "Please think step by step, then answer.\n" + stem
                elif method_key == "sd_debate":
                    prompt0 = stem
                else:
                    raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

            elif dataset_key == "sqa":
                stem = f"Question: {question}\nYour answer should be Yes or No.\nAnswer: "
                if method_key == "cot_debate":
                    prompt0 = f"Please think step by step, then answer Yes or No.\nQuestion: {question}\nAnswer: "
                elif method_key == "sd_debate":
                    prompt0 = stem
                else:
                    raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

            elif dataset_key == "date":
                stem = f"Question: {question}\nAnswer: "
                if method_key == "cot_debate":
                    prompt0 = f"To solve the problem, please think and reason step by step, then answer.\nQuestion: {question}\nAnswer: "
                elif method_key == "sd_debate":
                    prompt0 = stem
                else:
                    raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

            else:
                raise ValueError(f"Unsupported dataset: {dataset}. Please handle it in code.")

            # ---------- 第一轮请求 ----------
            history0 = []
            add_message('user', prompt0, history0)
            output0 = ai_request(history0)
            add_message('assistant', output0, history0)
            time.sleep(1)

            # ---------- 构造第二轮 prompt1（按 dataset 自动切换输出约束） ----------
            if dataset_key == "mmlu":
                prompt1 = f"""{stem}
            These are the solution to the problem from another agent:
            {output0}
            Using the reasoning from another agent as additional advice, can you give an updated answer?
            Examine the solution from another agent.
            Please answer with a single letter (A/B/C/D).
            Choice: """
            elif dataset_key == "sqa":
                prompt1 = f"""Question: {question}
            These are the solution to the problem from another agent:
            {output0}
            Using the reasoning from another agent as additional advice, can you give an updated answer?
            Examine the solution from another agent.
            Put your final answer in the form of Yes or No.
            Answer: """
            elif dataset_key == "date":
                prompt1 = f"""Question: {question}
            These are the solution to the problem from another agent:
            {output0}
            Using the reasoning from another agent as additional advice, can you give an updated answer?
            Examine the solution from another agent.
            Put your final answer in the form of MM/DD/YYYY.
            Answer: """
            else:
                # CLUTRR（以及其它“自由文本答案”的任务）
                prompt1 = f"""{question}
            These are the solution to the problem from another agent:
            {output0}
            Using the reasoning from another agent as additional advice, can you give an updated answer?
            Examine the solution from another agent.
            Please give your final answer.
            Answer: """

            # ---------- 第二轮请求 ----------
            history1 = []
            add_message('user', prompt1, history1)
            output = ai_request(history1)
            add_message('assistant', output, history1)
            time.sleep(1)

            # ---------- 判分（按 dataset 自动切换） ----------
            correctness = judge_correctness(dataset_key, answers, output)

            # Save log
            log_filename = f'{dataset}_{i}.json'
            log_path = os.path.join(log_dir, log_filename)
            with open(log_path, 'w') as log_file:
                json.dump({
                    "id": id,
                    "question": question,
                    'answer': answers,
                    "correctness": correctness,
                    "log0": history0,
                    "log1": history1,
                }, log_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline script with dataset and start index arguments")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")
    parser.add_argument("--method", help="sd_debate or cot_debate")
    args = parser.parse_args()

    baseline(args.dataset, args.method, args.start_index)

# python self_debate.py --dataset date --start_index 0 --method sd_debate

