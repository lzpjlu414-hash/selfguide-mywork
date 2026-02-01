import json
import time
import openai
from tqdm import tqdm
import os
from argparse import ArgumentParser
import re #正则：主要用于 MMLU 的 A/B/C/D 抽取：re.search(r"\b([a-d])\b", pred_s)

openai.api_key = ""
openai.api_base = ""

MODEL = "gpt-3.5-turbo-1106" #全局model


def add_message(role, content, history):
    history.append({"role": role, "content": content})


def ai_request(history, t=0.2):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=history,
        temperature=t,
    )
    return response["choices"][0]["message"]["content"]


def judge_correctness(dataset_key: str, gold: str, pred: str) -> str:
    gold_s = str(gold).strip().lower()
    pred_s = str(pred).strip().lower()

    if dataset_key == "mmlu":
        # gold 通常是 A/B/C/D
        m = re.search(r"\b([a-d])\b", pred_s)
        pred_choice = m.group(1) if m else (pred_s[:1] if pred_s else "")
        return "True" if pred_choice == gold_s[:1].lower() else "False"

    if dataset_key == "sqa":
        first = pred_s.split()[0] if pred_s.split() else ""
        return "True" if first in ("yes", "no") and first == gold_s else "False"

    # date / CLUTRR：沿用“子串包含”
    return "True" if gold_s in pred_s else "False"


def baseline(dataset, method, start_index=0):
    print(f"Running baseline for dataset: {dataset}")

    dataset_key = dataset.lower()
    method_key = method.lower()

    log_dir = f'log/{method}/{dataset}'
    os.makedirs(log_dir, exist_ok=True)

    with open(f"log_guideline/{dataset}.jsonl", 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in tqdm(enumerate(lines[start_index:], start=start_index)):
        data = json.loads(line)

        id = data['id']
        question = data['question']
        answers = data['answer']
        guideline = data.get('guideline', None)  # 目前没用到，先保留

        # ---------- 构造第一轮 prompt0（按 dataset + method 自动切换） ----------
        mcq = None  # 只给 mmlu 用
        stem = None

        if dataset_key == "mmlu":
            choice = data["choices"]  # dict: {"A": "...", "B": "...", ...}
            options = "\n".join([f"{k}: {v}" for k, v in choice.items()])
            mcq = f"Question: {question}\n{options}"

            if method_key == "cot_debate":
                prompt0 = (
                    "Please think step by step, then answer with a single letter (A/B/C/D).\n"
                    + mcq + "\nChoice: "
                )
            elif method_key == "sd_debate":
                prompt0 = mcq + "\nChoice: "
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

        elif dataset_key == "clutrr":
            stem = question
            if method_key == "cot_debate":
                prompt0 = "Please think step by step, then answer.\n" + stem + "\nAnswer: "
            elif method_key == "sd_debate":
                prompt0 = stem + "\nAnswer: "
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

        elif dataset_key == "sqa":
            stem = question
            if method_key == "cot_debate":
                prompt0 = (
                    "Please think step by step, then answer Yes or No.\n"
                    f"Question: {stem}\nAnswer: "
                )
            elif method_key == "sd_debate":
                prompt0 = (
                    f"Question: {stem}\nYour answer should be Yes or No.\nAnswer: "
                )
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

        elif dataset_key == "date":
            stem = question
            if method_key == "cot_debate":
                prompt0 = (
                    "To solve the problem, please think and reason step by step, then answer.\n"
                    f"Question: {stem}\nAnswer: "
                )
            elif method_key == "sd_debate":
                prompt0 = f"Question: {stem}\nAnswer: "
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
            prompt1 = f"""{mcq}
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
            # CLUTRR（以及其它自由文本）
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

        # ---------- 判分 ----------
        correctness = judge_correctness(dataset_key, answers, output)

        # ---------- 保存 log ----------
        log_filename = f'{dataset}_{i}.json'
        log_path = os.path.join(log_dir, log_filename)
        with open(log_path, 'w', encoding='utf-8') as log_file:
            json.dump({
                "id": id,
                "question": question,
                "answer": answers,
                "correctness": correctness,
                "log0": history0,
                "log1": history1,
            }, log_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline script with dataset and start index arguments")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")
    parser.add_argument("--method", help="sd_debate or cot_debate")
    args = parser.parse_args()

    baseline(args.dataset, args.method, args.start_index)
