import json
import time
from tqdm import tqdm
import os
import re #正则：主要用于 MMLU 的 A/B/C/D 抽取：re.search(r"\b([a-d])\b", pred_s)
from src.llm_client import chat_complete, resolve_model
from src.utils.dataset_io import resolve_data_path, validate_openai_api_key
from src.abs.common_entry import create_base_parser, ensure_log_dir, run_main, utc_now_iso, write_json

MODEL = resolve_model(None, purpose="solve")


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


def baseline(dataset, method, start_index=0, data_path=None, log_dir_override=None, mock_llm=False, mock_profile=None):
    validate_openai_api_key(mock_llm=mock_llm)
    print(f"Running baseline for dataset: {dataset}")

    dataset_key = dataset.lower()
    method_key = method.lower()

    log_dir = ensure_log_dir(log_dir_override or f'log/{method_key}/{dataset_key}')
    data_path = resolve_data_path(dataset_key, data_path)

    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in tqdm(enumerate(lines[start_index:], start=start_index)):
        data = json.loads(line)

        id = data['id']
        question = data['question']
        answers = data['answer']

        # ---------- 构造第一轮 prompt0（按 dataset + method 自动切换） ----------
        mcq = None  # 只给 mmlu 用

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
            if method_key == "cot_debate":
                prompt0 = "Please think step by step, then answer.\n" + question + "\nAnswer: "
            elif method_key == "sd_debate":
                prompt0 = question + "\nAnswer: "
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

        elif dataset_key == "sqa":
            if method_key == "cot_debate":
                prompt0 = (
                    "Please think step by step, then answer Yes or No.\n"
                    f"Question: {question}\nAnswer: "
                )
            elif method_key == "sd_debate":
                prompt0 = (
                    f"Question: {question}\nYour answer should be Yes or No.\nAnswer: "
                )
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

        elif dataset_key == "date":
            if method_key == "cot_debate":
                prompt0 = (
                    "To solve the problem, please think and reason step by step, then answer.\n"
                    f"Question: {question}\nAnswer: "
                )
            elif method_key == "sd_debate":
                prompt0 = f"Question: {question}\nAnswer: "
            else:
                raise ValueError("Invalid method. Method must be 'sd_debate' or 'cot_debate'.")

        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Please handle it in code.")

        # ---------- 第一轮请求 ----------
        history0 = [{"role": "user", "content": prompt0}]
        output0 = chat_complete(
            history0,
            model=MODEL,
            temperature=0.2,
            mock_llm=mock_llm,
            mock_profile=mock_profile,
            dataset_key=dataset_key,
            prompt_type="debate_round1",
        )
        history0.append({"role": "assistant", "content": output0})
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
        history1 = [{"role": "user", "content": prompt1}]
        output = chat_complete(
            history1,
            model=MODEL,
            temperature=0.2,
            mock_llm=mock_llm,
            mock_profile=mock_profile,
            dataset_key=dataset_key,
            prompt_type="debate_round2",
        )
        history1.append({"role": "assistant", "content": output})
        time.sleep(1)

        # ---------- 判分 ----------
        correctness = judge_correctness(dataset_key, answers, output)

        # ---------- 保存 log ----------
        log_filename = f'{dataset}_{i}.json'
        log_path = os.path.join(log_dir, log_filename)
        write_json(log_path, {
            "meta": {
                "entry": "self_debate_myself",
                "dataset": dataset_key,
                "method": method_key,
                "sample_index": i,
                "route": "llm_only",
                "timestamp_utc": utc_now_iso(),
                "error_code": None,
            },
                "id": id,
                "question": question,
                "answer": answers,
                "correctness": correctness,
                "log0": history0,
                "log1": history1,
            })


if __name__ == "__main__":
    def _main():
        parser = create_base_parser(
            "Self-debate baseline entrypoint",
            dataset_help="mmlu / clutrr / sqa / date",
            method_help="sd_debate or cot_debate",
            include_log_dir=True,
        )
        args = parser.parse_args()
        baseline(
            args.dataset,
            args.method,
            args.start_index,
            data_path=args.data_path,
            log_dir_override=args.log_dir,
            mock_llm=args.mock_llm,
            mock_profile=args.mock_profile,
        )

    run_main(_main)
