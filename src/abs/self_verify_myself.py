import json
import time
import os
import re
import sys
from typing import Optional


from tqdm import tqdm
from argparse import ArgumentParser


# ============= OpenAI config =============
# 推荐用环境变量：OPENAI_API_KEY / OPENAI_API_BASE


MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-1106")


def add_message(role: str, content: str, history: list):
    history.append({"role": role, "content": content})


from src.llm_client import chat
from src.utils.dataset_io import resolve_data_path, validate_openai_api_key

def ai_request(history: list, t: float = 0.2, max_retries: int = 3) -> str:
    return chat(messages=history, model=MODEL, temperature=t, max_retries=max_retries)



# ============= Prompt builders =============
#第一轮提示词（按数据集+方法）
def build_prompt0(dataset_key: str, method_key: str, data: dict) -> tuple[str, dict]:
    """
    Build first-round prompt.
    Returns: (prompt0, extra_context_dict)
    """
    question = data["question"]

    if dataset_key == "mmlu":
        choice = data["choices"]  # dict: {"A": "...", "B": "...", ...}
        options = "\n".join([f"{k}: {v}" for k, v in choice.items()])
        mcq = f"Question: {question}\n{options}"

        if method_key == "sd_verify":
            prompt0 = mcq + "\nChoice: "
        elif method_key == "cot_verify":
            prompt0 = (
                "Please think step by step, then answer with a single letter (A/B/C/D).\n"
                + mcq + "\nChoice: "
            )
        else:
            raise ValueError("Invalid method. Use 'sd_verify' or 'cot_verify'.")
        return prompt0, {"mcq": mcq}

    elif dataset_key == "clutrr":
        if method_key == "sd_verify":
            prompt0 = f"{question}\nAnswer: "
        elif method_key == "cot_verify":
            prompt0 = f"Please think step by step, then answer.\n{question}\nAnswer: "
        else:
            raise ValueError("Invalid method. Use 'sd_verify' or 'cot_verify'.")
        return prompt0, {"stem": question}

    elif dataset_key == "sqa":
        if method_key == "sd_verify":
            prompt0 = f"Question: {question}\nYour answer should be Yes or No.\nAnswer: "
        elif method_key == "cot_verify":
            prompt0 = f"Please think step by step, then answer Yes or No.\nQuestion: {question}\nAnswer: "
        else:
            raise ValueError("Invalid method. Use 'sd_verify' or 'cot_verify'.")
        return prompt0, {"stem": question}

    elif dataset_key == "date":
        if method_key == "sd_verify":
            prompt0 = f"Question: {question}\nAnswer: "
        elif method_key == "cot_verify":
            prompt0 = f"To solve the problem, please think step by step, then answer.\nQuestion: {question}\nAnswer: "
        else:
            raise ValueError("Invalid method. Use 'sd_verify' or 'cot_verify'.")
        return prompt0, {"stem": question}

    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")


def build_prompt1(dataset_key: str, data: dict, output0: str, guideline: Optional[str], extra: dict) -> str:
    """
    Build second-round VERIFY prompt (uses output0 + optional guideline).
    """
    question = data["question"]

    guide_block = ""
    if guideline:
        guide_block = (
            "\n\nHelpful guideline/checklist (may contain hints):\n"
            f"{guideline}\n"
        )

    if dataset_key == "mmlu":
        mcq = extra.get("mcq", f"Question: {question}")
        return (
            f"{mcq}\n\n"
            f"Your previous answer:\n{output0}\n"
            f"{guide_block}\n"
            "Please double-check carefully. If your previous answer is wrong, correct it.\n"
            "Only output ONE letter (A/B/C/D) as the final answer.\n"
            "Choice: "
        )

    if dataset_key == "sqa":
        return (
            f"Question: {question}\n\n"
            f"Your previous answer:\n{output0}\n"
            f"{guide_block}\n"
            "Please double-check carefully. If wrong, correct it.\n"
            "Only output Yes or No.\n"
            "Answer: "
        )

    if dataset_key == "date":
        return (
            f"Question: {question}\n\n"
            f"Your previous answer:\n{output0}\n"
            f"{guide_block}\n"
            "Please double-check carefully. If wrong, correct it.\n"
            "Only output the final answer in MM/DD/YYYY format.\n"
            "Answer: "
        )

    # clutrr / other free text
    return (
        f"{question}\n\n"
        f"Your previous answer:\n{output0}\n"
        f"{guide_block}\n"
        "Please double-check carefully. If wrong, correct it.\n"
        "Only output your final answer.\n"
        "Answer: "
    )


# ============= Scoring =============
def normalize_date_mmddyyyy(s: str) -> Optional[str]:
    """Extract and normalize a date to MM/DD/YYYY. Returns None if not found."""
    s = str(s)
    # match 1-2 digit month/day, 4 digit year
    m = re.search(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b", s)
    if not m:
        return None
    mm = int(m.group(1))
    dd = int(m.group(2))
    yyyy = m.group(3)
    return f"{mm:02d}/{dd:02d}/{yyyy}"


def extract_mmlu_choice(s: str) -> str:
    """Extract A/B/C/D (prefer last occurrence). Return '' if none."""
    s = str(s)
    matches = re.findall(r"\b([A-Da-d])\b", s)
    if not matches:
        return ""
    return matches[-1].upper()


def judge_correctness(dataset_key: str, gold: str, pred: str) -> str:
    gold_s = str(gold).strip().lower()
    pred_s = str(pred).strip()

    if dataset_key == "mmlu":
        gold_choice = gold_s[:1].upper()  # expected A/B/C/D
        pred_choice = extract_mmlu_choice(pred_s)
        return "True" if pred_choice == gold_choice else "False"

    if dataset_key == "sqa":
        g = gold_s
        first = pred_s.strip().lower().split()[0] if pred_s.strip().split() else ""
        return "True" if first in ("yes", "no") and first == g else "False"

    if dataset_key == "date":
        gold_norm = normalize_date_mmddyyyy(gold_s) or gold_s
        pred_norm = normalize_date_mmddyyyy(pred_s)
        return "True" if (pred_norm is not None and pred_norm.lower() == gold_norm.lower()) else "False"

    # clutrr / others: keep simple (lenient contains)
    return "True" if gold_s in pred_s.lower() else "False"


# ============= Main loop =============
def baseline(dataset: str, method: str, start_index: int = 0, data_path: Optional[str] = None):
    validate_openai_api_key(mock_llm=False)

    dataset_key = dataset.lower()
    method_key = method.lower()

    print(f"Running baseline for dataset={dataset_key}, method={method_key}, start_index={start_index}")

    log_dir = f"log/{method_key}/{dataset_key}"
    os.makedirs(log_dir, exist_ok=True)

    def baseline(dataset: str, method: str, start_index: int = 0, data_path: Optional[str] = None):
        validate_openai_api_key(mock_llm=False)
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in tqdm(enumerate(lines[start_index:], start=start_index), total=len(lines) - start_index):
        data = json.loads(line)

        sample_id = data.get("id", i)
        question = data.get("question", "")
        answers = data.get("answer", "")
        guideline = data.get("guideline", None)

        # ---------- round 1 ----------
        prompt0, extra = build_prompt0(dataset_key, method_key, data)

        history = []
        add_message("user", prompt0, history)
        output0 = ai_request(history)
        add_message("assistant", output0, history)
        time.sleep(1)

        # ---------- round 2 (verify) ----------
        prompt1 = build_prompt1(dataset_key, data, output0, guideline, extra)
        add_message("user", prompt1, history)
        output = ai_request(history)
        add_message("assistant", output, history)
        time.sleep(1)

        # ---------- score ----------
        correctness = judge_correctness(dataset_key, answers, output)

        # ---------- save log ----------
        log_filename = f"{dataset_key}_{i}.json"
        log_path = os.path.join(log_dir, log_filename)
        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(
                {
                    "id": sample_id,
                    "dataset": dataset_key,
                    "method": method_key,
                    "question": question,
                    "answer": answers,
                    "guideline_used": bool(guideline),
                    "correctness": correctness,
                    "output0": output0,
                    "output": output,
                    "log": history,
                },
                lf,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="mmlu / clutrr / sqa / date")
    parser.add_argument("--method", required=True, help="sd_verify or cot_verify")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()

    baseline(args.dataset, args.method, args.start_index, data_path=args.data_path)
