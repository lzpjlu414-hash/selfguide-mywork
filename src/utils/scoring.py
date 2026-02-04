import re


def extract_gsm8k_final_number(text: str) -> str:
    s = str(text or "")
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", s)
    if match:
        return match.group(1)

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else ""


def postprocess_pred(pred: str) -> str:
    pred = (pred or "").strip()
    num = extract_gsm8k_final_number(pred)
    return num if num else pred


def judge_correctness(gold: str, pred: str) -> str:
    gold_num = extract_gsm8k_final_number(gold)
    pred_num = extract_gsm8k_final_number(pred)
    return "True" if gold_num and pred_num and gold_num == pred_num else "False"