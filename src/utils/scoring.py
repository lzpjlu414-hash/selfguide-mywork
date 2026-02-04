import re
from typing import Any, Iterable, Optional


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


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_light_text(value: Any) -> str:
    """
    Light tolerance: drop punctuation and collapse whitespace.
    Used only after strict normalization fails (e.g., CLUTRR).
    """
    text = _normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_final_segment(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####")[-1].strip()


def normalize_date_mmddyyyy(s: str) -> Optional[str]:
    s = str(s)
    m = re.search(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b", s)
    if not m:
        return None
    mm = int(m.group(1))
    dd = int(m.group(2))
    yyyy = m.group(3)
    return f"{mm:02d}/{dd:02d}/{yyyy}"


def extract_mmlu_choice(text: str) -> str:
    s = str(text)
    matches = re.findall(r"\b([A-Da-d])\b", s)
    return matches[-1].upper() if matches else ""


def _normalize_yes_no(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    first = text.split()[0]
    if first in ("yes", "no"):
        return first
    return ""


def _iter_gold_values(gold: Any) -> Iterable[str]:
    if isinstance(gold, list):
        for item in gold:
            yield str(item)
    else:
        yield str(gold)


def judge_correctness(dataset_key: str, gold: Any, pred: Any) -> str:
    dataset_key = (dataset_key or "").lower()
    pred_text = str(pred or "")

    if dataset_key == "mmlu":
        gold_choice = extract_mmlu_choice(str(gold))
        pred_choice = extract_mmlu_choice(pred_text)
        return "True" if gold_choice and pred_choice and gold_choice == pred_choice else "False"

    if dataset_key == "sqa":
        gold_norm = _normalize_yes_no(gold)
        pred_norm = _normalize_yes_no(pred)
        return "True" if gold_norm and pred_norm and gold_norm == pred_norm else "False"

    if dataset_key == "date":
        gold_norm = normalize_date_mmddyyyy(str(gold)) or _normalize_text(gold)
        pred_norm = normalize_date_mmddyyyy(pred_text)
        return "True" if pred_norm and pred_norm.lower() == str(gold_norm).lower() else "False"

    if dataset_key == "gsm8k":
        gold_num = extract_gsm8k_final_number(str(gold))
        pred_num = extract_gsm8k_final_number(pred_text)
        return "True" if gold_num and pred_num and gold_num == pred_num else "False"

    if dataset_key == "clutrr":
        gold_strict = _normalize_text(_extract_final_segment(str(gold)))
        pred_strict = _normalize_text(_extract_final_segment(pred_text))
        if gold_strict and pred_strict and gold_strict == pred_strict:
            return "True"
        gold_light = _normalize_light_text(_extract_final_segment(str(gold)))
        pred_light = _normalize_light_text(_extract_final_segment(pred_text))
        return "True" if gold_light and pred_light and gold_light == pred_light else "False"

    if dataset_key in ("prontoqa", "proofwriter"):
        pred_norm = _normalize_text(pred_text)
        for gold_value in _iter_gold_values(gold):
            if _normalize_text(gold_value) == pred_norm:
                return "True"
        return "False"

    gold_norm = _normalize_text(str(gold))
    pred_norm = _normalize_text(pred_text)
    return "True" if gold_norm and pred_norm and gold_norm == pred_norm else "False"