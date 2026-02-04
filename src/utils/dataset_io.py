import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


def load_jsonl(path: str) -> List[Any]:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if "\n" not in text:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _format_data_path_error(dataset_key: str, tried_paths: Sequence[str], explicit: bool) -> str:
    prefix = "Provided --data_path does not exist." if explicit else "Dataset file not found."
    return (
        f"{prefix} Dataset={dataset_key}. "
        f"尝试过哪些路径: {', '.join(tried_paths)}. "
        "如何通过 --data_path 指定: "
        "python -m src.run_experiment --dataset <dataset> --method <method> --data_path /path/to/file.jsonl"
    )


def resolve_data_path(dataset_key: str, data_path: Optional[str] = None, base_dir: Optional[Path] = None) -> str:
    root = Path(base_dir) if base_dir is not None else _project_root()
    normalized = dataset_key.lower()
    tried: List[str] = []

    if data_path:
        candidate = Path(data_path)
        tried.append(str(candidate))
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(_format_data_path_error(normalized, tried, explicit=True))

    candidates = [
        root / "data" / normalized / "test.jsonl",
        root / "data" / normalized / "test_small.jsonl",
        root / "log" / f"{normalized}.jsonl",
        root / "log_guideline" / f"{normalized}.jsonl",
    ]
    for candidate in candidates:
        tried.append(str(candidate))
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(_format_data_path_error(normalized, tried, explicit=False))


def validate_openai_api_key(mock_llm: bool) -> None:
    if mock_llm:
        return
    if os.getenv("OPENAI_API_KEY", "").strip():
        return
    raise ValueError(
        "OPENAI_API_KEY is empty. "
        "In PowerShell, set it with: $env:OPENAI_API_KEY='your-key'."
    )

def _first_present(data: dict, keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in data and data[key] is not None:
            value = data[key]
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_after_hash(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####")[-1].strip()


def _normalize_mmlu_choices(raw_choices: Any) -> Optional[dict]:
    if isinstance(raw_choices, dict):
        if "label" in raw_choices and "text" in raw_choices:
            labels = raw_choices.get("label") or []
            texts = raw_choices.get("text") or []
            if len(labels) >= 4 and len(texts) >= 4:
                return {str(lab).upper(): str(txt) for lab, txt in zip(labels, texts)}
        if all(k in raw_choices for k in ("A", "B", "C", "D")):
            return {k: _normalize_text(raw_choices[k]) for k in ("A", "B", "C", "D")}
    if isinstance(raw_choices, list) and len(raw_choices) >= 4:
        return {chr(ord("A") + i): _normalize_text(raw_choices[i]) for i in range(4)}
    return None


def _log_skip(logger: logging.Logger, dataset_key: str, index: int, reason: str, sample_id: Any) -> None:
    suffix = f" id={sample_id}" if sample_id is not None else ""
    logger.error("Skipping %s sample %s%s: %s", dataset_key, index, suffix, reason)


def _normalize_mmlu(sample: dict, index: int, logger: logging.Logger) -> Optional[dict]:
    question = _normalize_text(_first_present(sample, ("question", "prompt", "query")))
    choices = _normalize_mmlu_choices(sample.get("choices") or sample.get("options"))
    answer = _normalize_text(_first_present(sample, ("answer", "answerKey", "label")))
    sample_id = sample.get("id", index)
    if not question:
        _log_skip(logger, "mmlu", index, "missing question", sample_id)
        return None
    if not choices:
        _log_skip(logger, "mmlu", index, "missing choices A-D", sample_id)
        return None
    if not answer:
        _log_skip(logger, "mmlu", index, "missing answer", sample_id)
        return None
    return {"id": sample_id, "question": question, "choices": choices, "answer": answer}


def _normalize_sqa(sample: dict, index: int, logger: logging.Logger) -> Optional[dict]:
    question = _normalize_text(_first_present(sample, ("question", "prompt", "query")))
    answer_raw = _first_present(sample, ("answer", "label"))
    sample_id = sample.get("id", index)
    if not question:
        _log_skip(logger, "sqa", index, "missing question", sample_id)
        return None
    if answer_raw is None:
        _log_skip(logger, "sqa", index, "missing answer", sample_id)
        return None
    if isinstance(answer_raw, bool):
        answer = "Yes" if answer_raw else "No"
    else:
        answer = _normalize_text(answer_raw)
    return {"id": sample_id, "question": question, "answer": answer}


def _normalize_date(sample: dict, index: int, logger: logging.Logger) -> Optional[dict]:
    question = _normalize_text(_first_present(sample, ("question", "prompt", "query")))
    answer_raw = _normalize_text(_first_present(sample, ("answer", "label")))
    sample_id = sample.get("id", index)
    if not question:
        _log_skip(logger, "date", index, "missing question", sample_id)
        return None
    if not answer_raw:
        _log_skip(logger, "date", index, "missing answer", sample_id)
        return None
    return {"id": sample_id, "question": question, "answer": answer_raw}


def _normalize_clutrr(sample: dict, index: int, logger: logging.Logger) -> Optional[dict]:
    question = _normalize_text(_first_present(sample, ("question", "prompt", "query")))
    answer_raw = _normalize_text(_first_present(sample, ("answer", "label")))
    sample_id = sample.get("id", index)
    if not question:
        _log_skip(logger, "clutrr", index, "missing question", sample_id)
        return None
    if not answer_raw:
        _log_skip(logger, "clutrr", index, "missing answer", sample_id)
        return None
    return {"id": sample_id, "question": question, "answer": _extract_after_hash(answer_raw)}


def _normalize_prontoqa_proofwriter(sample: dict, index: int, logger: logging.Logger, dataset_key: str) -> Optional[dict]:
    question = _first_present(sample, ("question", "prompt", "query"))
    context = _first_present(sample, ("context",))
    sample_id = sample.get("id", index)
    if question is None and context is None:
        _log_skip(logger, dataset_key, index, "missing question/prompt/context/query", sample_id)
        return None
    answer = _first_present(sample, ("answer", "gold", "label"))
    if answer is None:
        _log_skip(logger, dataset_key, index, "missing answer/gold/label", sample_id)
        return None
    question_text = _normalize_text(question or context)
    context_text = _normalize_text(context) if context is not None else ""
    payload = {"id": sample_id, "question": question_text, "answer": answer}
    if context_text:
        payload["context"] = context_text
    return payload


def normalize_dataset_samples(samples: List[Any], dataset_key: str) -> List[dict]:
    logger = logging.getLogger(__name__)
    normalized: List[dict] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            _log_skip(logger, dataset_key, index, "sample is not a dict", None)
            continue
        if dataset_key == "mmlu":
            item = _normalize_mmlu(sample, index, logger)
        elif dataset_key == "sqa":
            item = _normalize_sqa(sample, index, logger)
        elif dataset_key == "date":
            item = _normalize_date(sample, index, logger)
        elif dataset_key == "clutrr":
            item = _normalize_clutrr(sample, index, logger)
        elif dataset_key in ("prontoqa", "proofwriter"):
            item = _normalize_prontoqa_proofwriter(sample, index, logger, dataset_key)
        elif dataset_key == "gsm8k":
            question = _normalize_text(_first_present(sample, ("question", "prompt", "query")))
            answer = _normalize_text(_first_present(sample, ("answer", "label")))
            sample_id = sample.get("id", index)
            if not question or not answer:
                _log_skip(logger, dataset_key, index, "missing question/answer", sample_id)
                item = None
            else:
                item = {"id": sample_id, "question": question, "answer": answer}
        else:
            item = sample
        if item:
            normalized.append(item)
    return normalized


def load_dataset(path: str, dataset_key: str) -> List[dict]:
    samples = load_jsonl(path)
    return normalize_dataset_samples(samples, dataset_key)