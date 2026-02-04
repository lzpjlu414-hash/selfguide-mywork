import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

DEBUG = os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
_CLIENTS: Dict[Tuple[str, float], OpenAI] = {}

def _resolve_base_url(base_url: Optional[str] = None) -> Optional[str]:
    env_base = (os.getenv("OPENAI_API_BASE") or "").strip()
    explicit = (base_url or "").strip()
    return explicit or env_base or None


def validate_llm_config(mock_llm: bool, base_url: Optional[str] = None) -> None:
    if mock_llm:
        return
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    resolved_base = _resolve_base_url(base_url)
    if not api_key:
        base_desc = resolved_base or f"EMPTY (default {DEFAULT_BASE_URL})"
        raise RuntimeError(
            "Missing required env: OPENAI_API_KEY. "
            f"OPENAI_API_BASE={base_desc}."
        )


def get_client(base_url: Optional[str] = None, timeout: float = 30.0) -> OpenAI:
    resolved_base = _resolve_base_url(base_url) or ""
    key = (resolved_base, float(timeout))
    if key not in _CLIENTS:
        _CLIENTS[key] = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=resolved_base or None,
            timeout=timeout,
        )
    return _CLIENTS[key]


def resolve_model(
    model: Optional[str],
    purpose: Optional[str] = None,
    env_key: Optional[str] = None,
    default: str = "gpt-3.5-turbo-1106",
    **kwargs: Any,
) -> str:
    if model:
        return model
    _ = kwargs
    if env_key:
        env_value = os.getenv(env_key, "").strip()
    elif purpose:
        if purpose == "guide":
            env_value = os.getenv("OPENAI_GUIDE_MODEL", "").strip()
        else:
            env_value = os.getenv("OPENAI_MODEL", "").strip()
    else:
        env_value = os.getenv("OPENAI_MODEL", "").strip()
    return env_value or default


def _chat(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    timeout: float = 30.0,
    base_url: Optional[str] = None,
) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = get_client(base_url=base_url, timeout=timeout).chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # pragma: no cover - passthrough retry
            last_err = exc
            if DEBUG:
                print(
                    f"[llm_client] attempt {attempt + 1}/{max_retries} failed: {repr(exc)}",
                    file=sys.stderr,
                )
            time.sleep(1 + attempt)
    raise RuntimeError(f"Request failed after {max_retries} retries: {last_err}")

def _mock_seed(dataset_key: str, prompt_type: str, mock_profile: Optional[str], content: str) -> int:
    tag = f"{dataset_key}::{prompt_type}::{mock_profile or 'default'}::{content}"
    return int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8], 16)


def mock_answer_for_dataset(dataset_key: str, seed: int) -> str:
    dataset_key = (dataset_key or "generic").lower()
    if dataset_key == "mmlu":
        return "ABCD"[seed % 4]
    if dataset_key == "sqa":
        return "Yes" if seed % 2 == 0 else "No"
    if dataset_key == "date":
        month = seed % 12 + 1
        day = (seed // 12) % 28 + 1
        year = 2020 + (seed % 5)
        return f"{month:02d}/{day:02d}/{year}"
    if dataset_key == "gsm8k":
        return f"#### {100 + (seed % 900)}"
    if dataset_key in ("prontoqa", "proofwriter"):
        return "True" if seed % 2 == 0 else "False"
    if dataset_key == "clutrr":
        relations = [
            "father",
            "mother",
            "brother",
            "sister",
            "grandmother",
            "grandfather",
            "aunt",
            "uncle",
        ]
        relation = relations[seed % len(relations)]
        return f"A is B's {relation}."
    return f"Answer {seed % 100}"


def _mock_guideline(dataset_key: str, seed: int) -> str:
    dataset_key = (dataset_key or "generic").lower()
    if dataset_key in ("gsm8k", "prontoqa", "proofwriter"):
        task_type = "Yes" if seed % 2 == 0 else "Partial"
    else:
        task_type = "No"
    return (
        f"task_type: {task_type}\n"
        "schema: |\n"
        f"  - Use lowercase constants for {dataset_key} entities.\n"
        "  - Keep predicate names consistent (snake_case).\n"
        "  - Represent Unknown with explicit 'unknown' atoms.\n"
        "query_goal: |\n"
        "  - Derive the final answer in the required output format.\n"
        "  - Ensure the query targets only one concrete result.\n"
        "fallback: |\n"
        "  - If parsing fails, answer using the last known consistent step.\n"
        "  - If multiple answers, pick the most specific one.\n"
    )


def _mock_prolog(dataset_key: str, seed: int) -> str:
    answer_atom = "ans" + str(seed % 10)
    return (
        f"fact({answer_atom}).\n"
        f"answer(X) :- fact(X).\n"
        "answer(X).\n"
    )


def _mock_response(
    messages: List[Dict[str, Any]],
    dataset_key: Optional[str],
    prompt_type: Optional[str],
    mock_profile: Optional[str],
) -> str:
    dataset = (dataset_key or "generic").lower()
    prompt_kind = (prompt_type or "solve").lower()
    content = "\n".join(str(m.get("content", "")) for m in messages)
    seed = _mock_seed(dataset, prompt_kind, mock_profile, content)

    if prompt_kind in ("guideline", "guideline_merge"):
        return _mock_guideline(dataset, seed)
    if prompt_kind == "prolog":
        return _mock_prolog(dataset, seed)

    answer = mock_answer_for_dataset(dataset, seed)
    if prompt_kind == "draft":
        return f"Reasoning: mock reasoning steps.\nFinal answer: {answer}"
    return answer

def chat_complete(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    timeout: float = 30.0,
    base_url: Optional[str] = None,
    mock_llm: bool = False,
    mock_profile: Optional[str] = None,
    dataset_key: Optional[str] = None,
    prompt_type: Optional[str] = None,
    **kwargs: Any,
) -> str:
    if mock_llm:
        return _mock_response(messages, dataset_key, prompt_type, mock_profile)
    validate_llm_config(mock_llm=False, base_url=base_url)
    _ = kwargs
    return _chat(
        messages=messages,
        model=resolve_model(model),
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        base_url=base_url,
    )