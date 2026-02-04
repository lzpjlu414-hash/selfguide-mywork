import os
import sys
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI

DEBUG = os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes")
_CLIENT = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE") or None,
            timeout=30,  # 新增：30秒超时
        )
    return _CLIENT


def resolve_model(
    model: Optional[str],
    env_key: str = "OPENAI_MODEL",
    default: str = "gpt-3.5-turbo-1106",
) -> str:
    if model:
        return model
    env_value = os.getenv(env_key, "").strip()
    return env_value or default


def chat(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
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

def chat_complete(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    **kwargs: Any,
) -> str:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it before making real API calls."
        )
    _ = kwargs
    return chat(
        messages=messages,
        model=resolve_model(model),
        temperature=temperature,
        max_retries=max_retries,
    )