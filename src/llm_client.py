import os
import sys
import time
from typing import List, Dict, Any


from openai import OpenAI

DEBUG = os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes")
_CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_API_BASE") or None,
    timeout=30,  # 新增：30秒超时
)


def chat(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _CLIENT.chat.completions.create(
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
