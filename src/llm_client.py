import os
import time
from typing import List, Dict, Any

from openai import OpenAI

_CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_API_BASE") or None,
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
            time.sleep(1 + attempt)
    raise RuntimeError(f"Request failed after {max_retries} retries: {last_err}")
