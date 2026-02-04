import json
from pathlib import Path
from typing import Any, List


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