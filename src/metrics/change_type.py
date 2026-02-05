import re
from typing import Optional

TRAILING_PUNCTUATION = ".。!?！？"


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"\*+", "", normalized)
    normalized = re.sub(r"^final\s*answer\s*:\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(TRAILING_PUNCTUATION + " ")
    return normalized.strip()


def compute_change_type(
    draft: Optional[str],
    final: Optional[str],
    prolog_used: bool,
    prolog_overruled: bool,
) -> str:
    draft_norm = normalize_answer(draft)
    final_norm = normalize_answer(final)

    if draft_norm == final_norm:
        return "copy"
    if prolog_used and prolog_overruled:
        return "corrected_by_prolog"
    return "revise"