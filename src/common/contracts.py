from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


PROMPT_CONTRACT_VERSION = "p0.v1"
PROLOG_CONTRACT_VERSION = "p0.v1"

PROMPT_VERSIONS = {
    "round_a_draft": "round-a.v1",
    "round_b_guideline": "round-b.v1",
    "round_c_solve": "round-c.v1",
    "round_c_prolog": "round-c-prolog.v1",
}

PROMPT_REQUIRED_FIELDS = {
    "round_a_draft": ("qblock", "format_rule"),
    "round_b_guideline": ("dataset_key", "qblock", "format_rule"),
    "round_c_solve": ("dataset_key", "qblock", "guideline", "format_rule", "method_key"),
    "round_c_prolog": ("qblock", "guideline"),
}


@dataclass(frozen=True)
class ContractCheckResult:
    ok: bool
    error_code: str | None = None
    message: str | None = None


def validate_prompt_inputs(round_key: str, payload: Mapping[str, Any]) -> ContractCheckResult:
    required_fields = PROMPT_REQUIRED_FIELDS.get(round_key, ())
    missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
    if missing:
        return ContractCheckResult(
            ok=False,
            error_code="PROMPT_CONTRACT_MISSING_FIELD",
            message=f"{round_key} missing fields: {', '.join(missing)}",
        )
    return ContractCheckResult(ok=True)


def validate_output_format(dataset_key: str, text: str) -> ContractCheckResult:
    text = (text or "").strip()
    if not text:
        return ContractCheckResult(ok=False, error_code="PROMPT_CONTRACT_INVALID_OUTPUT", message="empty output")

    if dataset_key == "mmlu" and not re.fullmatch(r"[A-D]", text):
        return ContractCheckResult(ok=False, error_code="PROMPT_CONTRACT_INVALID_OUTPUT", message="mmlu must be A/B/C/D")
    if dataset_key == "sqa" and text not in {"Yes", "No"}:
        return ContractCheckResult(ok=False, error_code="PROMPT_CONTRACT_INVALID_OUTPUT", message="sqa must be Yes/No")
    if dataset_key == "date" and not re.fullmatch(r"\d{2}/\d{2}/\d{4}", text):
        return ContractCheckResult(
            ok=False,
            error_code="PROMPT_CONTRACT_INVALID_OUTPUT",
            message="date must be MM/DD/YYYY",
        )
    return ContractCheckResult(ok=True)


def validate_prolog_contract(clauses: list[str]) -> ContractCheckResult:
    if not clauses:
        return ContractCheckResult(ok=False, error_code="PROLOG_CONTRACT_INVALID", message="empty clauses")
    if not all(isinstance(clause, str) and clause.strip().endswith(".") for clause in clauses):
        return ContractCheckResult(
            ok=False,
            error_code="PROLOG_CONTRACT_INVALID",
            message="every clause must be a single line ending with '.'",
        )
    query = clauses[-1].strip()
    if ":-" in query:
        return ContractCheckResult(
            ok=False,
            error_code="PROLOG_CONTRACT_INVALID",
            message="last line must be a query, not a rule",
        )
    return ContractCheckResult(ok=True)