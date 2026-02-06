import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abs.self_guide_myself import (
    NO_PROOF_RETURNED_SENTINEL,
    SWIPL_OUT_SCHEMA_VERSION,
    is_nonempty_proof,
    parse_caring_swipl_answer,
)


def test_swipl_payload_has_solution_count_int_ge_0() -> None:
    """call_swipl 标准 payload 里 solution_count 应该是 >=0 的整数。"""
    payload = {
        "schema_version": SWIPL_OUT_SCHEMA_VERSION,
        "ok": True,
        "answer": "42",
        "proof": "p(42)",
        "error_code": None,
        "solution_count": 2,
        "raw": {"results_count": 2, "results": ["42", "43"]},
    }
    parsed = parse_caring_swipl_answer(json.dumps(payload))

    assert isinstance(parsed["solution_count"], int), parsed
    assert parsed["solution_count"] >= 0, parsed
    assert parsed["solution_count_valid"] is True, parsed


def test_swipl_payload_proof_sentinel_normalized() -> None:
    """proof 为空字符串或 sentinel 时，proof_nonempty 必须判为 False。"""
    payload_empty_proof = {
        "schema_version": SWIPL_OUT_SCHEMA_VERSION,
        "ok": True,
        "answer": "42",
        "proof": "",
        "error_code": None,
        "raw": {},
    }
    payload_sentinel_proof = {
        "schema_version": SWIPL_OUT_SCHEMA_VERSION,
        "ok": True,
        "answer": "42",
        "proof": NO_PROOF_RETURNED_SENTINEL,
        "error_code": None,
        "raw": {},
    }

    parsed_empty = parse_caring_swipl_answer(json.dumps(payload_empty_proof))
    parsed_sentinel = parse_caring_swipl_answer(json.dumps(payload_sentinel_proof))

    assert is_nonempty_proof(parsed_empty["proof"]) is False
    assert is_nonempty_proof(parsed_sentinel["proof"]) is False


def test_parse_rejects_legacy_schema_missing_required_keys() -> None:

    invalid = {"ok": True, "answer": "42"}
    parsed_invalid = parse_caring_swipl_answer(json.dumps(invalid))
    assert parsed_invalid["ok"] is False, parsed_invalid
    assert parsed_invalid["error_code"] in {"LEGACY_SCHEMA", "SCHEMA_MISSING_REQUIRED_KEYS"}, parsed_invalid

