import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abs.self_guide_myself import parse_caring_swipl_answer, SWIPL_OUT_SCHEMA_VERSION


def main() -> None:
    success = {
        "schema_version": SWIPL_OUT_SCHEMA_VERSION,
        "ok": True,
        "answer": "42",
        "proof": "p(42)",
        "error_code": None,
        "raw": {"foo": "bar"},
    }
    parsed = parse_caring_swipl_answer(json.dumps(success))
    assert parsed["ok"] is True, parsed
    assert parsed["answer"] == "42", parsed
    assert parsed["proof"] == "p(42)", parsed
    assert parsed["error_code"] is None, parsed

    invalid = {"ok": True, "answer": "42"}
    parsed_invalid = parse_caring_swipl_answer(json.dumps(invalid))
    assert parsed_invalid["ok"] is False, parsed_invalid
    assert parsed_invalid["error_code"] in {"LEGACY_SCHEMA", "SCHEMA_MISSING_REQUIRED_KEYS"}, parsed_invalid

    legacy = {"answer": ["42"], "proofs": ["legacy"]}
    parsed_legacy = parse_caring_swipl_answer(json.dumps(legacy))
    assert parsed_legacy["legacy"] is True, parsed_legacy
    assert parsed_legacy["error_code"] == "LEGACY_SCHEMA", parsed_legacy


if __name__ == "__main__":
    main()