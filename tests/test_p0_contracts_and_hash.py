import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import build_config_hash
from src.common.contracts import validate_output_format, validate_prompt_inputs, validate_prolog_contract
from src.summarize_logs import summarize_logs


def main() -> None:
    cfg = {
        "dataset": "gsm8k",
        "method": "cot_selfguide",
        "solve_model": "x",
        "guide_model": "y",
        "solve_temperature": 0.2,
        "guide_temperature": 0.7,
        "max_retries": 3,
        "prompt_contract_version": "p0.v1",
        "prompt_versions": {"round_a_draft": "round-a.v1"},
        "prolog_contract_version": "p0.v1",
        "swipl_schema_version": "1.0",
        "meta_interpreter": "iter_deep_with_proof",
        "max_depth": 25,
        "prolog_max_result": 20,
        "force_task_type": None,
        "mock_profile": None,
        "mock_llm": True,
        "mock_prolog": True,
    }
    h1 = build_config_hash(cfg)
    cfg2 = dict(cfg)
    cfg2["timestamp"] = 12345
    h2 = build_config_hash(cfg2)
    assert h1 == h2

    bad_round = validate_prompt_inputs("round_c_solve", {"dataset_key": "gsm8k", "qblock": "q"})
    assert not bad_round.ok and bad_round.error_code == "PROMPT_CONTRACT_MISSING_FIELD"

    assert not validate_output_format("sqa", "maybe").ok
    assert validate_output_format("mmlu", "A").ok

    assert not validate_prolog_contract(["a(1)", "b(2).?"]).ok
    assert validate_prolog_contract(["a(1).", "query(1). "]).ok

    tmp = ROOT / "tmp_schema_test"
    tmp.mkdir(exist_ok=True)
    (tmp / "p0_1.json").write_text(json.dumps({
        "correctness": True,
        "route": "executor",
        "error_code": "OK",
        "config_hash": h1,
        "prolog": {
            "enabled": True,
            "proof": "ok",
            "swipl": {"ok": True},
            "swipl_contract": {"schema_version": "1.0", "legacy": False},
        },
    }), encoding="utf-8")
    (tmp / "p0_2.json").write_text(json.dumps({
        "correctness": False,
        "route": "llm_only",
        "error_code": "LEGACY_SCHEMA",
        "config_hash": h1,
        "prolog": {
            "enabled": True,
            "swipl": {"ok": False},
            "swipl_contract": {"schema_version": "LEGACY_SCHEMA", "legacy": True},
        },
    }), encoding="utf-8")

    summary = summarize_logs(tmp)
    assert summary["N"] >= 2
    assert summary["error_code_distribution"]["OK"] >= 1
    assert summary["schema_version_distribution"]["LEGACY_SCHEMA"] >= 1
    assert summary["legacy_schema_hits"] >= 1


if __name__ == "__main__":
    main()