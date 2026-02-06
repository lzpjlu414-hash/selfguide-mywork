import json
from pathlib import Path

from src.summarize_logs import summarize_logs


def test_summarize_logs_route_buckets_and_rates(tmp_path: Path) -> None:
    logs = [
        {
            "correctness": True,
            "route": "executor",
            "error_code": "OK",
            "config_hash": "h1",
            "prolog_used": True,
            "prolog_ok": True,
            "solution_count": 1,
            "proof_shape_ok": True,
            "proof_nonempty": True,
            "prolog_overruled": True,
            "final_modified_by_prolog": True,
            "prolog_inconclusive": False,
            "multi_solution_conflict": False,
            "prolog_error_code": "OK",
            "prolog": {"enabled": True},
        },
        {
            "correctness": False,
            "route": "verifier",
            "error_code": "OK",
            "config_hash": "h1",
            "prolog_used": True,
            "prolog_ok": True,
            "solution_count": 2,
            "proof_nonempty": True,
            "prolog_overruled": True,
            "final_modified_by_prolog": False,
            "prolog_inconclusive": False,
            "multi_solution_conflict": True,
            "prolog_error_code": "OK",
            "prolog": {"enabled": True, "proof": "proof(step1)."},
        },
        {
            "correctness": True,
            "route": "fallback",
            "error_code": "SELF_GUIDE_SWIPL_EXCEPTION",
            "config_hash": "h1",
            "prolog_used": False,
            "prolog_ok": False,
            "solution_count": None,
            "proof_nonempty": False,
            "prolog_overruled": False,
            "final_modified_by_prolog": False,
            "prolog_inconclusive": True,
            "multi_solution_conflict": False,
            "prolog_error_code": "SELF_GUIDE_SWIPL_EXCEPTION",
            "prolog": {"enabled": True},
        },
    ]
    for idx, payload in enumerate(logs):
        (tmp_path / f"gsm8k_{idx}.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_logs(tmp_path)

    assert summary["N"] == 3
    assert summary["route_distribution"] == {"executor": 1, "verifier": 1, "llm_only": 1}
    assert summary["route_ratio"] == {
        "executor": 1 / 3,
        "verifier": 1 / 3,
        "llm_only": 1 / 3,
    }
    assert summary["prolog_ok"] == 2
    assert summary["prolog_ok_rate"] == 2 / 3
    assert summary["prolog_overrule_rate"] == 2 / 3
    assert summary["inconclusive_rate"] == 2 / 3
    assert summary["multi_solution_conflict_rate"] == 1 / 3
    assert summary["solution_count"] == {"missing": 1, "one": 1, "multi": 1}
    assert summary["proof_shape_ok"] == 2
    assert summary["proof_nonempty"] == 2