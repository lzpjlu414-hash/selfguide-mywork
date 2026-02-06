import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abs.self_guide_myself import (
    NO_PROOF_RETURNED_SENTINEL,
    is_nonempty_answer,
    is_nonempty_proof,
    resolve_solution_count,
    self_guide_run,
)


def _run_once(log_dir: Path, role: str) -> dict:
    self_guide_run(
        dataset="gsm8k",
        method="cot_selfguide",
        start_index=0,
        num_samples=1,
        force_task_type="Yes",
        log_dir_override=str(log_dir),
        mock_llm=True,
        mock_profile="prolog-role-test",
        mock_prolog=True,
        prolog_role=role,
    )
    log_file = next(log_dir.glob("gsm8k_*.json"))
    return json.loads(log_file.read_text(encoding="utf-8"))


def test_verifier_executor_roles_produce_different_routes(tmp_path: Path) -> None:
    verifier_payload = _run_once(tmp_path / "verifier", "verifier")
    executor_payload = _run_once(tmp_path / "executor", "executor")

    assert verifier_payload["route"] == "verifier"
    assert verifier_payload["final_answer"] == verifier_payload["prolog_answer"]
    assert verifier_payload["prolog_overruled"] is True
    assert verifier_payload["final_modified_by_prolog"] is True
    assert executor_payload["route"] == "executor"


def test_executor_consistency_constraints(tmp_path: Path) -> None:
    payload = _run_once(tmp_path / "executor", "executor")

    assert payload["route"] == "executor"
    assert payload["final_modified_by_prolog"] is True
    assert payload["final_answer"] == payload["prolog_answer"]

def _run_verifier_with_mock_payload(tmp_path: Path, monkeypatch, payload: dict) -> dict:
        from src.abs import self_guide_myself as mod

        def _fake_mock_swipl_output(dataset_key: str, llm_candidate_norm: str) -> dict:
            _ = dataset_key
            return {
                "ok": True,
                "error": None,
                "error_code": None,
                "raw": json.dumps(payload),
                "stdout": "",
                "stderr": "",
                "returncode": 0,
                "cmd": "mock_swipl",
            }

        monkeypatch.setattr(mod, "build_mock_swipl_output", _fake_mock_swipl_output)
        monkeypatch.setenv("SLEEP_SEC", "0")

        self_guide_run(
            dataset="gsm8k",
            method="cot_selfguide",
            start_index=0,
            num_samples=1,
            force_task_type="Yes",
            log_dir_override=str(tmp_path),
            mock_llm=True,
            mock_profile="prolog-role-test",
            mock_prolog=True,
            prolog_role="verifier",
        )
        log_file = next(tmp_path.glob("gsm8k_*.json"))
        return json.loads(log_file.read_text(encoding="utf-8"))

def test_verifier_conflict_without_proof_and_multi_solution_keeps_llm(tmp_path: Path, monkeypatch) -> None:
        payload = _run_verifier_with_mock_payload(
            tmp_path,
            monkeypatch,
            {
                "schema_version": "1.0",
                "ok": True,
                "answer": "999",
                "proof": "",
                "error_code": None,
                "solution_count": 2,
                "raw": {"results_count": 2},
            },
        )

        assert payload["route"] == "verifier"
        assert payload["prolog"]["verifier_gate"] == "multi_solution_conflict"
        assert payload["final_answer"] == payload["llm_candidate_norm"]
        assert payload["final_answer"] != payload["prolog_answer"]

def test_verifier_conflict_with_proof_overrides_to_prolog(tmp_path: Path, monkeypatch) -> None:
        payload = _run_verifier_with_mock_payload(
            tmp_path,
            monkeypatch,
            {
                "schema_version": "1.0",
                "ok": True,
                "answer": "999",
                "proof": "proof(step1).",
                "error_code": None,
                "solution_count": 2,
                "raw": {"results_count": 2},
            },
        )

        assert payload["route"] == "verifier"
        assert payload["prolog"]["verifier_gate"] == "override"
        assert payload["final_answer"] == payload["prolog_answer"]

def test_answer_nonempty_filters_placeholders() -> None:
    assert is_nonempty_answer(" 42 ") is True
    assert is_nonempty_answer("  unknown ") is False
    assert is_nonempty_answer(" N/A ") is False


def test_proof_nonempty_blocks_no_proof_sentinel() -> None:
    assert is_nonempty_proof("proof(step1).") is True
    assert is_nonempty_proof(NO_PROOF_RETURNED_SENTINEL) is False


def test_solution_count_resolution_and_invalid_types() -> None:
    count, valid = resolve_solution_count(
        {
            "solution_count": None,
            "raw": {
                "results_count": None,
                "results": ["a", "b"],
            },
        }
    )
    assert (count, valid) == (2, True)

    missing_count, missing_valid = resolve_solution_count({"raw": {}})
    assert (missing_count, missing_valid) == (None, True)

    invalid_count, invalid_valid = resolve_solution_count({"solution_count": "2", "raw": {"results": []}})
    assert (invalid_count, invalid_valid) == (None, False)


def test_verifier_invalid_solution_count_keeps_llm_as_inconclusive(tmp_path: Path, monkeypatch) -> None:
    payload = _run_verifier_with_mock_payload(
        tmp_path,
        monkeypatch,
        {
            "schema_version": "1.0",
            "ok": True,
            "answer": "999",
            "proof": "",
            "error_code": None,
            "solution_count": "invalid",
            "raw": {"results_count": 2},
        },
    )

    assert payload["route"] == "verifier"
    assert payload["prolog"]["solution_count_valid"] is False
    assert payload["prolog"]["verifier_gate"] == "prolog_inconclusive"
    assert payload["final_answer"] == payload["llm_candidate_norm"]
