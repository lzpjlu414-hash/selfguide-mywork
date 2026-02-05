import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.abs.self_guide_myself import self_guide_run


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
    assert executor_payload["route"] == "executor"


def test_executor_consistency_constraints(tmp_path: Path) -> None:
    payload = _run_once(tmp_path / "executor", "executor")

    assert payload["route"] == "executor"
    assert payload["final_modified_by_prolog"] is True
    assert payload["final_answer"] == payload["prolog_answer"]