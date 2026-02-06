import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_logs import summarize_logs
from src.abs.self_guide_myself import self_guide_run


def test_mock_prolog_smoke_reaches_non_llm_route_and_summary_marks_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLEEP_SEC", "0")

    self_guide_run(
        dataset="gsm8k",
        method="cot_selfguide",
        start_index=0,
        num_samples=2,
        force_task_type="Yes",
        log_dir_override=str(tmp_path),
        mock_llm=True,
        mock_profile="prolog-role-test",
        mock_prolog=True,
        prolog_role="verifier",
    )

    logs = sorted(tmp_path.glob("gsm8k_*.json"))
    assert logs, "expected at least one sample log"

    summary = summarize_logs(tmp_path)

    assert summary["prolog_enabled"] != 0
    assert any(route != "llm_only" for route in summary["route_distribution"])

    # Redundant guard to make smoke intent explicit at sample level too.
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in logs]
    assert any(str(p.get("route", "")).strip() in {"verifier", "executor"} for p in payloads)

    for payload in payloads:
        solution_count = payload.get("solution_count")
        assert isinstance(solution_count, int)
        assert solution_count >= 1

        prolog_error_code = payload.get("prolog_error_code")
        assert isinstance(prolog_error_code, str)
        assert prolog_error_code.strip()
        assert prolog_error_code == "OK"

    assert summary["solution_count"]["missing"] == 0
    assert summary["prolog_error_code_distribution"].get("OK", 0) == len(payloads)

