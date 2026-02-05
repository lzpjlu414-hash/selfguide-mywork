import json
from pathlib import Path

from scripts.summarize_runs import summarize_matrix, write_outputs


def test_summarize_matrix_and_write_outputs(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "mock"
    run_dir = matrix_dir / "baseline_single_round" / "20260101-010101"
    run_dir.mkdir(parents=True)

    sample_log = {
        "correctness": "True",
        "route": "llm_only",
        "error_code": "OK",
        "config_hash": "abc",
        "draft_final_same": True,
        "final_correct": True,
        "draft_correct": False,
        "draft_prolog_conflict": False,
        "prolog": {"enabled": False},
    }
    (run_dir / "gsm8k_0.json").write_text(json.dumps(sample_log), encoding="utf-8")

    rows, summary = summarize_matrix(matrix_dir)
    assert len(rows) == 1
    assert rows[0]["variant"] == "baseline_single_round"
    assert rows[0]["accuracy"] == 1.0
    assert "baseline_single_round" in summary["variants"]

    csv_path, json_path = write_outputs(matrix_dir)
    assert csv_path.exists()
    assert json_path.exists()
    assert "baseline_single_round" in json.loads(json_path.read_text(encoding="utf-8"))["variants"]

def test_summarize_matrix_prefers_most_recent_run_dir(tmp_path: Path) -> None:
        matrix_dir = tmp_path / "mock"
        older = matrix_dir / "abc_prolog_verifier" / "99999999-999999"
        newer = matrix_dir / "abc_prolog_verifier" / "20240101-000000"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)

        older_log = {"correctness": True, "route": "executor", "error_code": "OK", "config_hash": "h1",
                     "prolog": {"enabled": True}}
        newer_log = {"correctness": True, "route": "verifier", "error_code": "OK", "config_hash": "h2",
                     "prolog": {"enabled": True}}
        (older / "gsm8k_0.json").write_text(json.dumps(older_log), encoding="utf-8")
        (newer / "gsm8k_0.json").write_text(json.dumps(newer_log), encoding="utf-8")

        import os, time
        now = time.time()
        os.utime(older, (now - 1000, now - 1000))
        os.utime(newer, (now, now))

        rows, _ = summarize_matrix(matrix_dir)
        assert rows[0]["route_distribution"] == "verifier:1"