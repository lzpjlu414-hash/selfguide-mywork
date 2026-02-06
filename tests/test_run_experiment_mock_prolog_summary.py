import ast
import json
from pathlib import Path

from src.run_experiment import main


def _line_value(stdout: str, key: str) -> str:
    prefix = f"{key}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"missing line for {key}:\n{stdout}")


def test_run_experiment_summary_with_mock_prolog_forces_prolog_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("SLEEP_SEC", "0")

    main(
        [
            "--dataset",
            "gsm8k",
            "--method",
            "cot_selfguide",
            "--num_samples",
            "2",
            "--mock_llm",
            "--mock_prolog",
            "--log_dir",
            str(tmp_path),
            "--summarize",
        ]
    )

    captured = capsys.readouterr().out
    assert _line_value(captured, "prolog_enabled") == "2"

    route_distribution = ast.literal_eval(_line_value(captured, "route_distribution"))
    assert any(route in {"verifier", "executor"} for route in route_distribution)
    assert route_distribution.get("llm_only", 0) < 2

    schema_distribution = ast.literal_eval(_line_value(captured, "schema_version_distribution"))
    assert schema_distribution

    sample_logs = sorted(tmp_path.glob("gsm8k_*.json"))
    assert sample_logs
    payload = json.loads(sample_logs[0].read_text(encoding="utf-8"))
    assert payload["prolog"]["enabled"] is True