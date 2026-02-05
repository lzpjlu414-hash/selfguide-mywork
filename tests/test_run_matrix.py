import sys
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner_matrix import (
    _normalize_force_task_type,
    _prepare_run_dir,
    canonical_config_hash,
    main,
)


def test_canonical_config_hash_stable() -> None:
    cfg1 = {"dataset": "gsm8k", "method": "cot_selfguide", "start_index": 0, "num_samples": 1}
    cfg2 = {"num_samples": 1, "start_index": 0, "method": "cot_selfguide", "dataset": "gsm8k"}
    assert canonical_config_hash(cfg1) == canonical_config_hash(cfg2)


def test_prepare_run_dir_structure(tmp_path: Path) -> None:
    run_dir = _prepare_run_dir(tmp_path / "runs", "real", "abc_no_prolog", "20260101-010101")
    assert run_dir.exists()
    assert run_dir.as_posix().endswith("runs/real/abc_no_prolog/20260101-010101")

    def test_normalize_force_task_type_bool_mapping() -> None:
        assert _normalize_force_task_type(True) == "Yes"
        assert _normalize_force_task_type(False) == "No"

    def test_normalize_force_task_type_invalid_string() -> None:
        with pytest.raises(ValueError, match="Invalid force_task_type"):
            _normalize_force_task_type("False")

    def test_main_reports_invalid_force_task_type(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config = {
            "matrix_name": "invalid",
            "output_root": str(tmp_path / "runs"),
            "continue_on_error": True,
            "base_experiment": {
                "dataset": "gsm8k",
                "method": "cot_selfguide",
                "mock_llm": True,
                "mock_prolog": True,
            },
            "variants": [
                {
                    "tag": "bad_variant",
                    "overrides": {"force_task_type": "False"},
                }
            ],
        }
        config_path = tmp_path / "bad_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        code = main(["--config", str(config_path)])

        out = capsys.readouterr()
        assert code == 1
        assert "Invalid force_task_type" in out.err
        assert "Allowed values are: Yes, No, Partial" in out.err