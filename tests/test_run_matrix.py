import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner_matrix import _prepare_run_dir, canonical_config_hash


def test_canonical_config_hash_stable() -> None:
    cfg1 = {"dataset": "gsm8k", "method": "cot_selfguide", "start_index": 0, "num_samples": 1}
    cfg2 = {"num_samples": 1, "start_index": 0, "method": "cot_selfguide", "dataset": "gsm8k"}
    assert canonical_config_hash(cfg1) == canonical_config_hash(cfg2)


def test_prepare_run_dir_structure(tmp_path: Path) -> None:
    run_dir = _prepare_run_dir(tmp_path / "runs", "real", "abc_no_prolog", "20260101-010101")
    assert run_dir.exists()
    assert run_dir.as_posix().endswith("runs/real/abc_no_prolog/20260101-010101")