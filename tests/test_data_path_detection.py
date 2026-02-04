import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.dataset_io import resolve_data_path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        _touch(base_dir / "data" / "gsm8k" / "test_small.jsonl")
        _touch(base_dir / "log" / "gsm8k.jsonl")

        resolved = resolve_data_path("gsm8k", base_dir=base_dir)
        assert resolved.endswith("data/gsm8k/test_small.jsonl"), resolved

        _touch(base_dir / "data" / "gsm8k" / "test.jsonl")
        resolved = resolve_data_path("gsm8k", base_dir=base_dir)
        assert resolved.endswith("data/gsm8k/test.jsonl"), resolved

        explicit = base_dir / "custom.jsonl"
        _touch(explicit)
        resolved = resolve_data_path("gsm8k", data_path=str(explicit), base_dir=base_dir)
        assert resolved == str(explicit), resolved


if __name__ == "__main__":
    main()