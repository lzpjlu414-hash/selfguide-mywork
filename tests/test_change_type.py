import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import compute_change_type, normalize_answer


def test_normalize_answer_basic() -> None:
    assert normalize_answer(" Final Answer: 42. ") == "42"


def test_compute_change_type_copy() -> None:
    assert compute_change_type("Final Answer: 42", "42.", prolog_used=False, prolog_overruled=False) == "copy"


def test_compute_change_type_revise() -> None:
    assert compute_change_type("41", "42", prolog_used=False, prolog_overruled=False) == "revise"


def test_compute_change_type_corrected_by_prolog() -> None:
    assert compute_change_type("41", "42", prolog_used=True, prolog_overruled=True) == "corrected_by_prolog"