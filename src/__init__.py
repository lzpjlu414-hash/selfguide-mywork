"""Compatibility package for running `python -m src.run_experiment` from ./src."""

from pathlib import Path

_parent_src_dir = Path(__file__).resolve().parent.parent
_parent_src_dir_str = str(_parent_src_dir)

if _parent_src_dir_str not in __path__:
    __path__.append(_parent_src_dir_str)