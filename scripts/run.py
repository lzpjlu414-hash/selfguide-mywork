"""Deprecated shim for the old scripts/run.py entrypoint.
This file intentionally redirects all execution to src.run_experiment.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.run_experiment import main as run_experiment_main


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DEPRECATED: use src/run_experiment.py")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--method", default="cot_selfguide")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--mock_profile", default=None)
    parser.add_argument("--mock_prolog", action="store_true")
    parser.add_argument("--force_task_type", choices=("Yes", "No", "Partial"), default=None)
    parser.add_argument("--meta_interpreter", default="iter_deep_with_proof")
    parser.add_argument("--max_depth", type=int, default=25)
    parser.add_argument("--prolog_max_result", type=int, default=20)
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    warnings.warn(
        "scripts/run.py is deprecated and will be removed. "
        "Please use: python -m src.run_experiment ...",
        UserWarning,
        stacklevel=2,
    )


    parser = _legacy_parser()
    args = parser.parse_args(argv)

    forward_argv = [
        "--dataset", args.dataset,
        "--method", args.method,
        "--start_index", str(args.start_index),
        "--num_samples", str(args.num_samples),
        "--meta_interpreter", args.meta_interpreter,
        "--max_depth", str(args.max_depth),
        "--prolog_max_result", str(args.prolog_max_result),
    ]
    if args.data_path:
        forward_argv.extend(["--data_path", args.data_path])
    if args.log_dir:
        forward_argv.extend(["--log_dir", args.log_dir])
    if args.mock_profile:
        forward_argv.extend(["--mock_profile", args.mock_profile])
    if args.tmp_dir:
        forward_argv.extend(["--tmp_dir", args.tmp_dir])
    if args.force_task_type:
        forward_argv.extend(["--force_task_type", args.force_task_type])
    if args.mock_llm:
        forward_argv.append("--mock_llm")
    if args.mock_prolog:
        forward_argv.append("--mock_prolog")
    if args.keep_tmp:
        forward_argv.append("--keep_tmp")
    if args.summarize:
        forward_argv.append("--summarize")

    run_experiment_main(forward_argv)


if __name__ == "__main__":
    main(sys.argv[1:])