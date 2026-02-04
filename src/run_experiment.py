import os
import sys
from argparse import ArgumentParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.dataset_io import resolve_data_path, validate_openai_api_key

from src.abs.self_guide_myself import self_guide_run


def main() -> None:
    parser = ArgumentParser(description="Run Self-Guide experiments across datasets.")
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=["gsm8k", "prontoqa", "proofwriter", "mmlu", "sqa", "date", "clutrr"],
    )
    parser.add_argument("--method", required=True, choices=["cot_selfguide", "sd_selfguide"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--force_task_type", choices=("Yes", "No", "Partial"), default=None)
    parser.add_argument("--meta_interpreter", default="iter_deep_with_proof")
    parser.add_argument("--max_depth", type=int, default=25)
    parser.add_argument("--prolog_max_result", type=int, default=20)

    args = parser.parse_args()

    validate_openai_api_key(args.mock_llm)
    try:
        data_path = resolve_data_path(args.dataset, args.data_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    self_guide_run(
        dataset=args.dataset,
        method=args.method,
        start_index=args.start_index,
        num_samples=args.num_samples,
        force_task_type=args.force_task_type,
        data_path=data_path,
        log_dir_override=args.log_dir,
        mock_llm=args.mock_llm,
        meta_interpreter=args.meta_interpreter,
        max_depth=args.max_depth,
        prolog_max_result=args.prolog_max_result,
    )


if __name__ == "__main__":
    main()