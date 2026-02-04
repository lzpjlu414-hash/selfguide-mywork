import sys
from argparse import ArgumentParser
from pathlib import Path

from src.abs.self_guide_myself import self_guide_run
from src.utils.dataset_io import resolve_data_path, validate_openai_api_key



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
    parser.add_argument("--mock_profile", default=None)
    parser.add_argument("--mock_prolog", action="store_true")
    parser.add_argument("--force_task_type", choices=("Yes", "No", "Partial"), default=None)
    parser.add_argument("--meta_interpreter", default="iter_deep_with_proof")
    parser.add_argument("--max_depth", type=int, default=25)
    parser.add_argument("--prolog_max_result", type=int, default=20)
    parser.add_argument("--tmp_dir", default=None, help="root dir for Prolog temp files")
    parser.add_argument("--keep_tmp", action="store_true", help="keep Prolog temp files")
    parser.add_argument("--summarize", action="store_true", help="summarize logs after run")

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
        mock_profile=args.mock_profile,
        mock_prolog=args.mock_prolog,
        meta_interpreter=args.meta_interpreter,
        max_depth=args.max_depth,
        prolog_max_result=args.prolog_max_result,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp,
    )
    if args.summarize:
        from src.summarize_logs import summarize_logs
        log_dir = Path(args.log_dir) if args.log_dir else Path(f"log/{args.method}/{args.dataset}")
        summary = summarize_logs(log_dir)
        print(f"N={summary['N']}")
        print(f"accuracy={summary['accuracy']:.4f}")
        print(f"route_distribution={summary['route_distribution']}")
        print(f"prolog_enabled={summary['prolog_enabled']}")
        print(f"proof_nonempty={summary['proof_nonempty']}")
        print(f"correctness_missing={summary['correctness_missing']}")
        print(f"prolog_missing={summary['prolog_missing']}")
        print(f"prolog_swipl_ok={summary['prolog_swipl_ok']}")


if __name__ == "__main__":
    main()