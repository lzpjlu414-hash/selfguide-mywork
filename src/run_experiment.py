import sys
from pathlib import Path
from typing import Optional

from src.abs.self_guide_myself import self_guide_run
from src.experiment_contract import (
    build_experiment_parser,
    resolve_log_dir,
    validate_contract,
)
from src.utils.dataset_io import resolve_data_path, validate_openai_api_key

def _normalize_prolog_controls(args) -> None:
    """Make mock Prolog runs deterministically exercise the Prolog branch."""
    if not args.mock_prolog:
        return
    if args.prolog_role == "off":
        args.prolog_role = "verifier"
    if args.force_task_type is None:
        args.force_task_type = "Yes"

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_experiment_parser()
    args = parser.parse_args(argv)
    _normalize_prolog_controls(args)



    try:
        validate_contract(args)
    except ValueError as exc:
        print(f"Invalid experiment args: {exc}", file=sys.stderr)
        sys.exit(2)

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
        prolog_role=args.prolog_role,
        meta_interpreter=args.meta_interpreter,
        max_depth=args.max_depth,
        prolog_max_result=args.prolog_max_result,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp,
    )
    if args.summarize:
        from src.summarize_logs import summarize_logs
        log_dir: Path = resolve_log_dir(args)
        summary = summarize_logs(log_dir)
        print(f"N={summary['N']}")
        print(f"accuracy={summary['accuracy']:.4f}")
        print(f"route_distribution={summary['route_distribution']}")
        print(f"prolog_enabled={summary['prolog_enabled']}")
        print(f"proof_nonempty={summary['proof_nonempty']}")
        print(f"correctness_missing={summary['correctness_missing']}")
        print(f"prolog_missing={summary['prolog_missing']}")
        print(f"prolog_swipl_ok={summary['prolog_swipl_ok']}")
        print(f"error_code_distribution={summary['error_code_distribution']}")
        print(f"schema_version_distribution={summary['schema_version_distribution']}")
        print(f"legacy_schema_hits={summary['legacy_schema_hits']}")
        print(f"legacy_schema_hit_rate={summary['legacy_schema_hit_rate']:.4f}")
        print(f"config_hash_distribution={summary['config_hash_distribution']}")


if __name__ == "__main__":
    main()