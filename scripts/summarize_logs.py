import argparse
from pathlib import Path
from src.summarize_logs import summarize_logs



def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Self-Guide JSON logs.")
    parser.add_argument("--log_dir", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    summary = summarize_logs(log_dir)
    print(f"N={summary['N']}")
    print(f"accuracy={summary['accuracy']:.4f}")
    print(f"route_distribution={summary['route_distribution']}")
    print(f"route_ratio={summary['route_ratio']}")
    print(f"prolog_enabled={summary['prolog_enabled']}")
    print(f"proof_shape_ok={summary['proof_shape_ok']} (recommended)")
    print(f"proof_nonempty={summary['proof_nonempty']} (legacy)")
    print(f"prolog_ok={summary['prolog_ok']}")
    print(f"solution_count={summary['solution_count']}")
    print(f"correctness_missing={summary['correctness_missing']}")
    print(f"prolog_missing={summary['prolog_missing']}")
    print(f"prolog_swipl_ok={summary['prolog_swipl_ok']}")
    print(f"error_code_distribution={summary['error_code_distribution']}")
    print(f"prolog_error_code_distribution={summary['prolog_error_code_distribution']}")
    print(f"schema_version_distribution={summary['schema_version_distribution']}")
    print(f"legacy_schema_hits={summary['legacy_schema_hits']}")
    print(f"legacy_schema_hit_rate={summary['legacy_schema_hit_rate']:.4f}")
    print(f"config_hash_distribution={summary['config_hash_distribution']}")
    print(f"copy_rate={summary['copy_rate']:.4f}")
    print(f"copy_and_wrong_rate={summary['copy_and_wrong_rate']:.4f}")
    print(f"corrected_rate={summary['corrected_rate']:.4f}")
    print(f"prolog_overrule_rate={summary['prolog_overrule_rate']:.4f}")
    print(f"prolog_ok_rate={summary['prolog_ok_rate']:.4f}")
    print(f"inconclusive_rate={summary['inconclusive_rate']:.4f}")
    print(f"multi_solution_conflict_rate={summary['multi_solution_conflict_rate']:.4f}")
    print(f"guideline_schema_valid_rate={summary['guideline_schema_valid_rate']:.4f}")
    print(f"guideline_retry_avg={summary['guideline_retry_avg']:.4f}")


if __name__ == "__main__":
    main()