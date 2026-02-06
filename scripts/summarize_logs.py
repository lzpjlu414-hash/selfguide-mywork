import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional


TRUTHY = {"true", "t", "1", "yes", "y"}
FALSY = {"false", "f", "0", "no", "n"}


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUTHY:
            return True
        if lowered in FALSY:
            return False
    return None


def normalize_route(route: Any) -> str:
    if route is None:
        return "unknown"
    if not isinstance(route, str):
        route = str(route)
    route = route.strip()
    return route or "unknown"


def summarize_logs(log_dir: Path) -> dict:
    logs = sorted(p for p in log_dir.glob("*.json") if p.name != "run_config.json")
    total = len(logs)
    if total == 0:
        return {
            "N": 0,
            "accuracy": 0.0,
            "route_distribution": {},
            "prolog_enabled": 0,
            "proof_nonempty": 0,
            "correctness_missing": 0,
            "prolog_missing": 0,
            "prolog_swipl_ok": 0,
            "error_code_distribution": {},
            "schema_version_distribution": {},
            "legacy_schema_hits": 0,
            "legacy_schema_hit_rate": 0.0,
            "config_hash_distribution": {},
            "copy_rate": 0.0,
            "copy_and_wrong_rate": 0.0,
            "corrected_rate": 0.0,
            "prolog_overrule_rate": 0.0,
        }

    correct = 0
    correctness_missing = 0
    route_counts = Counter()
    prolog_enabled = 0
    prolog_missing = 0
    proof_nonempty = 0
    prolog_swipl_ok = 0
    error_code_counts = Counter()
    schema_version_counts = Counter()
    config_hash_counts = Counter()
    legacy_schema_hits = 0

    copy_count = 0
    copy_and_wrong_count = 0
    corrected_count = 0
    prolog_overrule_count = 0
    guideline_schema_valid_count = 0
    guideline_retry_total = 0

    for path in logs:
        data = json.loads(path.read_text(encoding="utf-8"))
        correctness = to_bool(data.get("correctness"))
        if correctness is None:
            correctness_missing += 1
        elif correctness:
            correct += 1

        route = normalize_route(data.get("route"))
        route_counts[route] += 1

        error_code = str(data.get("error_code") or "UNKNOWN")
        error_code_counts[error_code] += 1

        config_hash = str(data.get("config_hash") or "UNKNOWN")
        config_hash_counts[config_hash] += 1

        prolog = data.get("prolog")
        if not isinstance(prolog, dict):
            prolog = {}
            prolog_missing += 1

        enabled = to_bool(prolog.get("enabled"))
        if enabled:
            prolog_enabled += 1

        proof = prolog.get("proof")
        if isinstance(proof, str) and proof.strip():
            proof_nonempty += 1

        swipl = prolog.get("swipl")
        if isinstance(swipl, dict) and to_bool(swipl.get("ok")):
            prolog_swipl_ok += 1

        swipl_contract = prolog.get("swipl_contract") if isinstance(prolog, dict) else None
        if isinstance(swipl_contract, dict):
            schema_version = str(swipl_contract.get("schema_version") or "UNKNOWN")
            schema_version_counts[schema_version] += 1
            if to_bool(swipl_contract.get("legacy")):
                legacy_schema_hits += 1
        draft_final_same = to_bool(data.get("draft_final_same"))
        if draft_final_same:
            copy_count += 1

        final_correct = to_bool(data.get("final_correct"))
        if draft_final_same and final_correct is False:
            copy_and_wrong_count += 1

        draft_correct = to_bool(data.get("draft_correct"))
        if draft_correct is False and final_correct is True:
            corrected_count += 1

        draft_prolog_conflict = to_bool(data.get("draft_prolog_conflict"))
        if draft_prolog_conflict is None:
            draft_prolog_conflict = to_bool(data.get("prolog_overruled"))
        if draft_prolog_conflict:
            prolog_overrule_count += 1

        guideline_schema_valid = to_bool(data.get("guideline_schema_valid"))
        if guideline_schema_valid:
            guideline_schema_valid_count += 1

        retry_count = data.get("guideline_retry_count", 0)
        if isinstance(retry_count, int) and retry_count >= 0:
            guideline_retry_total += retry_count

    accuracy = correct / total
    return {
        "N": total,
        "accuracy": accuracy,
        "route_distribution": dict(route_counts),
        "prolog_enabled": prolog_enabled,
        "proof_nonempty": proof_nonempty,
        "correctness_missing": correctness_missing,
        "prolog_missing": prolog_missing,
        "prolog_swipl_ok": prolog_swipl_ok,
        "error_code_distribution": dict(error_code_counts),
        "schema_version_distribution": dict(schema_version_counts),
        "legacy_schema_hits": legacy_schema_hits,
        "legacy_schema_hit_rate": (legacy_schema_hits / total),
        "config_hash_distribution": dict(config_hash_counts),
        "copy_rate": (copy_count / total),
        "copy_and_wrong_rate": (copy_and_wrong_count / total),
        "corrected_rate": (corrected_count / total),
        "prolog_overrule_rate": (prolog_overrule_count / total),
        "guideline_schema_valid_rate": 0.0,
        "guideline_retry_avg": 0.0,
    }



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
    print(f"copy_rate={summary['copy_rate']:.4f}")
    print(f"copy_and_wrong_rate={summary['copy_and_wrong_rate']:.4f}")
    print(f"corrected_rate={summary['corrected_rate']:.4f}")
    print(f"prolog_overrule_rate={summary['prolog_overrule_rate']:.4f}")
    print(f"guideline_schema_valid_rate={summary['guideline_schema_valid_rate']:.4f}")
    print(f"guideline_retry_avg={summary['guideline_retry_avg']:.4f}")


if __name__ == "__main__":
    main()