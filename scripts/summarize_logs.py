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
    logs = sorted(log_dir.glob("*.json"))
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
        }

    correct = 0
    correctness_missing = 0
    route_counts = Counter()
    prolog_enabled = 0
    prolog_missing = 0
    proof_nonempty = 0
    prolog_swipl_ok = 0

    for path in logs:
        data = json.loads(path.read_text(encoding="utf-8"))
        correctness = to_bool(data.get("correctness"))
        if correctness is None:
            correctness_missing += 1
        elif correctness:
            correct += 1

        route = normalize_route(data.get("route"))
        route_counts[route] += 1

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


if __name__ == "__main__":
    main()