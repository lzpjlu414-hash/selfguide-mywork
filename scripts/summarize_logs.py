import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    logs = sorted(log_dir.glob("*.json"))
    total = len(logs)
    if total == 0:
        print("N=0")
        return

    correct = 0
    route_counts = {}
    prolog_enabled = 0
    proof_nonempty = 0

    for path in logs:
        data = json.loads(path.read_text(encoding="utf-8"))
        correctness = data.get("correctness") == "True"
        correct += 1 if correctness else 0
        route = data.get("route", "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1

        prolog = data.get("prolog", {}) or {}
        if prolog.get("enabled"):
            prolog_enabled += 1
            proof = prolog.get("proof", "")
            if isinstance(proof, str) and proof.strip():
                proof_nonempty += 1

    accuracy = correct / total
    print(f"N={total}")
    print(f"accuracy={accuracy:.4f}")
    print(f"route_distribution={route_counts}")
    print(f"prolog_enabled={prolog_enabled}")
    print(f"proof_nonempty={proof_nonempty}")


if __name__ == "__main__":
    main()