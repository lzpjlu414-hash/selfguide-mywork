import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from typing import Any

from src.summarize_logs import summarize_logs


def _iter_variant_dirs(matrix_dir: Path) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    for variant_dir in sorted(p for p in matrix_dir.iterdir() if p.is_dir()):
        run_dirs = [p for p in variant_dir.iterdir() if p.is_dir()]
        if not run_dirs:
            continue
        latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
        variants.append((variant_dir.name, latest))
    return variants


def _flatten_route_distribution(route_distribution: dict[str, Any]) -> str:
    if not route_distribution:
        return ""
    parts = [f"{k}:{route_distribution[k]}" for k in sorted(route_distribution.keys())]
    return "|".join(parts)


def summarize_matrix(matrix_out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "matrix_dir": str(matrix_out_dir),
        "variants": {},
    }

    for variant_tag, run_dir in _iter_variant_dirs(matrix_out_dir):
        metrics = summarize_logs(run_dir)
        row = {
            "variant": variant_tag,
            "run_dir": str(run_dir),
            "N": metrics["N"],
            "accuracy": metrics["accuracy"],
            "route_distribution": _flatten_route_distribution(metrics["route_distribution"]),
            "fallback_rate": metrics["route_distribution"].get("fallback", 0) / metrics["N"] if metrics["N"] else 0.0,
            "schema_error_rate": (metrics["error_code_distribution"].get("SCHEMA_ERROR", 0) / metrics["N"]) if metrics["N"] else 0.0,
            "copy_rate": metrics["copy_rate"],
            "copy_and_wrong_rate": metrics["copy_and_wrong_rate"],
            "corrected_rate": metrics["corrected_rate"],
            "prolog_overrule_rate": metrics["prolog_overrule_rate"],
        }
        rows.append(row)
        summary["variants"][variant_tag] = {
            "run_dir": str(run_dir),
            **metrics,
            "fallback_rate": row["fallback_rate"],
            "schema_error_rate": row["schema_error_rate"],
        }

    return rows, summary


def write_outputs(matrix_out_dir: Path) -> tuple[Path, Path]:
    rows, summary = summarize_matrix(matrix_out_dir)

    csv_path = matrix_out_dir / "results.csv"
    json_path = matrix_out_dir / "summary.json"

    fieldnames = [
        "variant",
        "run_dir",
        "N",
        "accuracy",
        "route_distribution",
        "fallback_rate",
        "schema_error_rate",
        "copy_rate",
        "copy_and_wrong_rate",
        "corrected_rate",
        "prolog_overrule_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize matrix outputs to results.csv + summary.json")
    parser.add_argument("--matrix_out", required=True, help="Matrix output directory, e.g. runs/mock")
    args = parser.parse_args()

    matrix_out_dir = Path(args.matrix_out)
    if not matrix_out_dir.exists() or not matrix_out_dir.is_dir():
        raise FileNotFoundError(f"matrix_out not found: {matrix_out_dir}")

    csv_path, json_path = write_outputs(matrix_out_dir)
    print(f"[summarize_runs] wrote {csv_path}")
    print(f"[summarize_runs] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())