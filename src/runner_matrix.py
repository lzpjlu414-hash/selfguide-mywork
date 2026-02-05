from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.common.config import stable_json_dumps
from src.run_experiment import main as run_single_experiment
from src.utils.dataset_io import load_dataset, resolve_data_path

logger = logging.getLogger(__name__)

VALID_FORCE_TASK_TYPES = ("Yes", "No", "Partial")


@dataclass(frozen=True)
class MatrixVariant:
    tag: str
    description: str
    overrides: dict[str, Any]


DEFAULT_VARIANTS: tuple[MatrixVariant, ...] = (
    MatrixVariant(
        tag="baseline_single_round",
        description="baseline（单轮）",
        overrides={},
    ),
    MatrixVariant(
        tag="abc_no_prolog",
        description="A+B+C（无 Prolog）",
        overrides={"force_task_type": "No"},
    ),
    MatrixVariant(
        tag="abc_prolog_verifier",
        description="A+B+C + Prolog Verifier",
        overrides={"force_task_type": "Yes", "prolog_max_result": 1},
    ),
    MatrixVariant(
        tag="abc_prolog_executor",
        description="A+B+C + Prolog Executor（可限定子集）",
        overrides={"force_task_type": "Yes"},
    ),
)


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "YAML config requires PyYAML, or provide JSON-formatted text in .yaml file.") from exc
            payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError("Matrix config must be a JSON/YAML object.")
    return payload


def load_matrix_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = _load_json_or_yaml(path)

    payload.setdefault("matrix_name", path.stem)
    payload.setdefault("output_root", "runs")
    payload.setdefault("continue_on_error", True)
    payload.setdefault("parallelism", 1)
    payload.setdefault("base_experiment", {})
    if not isinstance(payload["base_experiment"], dict):
        raise ValueError("base_experiment must be an object")
    return payload


def canonical_config_hash(config: dict[str, Any]) -> str:
    normalized = stable_json_dumps(config)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _build_variants(config: dict[str, Any]) -> list[MatrixVariant]:
    configured = config.get("variants")
    if not configured:
        return list(DEFAULT_VARIANTS)

    variants: list[MatrixVariant] = []
    for raw in configured:
        if not isinstance(raw, dict):
            raise ValueError("Each variants item must be an object")
        tag = str(raw.get("tag") or "").strip()
        if not tag:
            raise ValueError("Variant tag is required")
        desc = str(raw.get("description") or tag)
        overrides = raw.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Variant overrides must be object: {tag}")
        variants.append(MatrixVariant(tag=tag, description=desc, overrides=overrides))
    return variants


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _build_cli_args(expanded: dict[str, Any], run_dir: Path) -> list[str]:
    num_samples = expanded.get("num_samples", expanded.get("max_samples", 1))
    args = [
        "--method",
        str(expanded["method"]),
        "--dataset",
        str(expanded["dataset"]),
        "--start_index",
        str(expanded.get("start_index", 0)),
        "--num_samples",
        str(num_samples),
        "--log_dir",
        str(run_dir),
    ]

    option_fields = (
        "data_path",
        "mock_profile",
        "force_task_type",
        "meta_interpreter",
        "max_depth",
        "prolog_max_result",
        "tmp_dir",
    )
    for key in option_fields:
        value = expanded.get(key)
        if value is None:
            continue
        args.extend([f"--{key}", str(value)])

    for flag in ("mock_llm", "mock_prolog", "keep_tmp", "summarize"):
        if expanded.get(flag):
            args.append(f"--{flag}")
    return args

def _normalize_force_task_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in VALID_FORCE_TASK_TYPES:
            return normalized
        allowed = ", ".join(VALID_FORCE_TASK_TYPES)
        raise ValueError(
            f"Invalid force_task_type: {value!r}. Allowed values are: {allowed}. "
            "Booleans are also accepted and mapped as True->Yes, False->No."
        )

    allowed = ", ".join(VALID_FORCE_TASK_TYPES)
    raise ValueError(
        f"Invalid force_task_type type: {type(value).__name__}. "
        f"Use one of: {allowed}, or booleans True/False."
    )

def _collect_predictions(run_dir: Path, expanded: dict[str, Any], config_hash: str) -> None:
    dataset = str(expanded["dataset"]).lower()
    data_path = resolve_data_path(dataset, expanded.get("data_path"))
    samples = load_dataset(data_path, dataset)

    records = []
    for log_file in sorted(run_dir.glob(f"{dataset}_*.json")):
        raw = json.loads(log_file.read_text(encoding="utf-8"))
        sample_id = raw.get("id")
        question = raw.get("question")
        if not question:
            for sample in samples:
                if sample.get("id") == sample_id:
                    question = sample.get("question")
                    break
        records.append(
            {
                "id": sample_id,
                "question": question,
                "draft_answer": raw.get("draft_answer", raw.get("draft")),
                "final_answer": raw.get("final_answer"),
                "draft_final_same": raw.get("draft_final_same"),
                "final_modified_by_prolog": raw.get("final_modified_by_prolog"),
                "draft_to_final_change_type": raw.get("draft_to_final_change_type"),
                "draft_correct": raw.get("draft_correct"),
                "final_correct": raw.get("final_correct"),
                "draft_prolog_conflict": raw.get("draft_prolog_conflict"),
                "prolog_overruled": raw.get("prolog_overruled"),
                "pred": raw.get("pred"),
                "correctness": raw.get("correctness"),
                "route": raw.get("route"),
                "config_hash": config_hash,
            }
        )

    out_path = run_dir / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_run_dir(output_root: Path, matrix_name: str, variant_tag: str, ts: str) -> Path:
    run_dir = output_root / matrix_name / variant_tag / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_matrix(
    config: dict[str, Any],
    runner: Callable[[list[str]], None] = run_single_experiment,
) -> int:
    matrix_name = str(config["matrix_name"])
    output_root = Path(config["output_root"])
    continue_on_error = bool(config.get("continue_on_error", True))
    parallelism = int(config.get("parallelism", 1))
    base_experiment = dict(config["base_experiment"])
    if "max_samples" in base_experiment and "num_samples" not in base_experiment:
        base_experiment["num_samples"] = base_experiment["max_samples"]
    variants = _build_variants(config)

    ts = _timestamp()
    failed_tags: list[str] = []

    def _run_variant(variant: MatrixVariant) -> tuple[str, int]:
        expanded = dict(base_experiment)
        expanded.update(variant.overrides)
        if "max_samples" in expanded and "num_samples" not in expanded:
            expanded["num_samples"] = expanded["max_samples"]
        expanded["force_task_type"] = _normalize_force_task_type(expanded.get("force_task_type"))
        config_hash = canonical_config_hash(expanded)
        run_dir = _prepare_run_dir(output_root, matrix_name, variant.tag, ts)
        cli_args = _build_cli_args(expanded, run_dir)

        run_config = {
            "matrix_name": matrix_name,
            "variant_tag": variant.tag,
            "variant_description": variant.description,
            "timestamp": ts,
            "config_hash": config_hash,
            "expanded_config": expanded,
            "cli_args": cli_args,
        }

        logger.info("[matrix] start variant=%s dir=%s", variant.tag, run_dir)
        try:
            runner(cli_args)
            _collect_predictions(run_dir, expanded, config_hash)
            (run_dir / "run_config.json").write_text(
                json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("[matrix] done variant=%s", variant.tag)
            return variant.tag, 0
        except SystemExit as exc:
            logger.exception("[matrix] variant=%s failed with SystemExit code=%s", variant.tag, exc.code)
            return variant.tag, 0
        except Exception:
            logger.exception("[matrix] variant=%s failed with unhandled exception", variant.tag)
            return variant.tag, 1

    if parallelism <= 1:
            for variant in variants:
                    tag, code = _run_variant(variant)
                    if code != 0:
                        failed_tags.append(tag)
                        if not continue_on_error:
                            return code
    else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {executor.submit(_run_variant, variant): variant for variant in variants}
                    for future in concurrent.futures.as_completed(futures):
                        tag, code = future.result()
                        if code != 0:
                            failed_tags.append(tag)
                            if not continue_on_error:
                                return code

    if failed_tags:
        logger.error("[matrix] failed variants: %s", ", ".join(failed_tags))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 4-way experiment matrix")
    parser.add_argument("--config", required=True, help="Path to matrix config (.json/.yaml)")
    parser.add_argument("--out", default=None, help="Optional output root override")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_matrix_config(args.config)
        if args.out:
            config["output_root"] = args.out
        return run_matrix(config)
    except ValueError as exc:
        print(f"[matrix] config error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())