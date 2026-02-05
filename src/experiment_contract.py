from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    default_data_path: str
    sample_range_help: str = "[start_index, start_index + num_samples)"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    supported_datasets: tuple[str, ...]
    prompt_builder: str
    prolog_strategy: str
    default_decode: Mapping[str, float]
    default_force_task_type: str | None = None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(key="gsm8k", default_data_path="data/gsm8k/test.jsonl"),
    "prontoqa": DatasetSpec(key="prontoqa", default_data_path="data/prontoqa/dev.json"),
    "proofwriter": DatasetSpec(key="proofwriter", default_data_path="data/proofwriter/meta-test.shuffled.json"),
    "mmlu": DatasetSpec(key="mmlu", default_data_path="data/MMLU/test.jsonl"),
    "sqa": DatasetSpec(key="sqa", default_data_path="data/sqa/test.jsonl"),
    "date": DatasetSpec(key="date", default_data_path="data/date/test.jsonl"),
    "clutrr": DatasetSpec(key="clutrr", default_data_path="data/CLUTRR/test.jsonl"),
}


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "cot_selfguide": MethodSpec(
        key="cot_selfguide",
        supported_datasets=tuple(DATASET_REGISTRY.keys()),
        prompt_builder="src.abs.self_guide_myself.self_guide_run",
        prolog_strategy="optional-prolog-round",
        default_decode={"temperature": 0.2, "max_retries": 3},
    ),
    "sd_selfguide": MethodSpec(
        key="sd_selfguide",
        supported_datasets=tuple(DATASET_REGISTRY.keys()),
        prompt_builder="src.abs.self_guide_myself.self_guide_run",
        prolog_strategy="optional-prolog-round",
        default_decode={"temperature": 0.2, "max_retries": 3},
    ),
}


DEFAULTS = {
    "dataset": "gsm8k",
    "start_index": 0,
    "num_samples": 1,
    "meta_interpreter": "iter_deep_with_proof",
    "max_depth": 25,
    "prolog_max_result": 20,
    "prolog_role": "off",
}


def build_experiment_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run Self-Guide experiments (single entrypoint contract).")
    parser.add_argument("--dataset", default=DEFAULTS["dataset"], choices=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--method", required=True, choices=sorted(METHOD_REGISTRY.keys()))

    parser.add_argument("--start_index", type=int, default=DEFAULTS["start_index"])
    parser.add_argument("--num_samples", type=int, default=DEFAULTS["num_samples"])

    parser.add_argument("--data_path", default=None)
    parser.add_argument("--log_dir", default=None)

    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--mock_profile", default=None)
    parser.add_argument("--mock_prolog", action="store_true")
    parser.add_argument("--prolog_role", choices=("off", "verifier", "executor"), default=DEFAULTS["prolog_role"])

    parser.add_argument("--force_task_type", choices=("Yes", "No", "Partial"), default=None)
    parser.add_argument("--meta_interpreter", default=DEFAULTS["meta_interpreter"])
    parser.add_argument("--max_depth", type=int, default=DEFAULTS["max_depth"])
    parser.add_argument("--prolog_max_result", type=int, default=DEFAULTS["prolog_max_result"])
    parser.add_argument("--tmp_dir", default=None, help="root dir for Prolog temp files")
    parser.add_argument("--keep_tmp", action="store_true", help="keep Prolog temp files")

    parser.add_argument("--summarize", action="store_true", help="summarize logs after run")
    return parser


def validate_contract(args: Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("start_index must be >= 0")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if args.max_depth <= 0:
        raise ValueError("max_depth must be > 0")
    if args.prolog_max_result <= 0:
        raise ValueError("prolog_max_result must be > 0")
    if args.prolog_role not in ("off", "verifier", "executor"):
        raise ValueError("prolog_role must be one of: off, verifier, executor")

    method_spec = METHOD_REGISTRY[args.method]
    if args.dataset not in method_spec.supported_datasets:
        allowed = ", ".join(method_spec.supported_datasets)
        raise ValueError(f"dataset '{args.dataset}' is not supported by method '{args.method}'. allowed: {allowed}")


def resolve_log_dir(args: Namespace) -> Path:
    return Path(args.log_dir) if args.log_dir else Path(f"log/{args.method}/{args.dataset}")