from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def create_base_parser(
    description: str,
    *,
    dataset_help: str,
    method_help: str,
    include_num_samples: bool = False,
    include_log_dir: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True, help=dataset_help)
    parser.add_argument("--method", required=True, help=method_help)
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")
    if include_num_samples:
        parser.add_argument("--num_samples", type=int, default=1, help="Number of LLM responses to sample")
    parser.add_argument("--data_path", default=None, help="override dataset jsonl path")
    if include_log_dir:
        parser.add_argument("--log_dir", default=None, help="override log directory")
    parser.add_argument("--mock_llm", action="store_true", help="use deterministic mock outputs (no API call)")
    parser.add_argument("--mock_profile", default=None, help="mock profile name for mock_llm")
    return parser


def ensure_log_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_main(main_fn) -> None:
    try:
        main_fn()
    except KeyboardInterrupt:
        print(json.dumps({"error_code": "INTERRUPTED", "message": "Interrupted by user"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(130)
    except FileNotFoundError as exc:
        print(json.dumps({"error_code": "DATA_NOT_FOUND", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
    except ValueError as exc:
        print(json.dumps({"error_code": "INVALID_ARGUMENT", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)