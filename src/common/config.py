from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


DEFAULT_CONFIG_HASH_FIELDS = (
    "dataset",
    "method",
    "solve_model",
    "guide_model",
    "solve_temperature",
    "guide_temperature",
    "max_retries",
    "prompt_contract_version",
    "prompt_versions",
    "prolog_contract_version",
    "swipl_schema_version",
    "meta_interpreter",
    "max_depth",
    "prolog_max_result",
    "force_task_type",
    "mock_profile",
    "mock_llm",
    "mock_prolog",
    "prolog_role",
)


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_config_hash(config: Mapping[str, Any], fields: tuple[str, ...] = DEFAULT_CONFIG_HASH_FIELDS) -> str:
    stable_payload = {field: config.get(field) for field in fields}
    digest = hashlib.sha256(stable_json_dumps(stable_payload).encode("utf-8")).hexdigest()
    return digest[:12]