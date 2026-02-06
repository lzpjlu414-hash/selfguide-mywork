import json
from pathlib import Path

import pytest

from src.abs.self_guide_myself import parse_task_type_with_confidence, self_guide_run


LOW_CONF_GUIDELINE = '{"task_type":"Yes"}'
MID_CONF_GUIDELINE = (
    "task_type: Partial\n"
    "schema: |\n"
    "  Use fact(x, y).\n"
    "query_goal: |\n"
    "  query(result).\n"
    "fallback: |\n"
)
HIGH_CONF_GUIDELINE = (
    "task_type: Yes\n"
    "schema: |\n"
    "  Use snake_case naming for constants and predicates.\n"
    "  Use unknown atom for unknown values and not_fact(X) for negation.\n"
    "  Predicate signature example: score(student, points).\n"
    "query_goal: |\n"
    "  Build query(answer) to return one normalized answer.\n"
    "fallback: |\n"
    "  If parse error/timeout/multi-answer occurs, fallback to LLM answer and mark inconclusive.\n"
)


@pytest.mark.parametrize(
    "guideline,prolog_role,expected_effective_role,expect_executor",
    [
        (LOW_CONF_GUIDELINE, "executor", "off", False),
        (MID_CONF_GUIDELINE, "executor", "verifier", False),
        (HIGH_CONF_GUIDELINE, "executor", "executor", True),
    ],
)
def test_task_confidence_gate_and_logging_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guideline: str,
    prolog_role: str,
    expected_effective_role: str,
    expect_executor: bool,
) -> None:
    from src.abs import self_guide_myself as mod

    def _fake_generate_guideline_from_prompt(*args, **kwargs):
        _ = args, kwargs
        return guideline, True, 0

    monkeypatch.setattr(mod, "generate_guideline_from_prompt", _fake_generate_guideline_from_prompt)
    monkeypatch.setattr(mod, "SLEEP_SEC", 0.0)

    self_guide_run(
        dataset="gsm8k",
        method="cot_selfguide",
        start_index=0,
        num_samples=1,
        force_task_type="Yes",
        log_dir_override=str(tmp_path),
        mock_llm=True,
        mock_profile="prolog-role-test",
        mock_prolog=True,
        prolog_role=prolog_role,
    )

    log_file = next(tmp_path.glob("gsm8k_*.json"))
    payload = json.loads(log_file.read_text(encoding="utf-8"))

    parsed_task_type, confidence = parse_task_type_with_confidence(payload["guideline"])

    assert payload["task_type_raw"] == parsed_task_type
    assert payload["task_confidence"] == confidence

    if confidence <= 0.4:
        assert payload["task_type_effective"] == "No"
        assert payload["role_mode_effective"] == "off"
        assert payload["prolog"]["enabled"] is False
        assert payload["prolog_used"] is False
    elif confidence < 0.7:
        assert payload["task_type_effective"] == "Partial"
        assert payload["role_mode_effective"] == "verifier"
        assert payload["route"] != "executor"
    else:
        assert payload["task_type_effective"] == "Yes"

    if expect_executor:
        assert payload["route"] == "executor"
    else:
        assert payload["route"] != "executor"

    assert payload["role_mode_effective"] == expected_effective_role

    for obj in (payload, payload["prolog"]):
        assert "task_confidence" in obj
        assert "task_type_raw" in obj
        assert "task_type_effective" in obj
        assert "task_type_forced" in obj
        assert "task_type_source" in obj
        assert "confidence_gate_reason" in obj
        assert "role_mode_effective" in obj

    assert payload["task_type_forced"] == "Yes"
    assert payload["task_type_source"] == "forced"
    assert payload["prolog"]["task_type_forced"] == "Yes"
    assert payload["prolog"]["task_type_source"] == "forced"