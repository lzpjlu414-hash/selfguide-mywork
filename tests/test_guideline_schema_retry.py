from src.abs import self_guide_myself as sg


def test_guideline_retry_recovers_to_structured_schema(monkeypatch) -> None:
    calls = {"guideline": 0, "guideline_retry": 0}

    def fake_ai_request(history, model, t=0.2, max_retries=3, timeout=None, mock_llm=False, mock_profile=None, dataset_key=None, prompt_type=None):
        if prompt_type == "guideline":
            calls["guideline"] += 1
            return "task_type: Yes\nschema: |\n  only naming\nquery_goal: |\n  prove x"
        if prompt_type == "guideline_retry":
            calls["guideline_retry"] += 1
            return (
                "task_type: Yes\n"
                "schema: |\n"
                "  - naming: snake_case constants.\n"
                "  - unknown: use unknown atom; negation with not_fact(X).\n"
                "  - predicate signature: score(student, points).\n"
                "query_goal: |\n"
                "  - Return one final score answer.\n"
                "fallback: |\n"
                "  - If parse fails, keep LLM final answer.\n"
            )
        return ""

    monkeypatch.setattr(sg, "ai_request", fake_ai_request)

    guideline, valid, retry_count = sg.generate_guideline_from_prompt(
        "prompt", "rule", dataset_key="gsm8k", mock_llm=True, mock_profile="x"
    )

    assert valid is True
    assert retry_count >= 1
    assert "task_type:" in guideline and "query_goal:" in guideline and "fallback:" in guideline
    assert calls["guideline_retry"] >= 1


def test_guideline_retry_exhausted_uses_fallback(monkeypatch) -> None:
    def fake_ai_request(history, model, t=0.2, max_retries=3, timeout=None, mock_llm=False, mock_profile=None, dataset_key=None, prompt_type=None):
        return "task_type: Yes\nschema: |\n  bad"

    monkeypatch.setattr(sg, "ai_request", fake_ai_request)

    guideline, valid, retry_count = sg.generate_guideline_from_prompt(
        "prompt", "rule", dataset_key="gsm8k", mock_llm=True, mock_profile="x"
    )

    assert valid is False
    assert retry_count == sg.MAX_RETRIES
    assert "fallback:" in guideline