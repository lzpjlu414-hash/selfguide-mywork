from src.abs import self_guide_myself as sg


def test_parse_guideline_prolog_mode_from_json_text() -> None:
    guideline = '{"task_type":"Yes","schema":"a","query_goal":"b","fallback":"c","prolog_mode":"verifier"}'
    mode, defaulted = sg.parse_guideline_prolog_mode(guideline)
    assert mode == "verifier"
    assert defaulted is False


def test_parse_guideline_prolog_mode_missing_marks_defaulted() -> None:
    guideline = """task_type: Yes\nschema: |\n  x\nquery_goal: |\n  y\nfallback: |\n  z\n"""
    mode, defaulted = sg.parse_guideline_prolog_mode(guideline)
    assert mode is None
    assert defaulted is True