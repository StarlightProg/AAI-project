import json
from pathlib import Path

from traceguard.types import Decision


def load(name):
    return json.loads((Path(__file__).parent / name).read_text(encoding="utf-8"))


def test_golden_cases_cover_all_decisions():
    cases = load("golden_cases.json")
    decisions = {case["expected_decision"] for case in cases}
    assert {item.value for item in Decision}.issubset(decisions)


def test_golden_cases_label_relevance_and_necessity_independently():
    for case in load("golden_cases.json"):
        assert case["goal_relevance"] in {"RELEVANT", "UNRELATED", "UNCERTAIN"}
        assert case["necessity"] in {"NECESSARY", "UNNECESSARY", "UNCERTAIN"}
        assert case["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_calibration_cases_are_not_empty():
    cases = load("calibration_cases.json")
    assert len(cases) >= 4
    assert all("expected_decision" in case for case in cases)
