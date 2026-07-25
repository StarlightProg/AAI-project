import json
from pathlib import Path

from traceguard.supervisor.llm import QwenSupervisor
from traceguard.types import (
    Decision,
    PostRunDisposition,
    RiskLevel,
    SandboxEvidence,
    ThreatModel,
    ToolCall,
)


def load(name):
    return json.loads((Path(__file__).parent / name).read_text(encoding="utf-8"))


def test_golden_cases_cover_all_decisions():
    cases = load("golden_cases.json")
    decisions = {case["expected_decision"] for case in cases}
    assert {item.value for item in Decision}.issubset(decisions)


def test_golden_cases_label_relevance_and_necessity_independently():
    for case in load("golden_cases.json"):
        assert case["threat_model"] in {item.value for item in ThreatModel}
        assert case["goal_relevance"] in {"RELEVANT", "UNRELATED", "UNCERTAIN"}
        assert case["necessity"] in {"NECESSARY", "UNNECESSARY", "UNCERTAIN"}
        assert case["risk_level"] in {item.value for item in RiskLevel}


def test_golden_cases_cover_all_primary_threat_models():
    cases = load("golden_cases.json")
    threat_models = {case["threat_model"] for case in cases}
    assert {item.value for item in ThreatModel}.issubset(threat_models)


def test_golden_cases_cover_independent_relevance_necessity_edges():
    cases = load("golden_cases.json")
    assert any(
        case["goal_relevance"] == "RELEVANT" and case["necessity"] == "UNNECESSARY"
        for case in cases
    )
    assert any(
        case["necessity"] == "NECESSARY" and case["risk_level"] in {"HIGH", "CRITICAL"}
        for case in cases
    )
    assert any(
        case["goal_relevance"] == "UNRELATED" and case["risk_level"] == "LOW" for case in cases
    )


def test_calibration_cases_are_not_empty():
    cases = load("calibration_cases.json")
    assert len(cases) >= 4
    assert all("expected_decision" in case for case in cases)


def test_post_run_golden_cases_use_runtime_enums_and_required_scenarios():
    cases = load("post_run_golden_cases.json")
    expected_names = {
        "post_accept_harmless_success",
        "post_reject_blocked_network",
        "post_reject_suspicious_file_creation",
        "post_escalate_timeout",
        "post_reject_resource_exhaustion",
        "post_reject_deceptive_stdout",
    }

    assert expected_names.issubset({case["case_id"] for case in cases})
    for case in cases:
        assert case["expected_post_decision"] in {item.value for item in PostRunDisposition}
        assert case["expected_risk"] in {item.value for item in RiskLevel}


def test_post_run_golden_cases_match_runtime_behavior():
    supervisor = QwenSupervisor(post_run_provider_enabled=False)
    call = ToolCall(
        task_id="post-run-golden",
        step_id=0,
        tool_name="restricted_command",
        arguments={"command": ["test"]},
    )
    for case in load("post_run_golden_cases.json"):
        result = supervisor.reevaluate(
            "Run the requested operation.",
            call,
            SandboxEvidence.model_validate(case["sandbox_evidence"]),
        )
        assert result.disposition.value == case["expected_post_decision"], case["case_id"]
        assert result.risk.value == case["expected_risk"], case["case_id"]
