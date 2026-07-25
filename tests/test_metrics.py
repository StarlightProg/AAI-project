from traceguard.metrics import (
    CallRecord,
    EpisodeRecord,
    bootstrap_ci,
    build_metric_report,
    call_metrics,
    episode_metrics,
    macro_f1,
    paired_ablation_delta,
    validate_call_labels,
)
from traceguard.types import ThreatModel


def test_call_metrics_formulas():
    records = [
        CallRecord(
            unsafe_proposal=True,
            call_blocked=True,
            harmful_effect=False,
            disruptive_intervention=False,
            tool_name_correct=False,
            arguments_correct=False,
        ),
        CallRecord(
            unsafe_proposal=True,
            call_allowed=True,
            harmful_effect=True,
            disruptive_intervention=False,
            tool_name_correct=False,
            arguments_correct=False,
        ),
        CallRecord(
            unsafe_proposal=False,
            call_blocked=True,
            harmful_effect=False,
            disruptive_intervention=True,
            tool_name_correct=True,
            arguments_correct=True,
        ),
        CallRecord(
            unsafe_proposal=False,
            call_allowed=True,
            harmful_effect=False,
            disruptive_intervention=False,
            tool_name_correct=True,
            arguments_correct=True,
        ),
    ]
    result = call_metrics(records)
    assert result["unsafe_effect_prevention"] == 0.5
    assert result["unsafe_call_blocking_rate"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["false_negative_rate"] == 0.5
    assert result["tool_call_accuracy"] == 0.5


def test_call_metrics_report_post_run_latency_and_tokens():
    records = [
        CallRecord(
            unsafe_proposal=False,
            harmful_effect=False,
            disruptive_intervention=False,
            post_run_correct=True,
            post_run_latency_ms=25,
            post_run_token_usage=12,
        ),
        CallRecord(
            unsafe_proposal=False,
            harmful_effect=False,
            disruptive_intervention=False,
            post_run_correct=False,
            post_run_latency_ms=35,
            post_run_token_usage=18,
        ),
    ]

    result = call_metrics(records)

    assert result["mean_post_run_latency_ms"] == 30
    assert result["mean_post_run_token_usage"] == 15
    assert result["total_post_run_token_usage"] == 30


def test_episode_metrics_formulas():
    records = [
        EpisodeRecord(
            adversarial=True,
            attacker_goal_achieved=True,
            utility_achieved=False,
            prohibited_effect=True,
        ),
        EpisodeRecord(
            adversarial=False,
            attacker_goal_achieved=False,
            utility_achieved=True,
            prohibited_effect=False,
        ),
    ]
    assert episode_metrics(records)["attack_success_rate"] == 1
    assert episode_metrics(records)["safe_completion_rate"] == 0.5


def test_macro_f1_perfect():
    assert macro_f1(["LOW", "HIGH"], ["LOW", "HIGH"]) == 1


def test_bootstrap_and_paired_delta():
    ci = bootstrap_ci([0.0, 1.0, 1.0, 0.0], n_boot=200, seed=1)
    assert ci["mean"] == 0.5
    assert ci["low"] is not None and ci["high"] is not None
    delta = paired_ablation_delta([0.0, 0.5], [0.5, 1.0])
    assert delta["mean_delta"] == 0.5


def test_build_metric_report_stratifies():
    episodes = [
        EpisodeRecord(
            case_id="a",
            threat_model=ThreatModel.DIRECT_ATTACK,
            adversarial=True,
            attacker_goal_achieved=False,
            utility_achieved=False,
            prohibited_effect=False,
        ),
        EpisodeRecord(
            case_id="b",
            threat_model=ThreatModel.BENIGN,
            adversarial=False,
            attacker_goal_achieved=False,
            utility_achieved=True,
            prohibited_effect=False,
        ),
    ]
    report = build_metric_report([], episodes, seed=0)
    assert "DIRECT_ATTACK" in report.by_threat_model
    assert "BENIGN" in report.by_threat_model


def test_call_label_validation_rejects_missing_gold():
    records = [
        CallRecord(unsafe_proposal=False, harmful_effect=False, disruptive_intervention=False)
    ]
    assert validate_call_labels(records) == [
        "call[0] missing relevance_gold",
        "call[0] missing necessity_gold",
    ]
