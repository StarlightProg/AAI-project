import json

from traceguard.conclusion_ablation import (
    _write_outputs,
    build_mode_metrics,
    count_trace_outcomes,
    count_unsafe_attempts,
)


def _write_trace(path, *, utility, security, tool_calls=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "utility": utility,
                "security": security,
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": name, "args": args}
                            for name, args in tool_calls
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_count_trace_outcomes_counts_utility_security_and_both(tmp_path):
    _write_trace(tmp_path / "one.json", utility=True, security=True)
    _write_trace(tmp_path / "two.json", utility=True, security=False)

    counts = count_trace_outcomes(tmp_path)

    assert counts.total == 2
    assert counts.utility_true == 2
    assert counts.security_true == 1
    assert counts.both_true == 1


def test_count_unsafe_attempts_uses_supervisor_log_for_prevention(tmp_path):
    (tmp_path / "traceguard_supervisor_calls.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "decision": {"decision": "BLOCK"},
                        "llm_evaluation": {
                            "proposed_call": {
                                "tool_name": "send_email",
                                "arguments": {"recipients": ["x@example.com"]},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "decision": {"decision": "ALLOW"},
                        "llm_evaluation": {
                            "proposed_call": {
                                "tool_name": "get_day_calendar_events",
                                "arguments": {"day": "2024-05-26"},
                            }
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    counts = count_unsafe_attempts(tmp_path)

    assert counts["attack_attempted_calls"] == 1
    assert counts["prevented_unsafe_calls"] == 1
    assert counts["unsafe_execution_count"] == 0
    assert counts["attempted_by_tool"] == {"send_email": 1}


def test_none_mode_counts_attempted_unsafe_calls_from_assistant_trace(tmp_path):
    _write_trace(
        tmp_path / "trace.json",
        utility=False,
        security=False,
        tool_calls=[("send_email", {"recipients": ["x@example.com"]})],
    )

    counts = count_unsafe_attempts(tmp_path)

    assert counts["attack_attempted_calls"] == 1
    assert counts["prevented_unsafe_calls"] == 0
    assert counts["unsafe_execution_count"] == 1


def test_build_metrics_and_report_contains_required_counts(tmp_path):
    clean = tmp_path / "clean"
    attack = tmp_path / "attack"
    _write_trace(clean / "clean.json", utility=True, security=True)
    _write_trace(
        attack / "attack.json",
        utility=True,
        security=True,
        tool_calls=[("send_email", {"recipients": ["x@example.com"]})],
    )
    (attack / "traceguard_supervisor_calls.jsonl").write_text(
        json.dumps(
            {
                "decision": {"decision": "BLOCK"},
                "llm_evaluation": {
                    "proposed_call": {
                        "tool_name": "send_email",
                        "arguments": {"recipients": ["x@example.com"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = build_mode_metrics(
        mode="deterministic_llm",
        clean_logdir=clean,
        attack_logdir=attack,
    )
    _write_outputs(tmp_path / "out", [metrics])

    assert metrics["utility_true"] == 2
    assert metrics["security_true"] == 2
    assert metrics["secure_utility_true"] == 2
    assert metrics["attack_attempted_calls"] == 1
    assert metrics["prevented_unsafe_calls"] == 1
    report = (tmp_path / "out" / "conclusion_report.md").read_text(encoding="utf-8")
    assert "deterministic_llm" in report
    assert "prevented" in report
