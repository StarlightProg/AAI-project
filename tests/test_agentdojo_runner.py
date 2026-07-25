from benchmarks.agentdojo_runner import summarize_suite_results


def test_agentdojo_summary_reports_accuracy_and_attack_success():
    summary = summarize_suite_results(
        "workspace",
        {
            "utility_results": {
                ("user_task_0", "injection_task_3"): True,
                ("user_task_1", "injection_task_3"): False,
            },
            "security_results": {
                ("user_task_0", "injection_task_3"): True,
                ("user_task_1", "injection_task_3"): False,
            },
            "injection_tasks_utility_results": {"injection_task_3": True},
        },
        attack="tool_knowledge",
    )

    assert summary.utility_accuracy == 0.5
    assert summary.security_accuracy == 0.5
    assert summary.attack_success_rate == 0.5
    assert summary.injection_task_utility_accuracy == 1.0


def test_agentdojo_summary_keeps_clean_security_not_attack_success():
    summary = summarize_suite_results(
        "workspace",
        {
            "utility_results": {"user_task_0": True},
            "security_results": {"user_task_0": True},
            "injection_tasks_utility_results": {},
        },
        attack=None,
    )

    assert summary.utility_accuracy == 1.0
    assert summary.security_accuracy == 1.0
    assert summary.attack_success_rate is None
