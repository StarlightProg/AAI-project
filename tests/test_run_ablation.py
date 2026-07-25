import argparse
from pathlib import Path
from types import SimpleNamespace

from traceguard.run_ablation import _run_custom


def test_custom_ablation_forwards_requested_supervisor_provider(monkeypatch, tmp_path):
    captured = {}

    def fake_load_cases(path, *, split):
        captured["load_cases"] = {"path": path, "split": split}
        return []

    def fake_run_experiment(**kwargs):
        captured["run_experiment"] = kwargs
        return [], SimpleNamespace(episode={}), tmp_path / "run"

    monkeypatch.setattr("traceguard.run_ablation.load_cases", fake_load_cases)
    monkeypatch.setattr("traceguard.run_ablation.run_experiment", fake_run_experiment)

    args = argparse.Namespace(
        output_dir=tmp_path,
        supervisor="llm",
        provider="ollama",
        supervisor_model="qwen3:4b",
        supervisor_url="http://127.0.0.1:11434",
        timeout=37.0,
        seed=11,
        agent_model="qwen3:4b",
        dry_run=False,
    )

    assert _run_custom(args) == 0

    run_kwargs = captured["run_experiment"]
    assert run_kwargs["supervisor_provider"] == "ollama"
    assert run_kwargs["supervisor_model"] == "qwen3:4b"
    assert run_kwargs["supervisor_url"] == "http://127.0.0.1:11434"
    assert run_kwargs["timeout"] == 37.0
    assert run_kwargs["seed"] == 11
    assert run_kwargs["artifacts_dir"] == Path(tmp_path)
