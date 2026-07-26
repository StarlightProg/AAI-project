from argparse import Namespace

from traceguard.cli import _demo


def test_demo_runs_baseline_and_hybrid(tmp_path, capsys):
    result = _demo(
        Namespace(
            root=tmp_path,
            artifacts=tmp_path / "artifacts",
            seed=0,
            gemini=False,
            gemini_model=None,
        )
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Baseline indirect injection : SUCCEEDED" in output
    assert "Hybrid indirect injection   : PREVENTED" in output
    assert "security=PASS" in output
    assert list((tmp_path / "artifacts").glob("run_*"))
