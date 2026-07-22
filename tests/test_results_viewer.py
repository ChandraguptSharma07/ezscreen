from __future__ import annotations

from pathlib import Path

from ezscreen.tui.screens.results_viewer import ResultsScreen


def test_results_screen_prefers_team_runner_root_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-team"
    run_dir.mkdir()
    (run_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")

    assert ResultsScreen._resolve_output_dir(run_dir) == run_dir


def test_results_screen_uses_legacy_output_dir_when_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-legacy"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")

    assert ResultsScreen._resolve_output_dir(run_dir) == output_dir


def test_results_screen_work_dir_stays_at_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-team"
    run_dir.mkdir()
    (run_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")

    screen = ResultsScreen("ezs-team")
    screen._run_dir = run_dir
    screen._output = run_dir

    assert screen._work_dir() == run_dir


def test_results_screen_formats_benchmark_summary() -> None:
    summary = ResultsScreen._format_benchmark_summary(
        {
            "ef1": 2.0,
            "ef5": 1.5,
            "auc_roc": 0.8123,
            "n_actives": 8,
            "total_screened": 100,
        }
    )

    assert "EF 1%:" in summary
    assert "2.00x" in summary
    assert "0.812" in summary
    assert "8 / 100" in summary


def test_results_screen_uses_shared_benchmark_artifact_helper(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    def fake_generate_benchmark_artifacts(**kwargs):
        calls.update(kwargs)
        return {"result": {}, "report_path": kwargs["report_path"]}

    monkeypatch.setattr("ezscreen.commands.benchmark.generate_benchmark_artifacts", fake_generate_benchmark_artifacts)

    actives = tmp_path / "actives.smi"
    scores = tmp_path / "scores.csv"
    output = tmp_path / "output"
    output.mkdir()

    result = ResultsScreen._generate_benchmark_artifacts(actives, scores, output)

    assert result["report_path"] == output / "benchmark_report.html"
    assert calls["actives_path"] == actives
    assert calls["scores_csv"] == scores
    assert calls["report_path"] == output / "benchmark_report.html"
    assert calls["context_dir"] == output
