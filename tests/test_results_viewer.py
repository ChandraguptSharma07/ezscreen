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
