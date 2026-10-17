from __future__ import annotations

from pathlib import Path

from ezscreen.results.run_layout import (
    open_last_run_log,
    resolve_output_dir,
    resolve_receptor_pdb,
    resolve_run_dir,
)


def test_resolve_output_dir_prefers_team_runner_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-team"
    run_dir.mkdir()
    (run_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")

    assert resolve_output_dir(run_dir) == run_dir


def test_resolve_output_dir_uses_legacy_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-legacy"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")

    assert resolve_output_dir(run_dir) == output_dir


def test_resolve_run_dir_normalizes_output_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-legacy"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)

    assert resolve_run_dir(output_dir) == run_dir
    assert resolve_run_dir(run_dir) == run_dir


def test_resolve_receptor_pdb_supports_team_runner_root_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-team"
    receptor_dir = run_dir / "receptor"
    receptor_dir.mkdir(parents=True)
    receptor_pdb = receptor_dir / "receptor_prep.pdb"
    receptor_pdb.write_text("HEADER\nEND\n", encoding="utf-8")
    (run_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")

    assert resolve_receptor_pdb(run_dir) == receptor_pdb


def test_resolve_receptor_pdb_supports_resume_pointer(tmp_path: Path) -> None:
    run_dir = tmp_path / "ezs-legacy"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    pointed = tmp_path / "receptor_prep.pdb"
    pointed.write_text("HEADER\nEND\n", encoding="utf-8")
    (output_dir / "scores.csv").write_text("ligand,score\nL1,-7.0\n", encoding="utf-8")
    (run_dir / "resume.json").write_text(
        '{"receptor_pdb": "' + str(pointed).replace("\\", "\\\\") + '"}',
        encoding="utf-8",
    )

    assert resolve_receptor_pdb(run_dir) == pointed


def test_open_last_run_log_creates_runs_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with open_last_run_log() as handle:
        handle.write("started\n")

    log_path = tmp_path / ".ezscreen" / "runs" / "last_run.log"
    assert log_path.read_text(encoding="utf-8") == "started\n"
