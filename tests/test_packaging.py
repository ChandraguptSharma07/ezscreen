from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

from hatchling.build import build_wheel


def test_wheel_includes_runtime_assets(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    wheel_name = build_wheel(str(wheel_dir))

    with zipfile.ZipFile(wheel_dir / wheel_name) as zf:
        names = set(zf.namelist())

    assert "ezscreen/tui/theme.tcss" in names
    assert "ezscreen/backends/kaggle/templates/vina_shard.ipynb.j2" in names
    assert "ezscreen/backends/kaggle/templates/gnina_rescore.ipynb.j2" in names
    assert "ezscreen/backends/kaggle/templates/plip_analysis.ipynb.j2" in names
    assert "ezscreen/backends/kaggle/templates/md_openmm.ipynb.j2" in names
    assert "ezscreen/team/team-worker.yml" in names
    assert "ezscreen/benchmark/templates/campaign_manifest.template.json" in names
    assert "ezscreen/benchmark/templates/targets.template.json" in names
    assert "ezscreen/benchmark/templates/cell_manifest.template.json" in names
    assert "ezscreen/benchmark/templates/provenance.template.json" in names
    assert "ezscreen/benchmark/templates/failure_summary.template.json" in names
    assert "ezscreen/benchmark/templates/notes.template.md" in names


def test_sdist_leaves_out_benchmarks_and_docs():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    exclude = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]

    assert "/benchmarks" in exclude
    assert "/docs" in exclude
