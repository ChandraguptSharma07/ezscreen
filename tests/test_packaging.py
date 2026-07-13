from __future__ import annotations

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
