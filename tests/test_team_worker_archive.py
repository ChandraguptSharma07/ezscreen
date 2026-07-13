from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ezscreen.team.worker import _safe_extract_zip


def _write_zip(path: Path, member: str, payload: bytes = b"data") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, payload)
    return path


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = _write_zip(tmp_path / "bad.zip", "../escape.txt")

    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError, match="Unsafe zip member path"):
            _safe_extract_zip(zf, tmp_path / "extract")


def test_safe_extract_zip_extracts_safe_members(tmp_path):
    archive = _write_zip(tmp_path / "ok.zip", "nested/notebook.ipynb", b"{}")

    dest = tmp_path / "extract"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        _safe_extract_zip(zf, dest)

    assert (dest / "nested" / "notebook.ipynb").read_bytes() == b"{}"
