from __future__ import annotations

import platform
import stat
from pathlib import Path

import requests

_VERSION = "1.2.7"
_BASE_URL = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v{_VERSION}"
_BIN_DIR  = Path.home() / ".ezscreen" / "bin"


def _platform_filename() -> str:
    system  = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return f"vina_{_VERSION}_win.exe"
    if system == "darwin":
        arch = "arm64" if "arm" in machine else "x86_64"
        return f"vina_{_VERSION}_mac_{arch}"
    return f"vina_{_VERSION}_linux_x86_64"


def _download(url: str, dest: Path) -> None:
    from rich.console import Console
    console = Console()
    console.print(f"  Downloading AutoDock Vina from {url}...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def get_vina_binary() -> Path:
    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    fname = _platform_filename()
    dest  = _BIN_DIR / fname
    if not dest.exists():
        _download(f"{_BASE_URL}/{fname}", dest)
        if platform.system().lower() != "windows":
            dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest
