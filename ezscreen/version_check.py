from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ezscreen import __version__

_PYPI_URL   = "https://pypi.org/pypi/ezscreen/json"
_CACHE_FILE = Path.home() / ".ezscreen" / "version_cache.json"
_CACHE_TTL  = 86_400   # 24 hours

_latest:   str | None = None
_done_evt: threading.Event = threading.Event()


def _load_cache() -> str | None:
    """Return cached latest version if still fresh, else None."""
    try:
        data = json.loads(_CACHE_FILE.read_text())
        age  = datetime.now(timezone.utc).timestamp() - data["checked_at"]
        if age < _CACHE_TTL:
            return data["latest"]
    except Exception:
        pass
    return None


def _save_cache(latest: str) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({
            "latest":     latest,
            "checked_at": datetime.now(timezone.utc).timestamp(),
        }))
    except Exception:
        pass


def _fetch() -> None:
    global _latest
    cached = _load_cache()
    if cached:
        _latest = cached
        _done_evt.set()
        return
    try:
        with urllib.request.urlopen(_PYPI_URL, timeout=3) as r:
            data    = json.loads(r.read())
            _latest = data["info"]["version"]
            _save_cache(_latest)
    except Exception:
        _latest = None
    _done_evt.set()


def start() -> None:
    """Fire-and-forget background version fetch."""
    threading.Thread(target=_fetch, daemon=True).start()


def banner() -> str | None:
    """
    Return a Rich-formatted banner if a newer version is available.
    Non-blocking -- returns None immediately if the check is still running.
    """
    if not _done_evt.is_set():
        return None
    if not _latest or _latest == __version__:
        return None
    return (
        f"[yellow bold]>> ezscreen {_latest} available[/yellow bold]  "
        f"[dim](you have {__version__})[/dim]  "
        f"[cyan]pip install -U ezscreen[/cyan]"
    )