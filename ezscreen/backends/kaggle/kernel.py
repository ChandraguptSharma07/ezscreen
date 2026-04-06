from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from rich.console import Console

from ezscreen.errors import (
    KaggleForbiddenError,
    KaggleRateLimitError,
    KaggleServerError,
    KaggleUnauthorizedError,
)

console = Console()
_MAX_RETRIES = 3
_BACKOFF_BASE = 2


def _api():
    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


def _with_backoff(fn, *args, **kwargs):
    """Retry transient errors with exponential backoff. Never retries 401/403."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (KaggleUnauthorizedError, KaggleForbiddenError):
            raise
        except Exception as exc:
            msg = str(exc).lower()
            is_transient = any(c in msg for c in ("429", "500", "502", "503", "504", "rate"))
            if not is_transient or attempt == _MAX_RETRIES - 1:
                raise
            wait = _BACKOFF_BASE ** (attempt + 1)
            console.print(f"  [dim]Kaggle error — retrying in {wait}s ({attempt + 1}/{_MAX_RETRIES})[/dim]")
            time.sleep(wait)


def push_kernel(
    run_id: str,
    notebook_path: Path,
    dataset_ref: str,
    username: str,
    work_dir: Path,
) -> str:
    """Render and push the notebook to Kaggle. Returns kernel ref."""
    api = _api()

    kernel_dir = work_dir / f"kernel_{run_id}"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, kernel_dir / "notebook.ipynb")

    slug = f"ezs-{run_id}"
    meta = {
        "id": f"{username}/{slug}",
        "title": f"ezscreen {run_id}",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    def _push():
        api.kernels_push(str(kernel_dir))

    _with_backoff(_push)
    return f"{username}/{slug}"


def delete_kernel(run_id: str, username: str) -> None:
    """Delete run kernel. Used by ezscreen clean."""
    api = _api()
    slug = f"ezs-{run_id}"
    try:
        api.kernel_delete(username, slug)
    except Exception:
        pass  # best-effort
