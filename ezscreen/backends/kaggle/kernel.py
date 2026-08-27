from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from rich.console import Console

from ezscreen.errors import KaggleForbiddenError, KaggleUnauthorizedError

console = Console()
MAX_KERNEL_SLUG_LEN = 50
_MAX_RETRIES = 5
_BACKOFF_BASE = 2
# 409 = kernel version currently queued/saving — transient lock, safe to retry
_TRANSIENT_CODES = ("409", "429", "500", "502", "503", "504", "rate")
KAGGLE_BACKENDS = {
    "cpu": "none",
    "p100": "nvidiaTeslaP100",
    "t4x2": "nvidiaTeslaT4",
}


def accelerator_for_backend(backend: str) -> str:
    key = str(backend or "").strip().lower()
    if key not in KAGGLE_BACKENDS:
        raise ValueError(
            f"Unknown Kaggle backend {backend!r}; choose cpu, p100, or t4x2"
        )
    return KAGGLE_BACKENDS[key]


def kernel_slug(run_id: str) -> str:
    """Return a deterministic Kaggle-safe kernel slug bounded for SaveKernel."""
    safe = re.sub(r"[^a-z0-9-]+", "-", str(run_id).lower()).strip("-") or "run"
    if len(safe) <= MAX_KERNEL_SLUG_LEN:
        return safe
    digest = hashlib.sha1(str(run_id).encode("utf-8")).hexdigest()[:8]
    keep = MAX_KERNEL_SLUG_LEN - len(digest) - 1
    return f"{safe[:keep].rstrip('-')}-{digest}"


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
            is_transient = any(c in msg for c in _TRANSIENT_CODES)
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
    accelerator: str = "nvidiaTeslaP100",
    backend: str | None = None,
) -> str:
    """Render and push the notebook to Kaggle. Returns kernel ref."""
    api = _api()

    slug = kernel_slug(run_id)
    kernel_dir = Path(tempfile.gettempdir()) / "ezscreen_kaggle_kernels" / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, kernel_dir / "notebook.ipynb")

    # title must slugify to exactly the slug — replace hyphens with spaces so
    # Kaggle's slug derivation round-trips back to the same value
    title = slug.replace("-", " ")
    # Do not combine the deprecated generic enable_gpu flag with an explicit
    # machine shape. Kaggle gives the generic flag precedence and may silently
    # allocate its default P100 even when NvidiaTeslaT4 was requested.
    _MACHINE_SHAPE = {
        "nvidiaTeslaT4": "NvidiaTeslaT4",
        "nvidiaTeslaP100": "NvidiaTeslaP100",
    }
    accelerator_key = (
        accelerator_for_backend(backend)
        if backend is not None
        else str(accelerator or "").strip()
    )
    wants_gpu = accelerator_key.lower() not in {"", "none", "cpu"}
    machine_shape = _MACHINE_SHAPE.get(accelerator_key) if wants_gpu else None
    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }
    if machine_shape:
        meta["machine_shape"] = machine_shape
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    def _push():
        api.kernels_push(str(kernel_dir))

    _with_backoff(_push)
    return f"{username}/{slug}"


def delete_kernel(run_id: str, username: str) -> None:
    """Delete run kernel. Used by ezscreen clean."""
    api = _api()
    slug = run_id
    try:
        api.kernel_delete(username, slug)
    except Exception:
        pass  # best-effort
