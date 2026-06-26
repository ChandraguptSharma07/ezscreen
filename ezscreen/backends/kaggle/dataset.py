from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ezscreen.errors import (
    KaggleBadRequestError,
    KaggleForbiddenError,
    KaggleNotFoundError,
    KaggleRateLimitError,
    KaggleServerError,
    KaggleUnauthorizedError,
)

MANIFEST_PATH = Path.home() / ".ezscreen" / "manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api():
    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


def _error_detail(exc: Exception) -> str:
    """str(exc) plus the Kaggle HTTP response body, which carries the real reason
    (e.g. why a 400 was rejected). The SDK's HTTPError keeps it on .response."""
    base = str(exc)
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.text
        except Exception:
            body = ""
        if body and body.strip() and body.strip() not in base:
            return f"{base} :: {body.strip()[:600]}"
    return base


def _handle_error(exc: Exception) -> None:
    detail = _error_detail(exc)
    msg = detail.lower()
    if "401" in msg or "unauthorized" in msg:
        raise KaggleUnauthorizedError(
            "API key rejected — go to kaggle.com/settings → API → Create New Token"
        ) from exc
    if "403" in msg or "forbidden" in msg:
        raise KaggleForbiddenError(
            "Account needs phone verification — complete at kaggle.com/settings"
        ) from exc
    if "404" in msg or "not found" in msg:
        raise KaggleNotFoundError(detail) from exc
    if "429" in msg or "rate limit" in msg:
        raise KaggleRateLimitError(detail) from exc
    if any(c in msg for c in ("500", "502", "503", "504")):
        raise KaggleServerError(detail) from exc
    # Empty/non-JSON response body — Kaggle's edge layer returned a 5xx-equivalent
    # without a JSON body. Treat as a transient server error so callers can retry.
    if "expecting value" in msg or "jsondecodeerror" in msg:
        raise KaggleServerError(
            f"Kaggle returned an empty response (likely transient server issue): {detail}"
        ) from exc
    raise KaggleBadRequestError(detail) from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(m: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_run_dataset(
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    username: str,
    work_dir: Path,
) -> str:
    """
    Upload receptor (skipped if SHA-256 matches) and ligand shards.
    Returns dataset ref: 'username/ezscreen-{run_id}'.
    """
    api = _api()
    manifest = _load_manifest()

    dataset_dir = work_dir / f"dataset_{run_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Receptor — dedup; always uploaded as "receptor.pdbqt" so the notebook
    # template can reference a stable, predictable filename
    receptor_hash = sha256(receptor_pdbqt)
    cache_key = str(receptor_pdbqt.resolve())
    manifest[cache_key] = receptor_hash
    shutil.copy2(receptor_pdbqt, dataset_dir / "receptor.pdbqt")

    # Shards — always fresh
    for sp in shard_paths:
        shutil.copy2(sp, dataset_dir / sp.name)

    slug = f"ezscreen-{run_id}"
    meta = {
        "title": f"ezscreen {run_id}",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Retry transient server errors (empty body, 5xx) — these happen sporadically
    # at Kaggle's edge layer and almost always succeed on a second attempt.
    import time
    for attempt in range(5):
        try:
            api.dataset_create_new(str(dataset_dir), public=False, quiet=True)
            break
        except Exception as exc:
            try:
                _handle_error(exc)
            except KaggleServerError:
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 1))

    _save_manifest(manifest)
    return f"{username}/{slug}"


def delete_run_dataset(run_id: str, username: str) -> None:
    """Delete a run's Kaggle dataset. Used by ezscreen clean."""
    api = _api()
    slug = f"ezscreen-{run_id}"
    try:
        api.dataset_delete(username, slug)
    except Exception as exc:
        _handle_error(exc)
