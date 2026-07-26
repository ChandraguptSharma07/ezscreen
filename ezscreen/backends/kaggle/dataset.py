from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
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
MAX_DATASET_SLUG_LEN = 50


def dataset_slug(run_id: str) -> str:
    """Return a deterministic Kaggle-safe dataset slug."""
    safe = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-") or "run"
    slug = f"ezscreen-{safe}"
    if len(slug) <= MAX_DATASET_SLUG_LEN:
        return slug
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]
    keep = MAX_DATASET_SLUG_LEN - len(digest) - 1
    return f"{slug[:keep].rstrip('-')}-{digest}"


def _bundle_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:8]


def _validate_create_response(response: object) -> None:
    """Kaggle SDK 2.x can return an error response without raising."""
    error = getattr(response, "error", None)
    status = str(getattr(response, "status", "") or "").strip().lower()
    if error or status in {"error", "failed", "failure"}:
        raise KaggleBadRequestError(
            f"Kaggle dataset creation failed: {error or status or 'unknown error'}"
        )


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
    receptor_files: dict[str, Path] | None = None,
    receptor_pdb: Path | None = None,
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
    for name, path in (receptor_files or {}).items():
        if name == "receptor.pdbqt":
            continue
        manifest[str(path.resolve())] = sha256(path)
        shutil.copy2(path, dataset_dir / name)
    if receptor_pdb is not None:
        manifest[str(receptor_pdb.resolve())] = sha256(receptor_pdb)
        shutil.copy2(receptor_pdb, dataset_dir / "receptor_fixed.pdb")
    # Shards — always fresh
    for sp in shard_paths:
        shutil.copy2(sp, dataset_dir / sp.name)

    slug = dataset_slug(run_id)
    meta = {
        "title": slug.replace("-", " ")[:MAX_DATASET_SLUG_LEN],
        "id": f"{username}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # Retry transient server errors (empty body, 5xx) — these happen sporadically
    # at Kaggle's edge layer and almost always succeed on a second attempt.
    import time
    for attempt in range(5):
        try:
            response = api.dataset_create_new(str(dataset_dir), public=False, quiet=True, dir_mode="skip")
            _validate_create_response(response)
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


def upload_files_dataset(
    run_id: str,
    files: list[Path],
    username: str,
    work_dir: Path,
    *,
    title: str | None = None,
) -> str:
    """Upload an arbitrary immutable input bundle for a custom Kaggle kernel.

    File basenames must be unique.  The local staging directory and manifest
    record hashes, never credential contents.
    """
    if not files:
        raise ValueError("At least one dataset file is required")
    resolved = [Path(path).expanduser().resolve() for path in files]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset input missing: {missing[0]}")
    names = [path.name for path in resolved]
    if len(names) != len(set(names)):
        raise ValueError("Dataset input basenames must be unique")

    api = _api()
    manifest = _load_manifest()
    fingerprint = _bundle_fingerprint(resolved)
    slug = dataset_slug(f"{run_id}-{fingerprint}")
    dataset_dir = Path(tempfile.gettempdir()) / "ezscreen_kaggle_datasets" / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for path in resolved:
        manifest[str(path)] = sha256(path)
        shutil.copy2(path, dataset_dir / path.name)

    metadata = {
        "title": title or f"ezscreen {run_id}",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    import time
    for attempt in range(5):
        try:
            response = api.dataset_create_new(str(dataset_dir), public=False, quiet=True, dir_mode="skip")
            _validate_create_response(response)
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
    slug = dataset_slug(run_id)
    try:
        api.dataset_delete(username, slug)
    except Exception as exc:
        _handle_error(exc)
