from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import requests as _req
from rich.console import Console

from ezscreen.backends.base import DockingBackend
from ezscreen.backends.kaggle.dataset import upload_run_dataset
from ezscreen.backends.kaggle.kernel import delete_kernel, push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done

console = Console()


def _safe_extract_zip(zf, destination: Path) -> None:
    destination = destination.resolve()
    for info in zf.infolist():
        target = (destination / info.filename).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"Unsafe zip member path: {info.filename}")
    zf.extractall(destination)
_DATASET_POLL_INTERVAL = 10   # seconds
_DATASET_TIMEOUT = 300        # 5 minutes
_DATASET_SYNC_POLL_INTERVAL = 10
_DATASET_SYNC_TIMEOUT = 90
_DATASET_SYNC_READY_STREAK = 7
_MAX_PARALLEL_KERNELS = 2
_KAGGLE_API_LOCK = threading.Lock()  # serialises env-var switching + authenticate()


class KaggleVinaBackend(DockingBackend):
    def __init__(self, engine: str = "unidock") -> None:
        self.engine_key = engine

    def run(
        self,
        ligands: Sequence[Path],
        receptor: Path,
        box: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if self.requires_box and box is None:
            raise ValueError(f"{self.engine_key} backend requires a docking box")
        cfg = dict(config)
        center = list(box.center if hasattr(box, "center") else box["center"])
        size = list(box.size if hasattr(box, "size") else box["size"])
        work_dir = Path(cfg["work_dir"])

        account_assignments = cfg.get("account_assignments")
        if account_assignments:
            return run_multi_account_screening(
                run_id=str(cfg["run_id"]),
                receptor_pdbqt=Path(receptor),
                shard_paths=[Path(path) for path in ligands],
                account_assignments=account_assignments,
                work_dir=work_dir,
                receptor_pdbqts=cfg.get("receptor_pdbqts"),
                receptor_variant_modes=cfg.get("receptor_variant_modes"),
                receptor_pdb=cfg.get("receptor_pdb"),
                box_center=center,
                box_size=size,
                search_mode=str(cfg.get("search_mode", "balance")),
                ph=float(cfg.get("ph", 7.4)),
                retry_limit=int(cfg.get("retry_limit", 3)),
                accelerator=str(cfg.get("accelerator", "nvidiaTeslaP100")),
                prep_on_kaggle=bool(cfg.get("prep_on_kaggle", True)),
                force_field=cfg.get("force_field"),
                enumerate_opts=cfg.get("enumerate_opts"),
                engine=self.engine_key,
                scoring=str(cfg.get("scoring", "vina")),
            )

        notebook_path = Path(cfg.get("notebook_path", work_dir / "notebook.ipynb"))
        if not notebook_path.exists():
            notebook_path.parent.mkdir(parents=True, exist_ok=True)
            notebook_path.write_text(
                render_vina_notebook(
                    run_id=str(cfg["run_id"]),
                    shard_filenames=[Path(path).name for path in ligands],
                    box_center=center,
                    box_size=size,
                    search_mode=str(cfg.get("search_mode", "balance")),
                    ph=float(cfg.get("ph", 7.4)),
                    prep_on_kaggle=bool(cfg.get("prep_on_kaggle", True)),
                    accelerator=str(cfg.get("accelerator", "nvidiaTeslaP100")),
                    force_field=cfg.get("force_field"),
                    enumerate_opts=cfg.get("enumerate_opts"),
                    prep_cfg=cfg.get("prep_cfg"),
                    results_cfg=cfg.get("results_cfg"),
                    engine=self.engine_key,
                    scoring=str(cfg.get("scoring", "vina")),
                ),
                encoding="utf-8",
            )

        return run_screening_job(
            run_id=str(cfg["run_id"]),
            receptor_pdbqt=Path(receptor),
            shard_paths=[Path(path) for path in ligands],
            notebook_path=notebook_path,
            username=str(cfg["username"]),
            work_dir=work_dir,
            retry_limit=int(cfg.get("retry_limit", 3)),
            accelerator=str(cfg.get("accelerator", "nvidiaTeslaP100")),
            engine=self.engine_key,
        )

def _wait_for_dataset(dataset_ref: str) -> None:
    """Block until the Kaggle dataset status is 'ready'."""
    import kaggle
    kaggle.api.authenticate()
    elapsed = 0
    console.print("  [dim]Waiting for dataset to be ready...[/dim]", end="")
    while elapsed < _DATASET_TIMEOUT:
        try:
            status = kaggle.api.dataset_status(dataset_ref)
            if status == "ready":
                console.print(f" ready ({elapsed}s)")
                return
        except Exception:
            pass
        time.sleep(_DATASET_POLL_INTERVAL)
        elapsed += _DATASET_POLL_INTERVAL
        console.print(".", end="")
    console.print(f" [yellow]timed out after {elapsed}s — proceeding anyway[/yellow]")


def _wait_for_dataset_sync(
    dataset_refs: str | Sequence[str],
    *,
    label: str = "dataset files",
) -> None:
    """Wait for repeated ready states before pushing a kernel.

    Kaggle often reports dataset metadata as ready before the uploaded files are
    consistently visible to compute nodes. We treat a sustained sequence of
    `ready` responses as the best lightweight proxy we have before submission.
    """
    refs = [dataset_refs] if isinstance(dataset_refs, str) else [ref for ref in dataset_refs if ref]
    if not refs:
        return
    import kaggle

    kaggle.api.authenticate()
    elapsed = 0
    ready_streak = 0
    console.print(f"  [dim]Waiting for {label} to sync to compute nodes...[/dim]", end="")
    while elapsed < _DATASET_SYNC_TIMEOUT:
        try:
            if all(kaggle.api.dataset_status(ref) == "ready" for ref in refs):
                ready_streak += 1
                if ready_streak >= _DATASET_SYNC_READY_STREAK:
                    console.print(f" ready ({elapsed}s)")
                    return
            else:
                ready_streak = 0
        except Exception:
            ready_streak = 0
        time.sleep(_DATASET_SYNC_POLL_INTERVAL)
        elapsed += _DATASET_SYNC_POLL_INTERVAL
        console.print(".", end="")
    console.print(f" [yellow]timed out after {elapsed}s - proceeding anyway[/yellow]")


def _load_index(output_dir: Path) -> dict[str, dict]:
    """Load ligand index.csv (ligand → {name, smiles}) if present.

    index.csv is written by ligand prep next to the shard files at
    <work_dir>/shards/index.csv.  output_dir is <work_dir>/output/.
    """
    import csv
    candidates = [
        output_dir / "index.csv",                      # flat download
        output_dir.parent / "shards" / "index.csv",   # normal run layout
        output_dir.parent / "index.csv",               # fallback
    ]
    for index_path in candidates:
        if index_path.exists():
            with index_path.open() as f:
                return {row["ligand"]: row for row in csv.DictReader(f)}
    return {}


def _recover_scores(output_dir: Path) -> None:
    """Parse docked PDBQT files locally to regenerate scores.csv if missing.

    Kaggle's kernels_output API downloads subdirectories but not all flat files,
    so scores.csv written by Cell 8 may not be downloaded even when docking succeeds.
    """
    import csv
    import re
    scores_csv = output_dir / "scores.csv"
    if scores_csv.exists():
        return
    docked_dir = output_dir / "docked"
    if not docked_dir.exists():
        return
    out_files = sorted(docked_dir.glob("*_out.pdbqt"))
    if not out_files:
        return

    index = _load_index(output_dir)

    rows = []
    for p in out_files:
        if p.stem.startswith("lig_pad_"):
            continue
        text = p.read_text(errors="replace")
        m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", text)
        if not m:
            continue
        lig_id = p.stem.removesuffix("_out")
        row: dict = {
            "ligand": lig_id,
            "score":  float(m.group(1)),
        }
        if lig_id in index:
            row["name"]   = index[lig_id]["name"]
            row["smiles"] = index[lig_id]["smiles"]
        rows.append(row)

    rows.sort(key=lambda r: r["score"])

    # Vina-family scores are always negative for real binding events.
    # Scores < -15 are GPU underflow artifacts; scores >= 0 are GPU overflow artifacts
    # (UniDock reports values like 3.776e+06 kcal/mol for "Large" high-torsion molecules).
    _SCORE_FLOOR = -15.0
    _SCORE_CEIL  =   0.0
    filtered = [r for r in rows if _SCORE_FLOOR <= r["score"] < _SCORE_CEIL]
    n_artifacts = len(rows) - len(filtered)
    if n_artifacts:
        console.print(f"  [dim]Filtered {n_artifacts} artifact score(s) outside [{_SCORE_FLOOR}, {_SCORE_CEIL}) kcal/mol[/dim]")
    rows = filtered

    # Include name/smiles columns only if any row has them
    has_identity = any("smiles" in r for r in rows)
    fieldnames = ["ligand", "score"]
    if has_identity:
        fieldnames += ["name", "smiles"]

    with scores_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    console.print(f"  [dim]Recovered scores.csv locally ({len(rows)} poses)[/dim]")


def _enrich_scores_with_identity(output_dir: Path) -> None:
    """Add name/smiles columns to an existing scores.csv if index.csv is available.

    Kaggle's Cell 8 writes scores.csv without identity columns.  This runs
    after download (whether scores.csv came from Kaggle or was recovered locally)
    and patches in name/smiles from the local index.csv when available.
    """
    import csv
    scores_csv = output_dir / "scores.csv"
    if not scores_csv.exists():
        return
    index = _load_index(output_dir)
    if not index:
        return

    with scores_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows or "smiles" in fieldnames:
        return  # nothing to enrich

    # Drop legacy rmsd columns while we're here
    fieldnames = [c for c in fieldnames if c not in ("rmsd_lb", "rmsd_ub")]

    enriched = 0
    for row in rows:
        lig_id = row.get("ligand", "")
        if lig_id in index:
            row["name"]   = index[lig_id]["name"]
            row["smiles"] = index[lig_id]["smiles"]
            enriched += 1
        # remove rmsd keys from row dict too
        row.pop("rmsd_lb", None)
        row.pop("rmsd_ub", None)

    if not enriched:
        return

    fieldnames += ["name", "smiles"]

    # Compute LE/BEI now that we have SMILES — mirrors what merge_shard_results does
    # for multi-account runs, ensuring single-account runs get the same columns.
    score_col = next(
        (h for h in fieldnames if "score" in h.lower() or "affinity" in h.lower()),
        None,
    )
    from ezscreen.results.merger import _add_efficiency_cols
    rows, fieldnames = _add_efficiency_cols(rows, fieldnames, score_col)

    with scores_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    console.print(f"  [dim]Enriched scores.csv with identity + LE/BEI ({enriched} rows)[/dim]")



def _tag_scores_with_receptor_variant(output_dir: Path, receptor_variant: str | None) -> None:
    """Ensure scores/failed rows identify the receptor variant that produced them."""
    if not receptor_variant:
        return
    import csv
    for filename in ("scores.csv", "failed_docking.csv"):
        path = output_dir / filename
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        if not fieldnames or "receptor_variant" in fieldnames:
            continue
        fieldnames.append("receptor_variant")
        for row in rows:
            row["receptor_variant"] = receptor_variant
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)




def _tag_scores_with_engine(output_dir: Path, engine: str) -> None:
    """Ensure scores rows identify the engine that produced them."""
    import csv

    path = output_dir / "scores.csv"
    if not path.exists():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        return
    if "engine" not in fieldnames:
        fieldnames.append("engine")
    for row in rows:
        row.setdefault("engine", engine)
        if not row["engine"]:
            row["engine"] = engine
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_results_meta(output_dir: Path, engine: str, score_col: str | None = None) -> None:
    """Write viewer/report metadata for a downloaded engine output."""
    import csv

    from ezscreen.backends import engines as _engines

    if score_col is None:
        scores_csv = output_dir / "scores.csv"
        if scores_csv.exists():
            with scores_csv.open(newline="") as f:
                fieldnames = list(csv.DictReader(f).fieldnames or [])
            score_col = next(
                (h for h in fieldnames if "score" in h.lower() or "affinity" in h.lower()),
                fieldnames[-1] if fieldnames else None,
            )
    (output_dir / "results_meta.json").write_text(
        json.dumps(
            {
                "score_type": _engines.native_score_type(engine),
                "score_col": score_col,
                "engine": engine,
            },
            indent=2,
        )
    )

def _write_single_unscored_reasons(output_dir: Path) -> None:
    """Write unscored_reasons.csv for single-notebook outputs."""
    import csv

    reasons: list[dict[str, str]] = []

    for filename, reason in (
        ("filtered_gpu_size.csv", "gpu_size_filter"),
        ("failed_prep.csv", "prep_failed"),
    ):
        path = output_dir / filename
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                reasons.append(
                    {
                        "ligand": "",
                        "name": row.get("name", ""),
                        "smiles": row.get("smiles", ""),
                        "reason": row.get("reason", reason),
                        "raw_score": "",
                    }
                )

    failed_docking = output_dir / "failed_docking.csv"
    if failed_docking.exists():
        index = _load_index(output_dir)
        with failed_docking.open(newline="") as f:
            for row in csv.DictReader(f):
                lig_id = row.get("ligand", "")
                info = index.get(lig_id, {})
                reasons.append(
                    {
                        "ligand": lig_id,
                        "name": info.get("name", ""),
                        "smiles": info.get("smiles", ""),
                        "reason": row.get("reason", "docking_failed"),
                        "raw_score": row.get("raw_score", ""),
                    }
                )

    if not reasons:
        return

    with (output_dir / "unscored_reasons.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["ligand", "name", "smiles", "reason", "raw_score"]
        )
        w.writeheader()
        w.writerows(reasons)


def _fetch_output_urls(kernel_ref: str) -> list:
    """Return list of (filename, url) tuples for a kernel's output files.
    Caller must hold _KAGGLE_API_LOCK and have already set credentials.
    """
    import kaggle
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )
    if "/" in kernel_ref:
        owner_slug, kernel_slug = kernel_ref.split("/", 1)
    else:
        owner_slug = kaggle.api.config_values.get(kaggle.api.CONFIG_NAME_USER, "")
        kernel_slug = kernel_ref
    with kaggle.api.build_kaggle_client() as kc:
        req = ApiListKernelSessionOutputRequest()
        req.user_name   = owner_slug
        req.kernel_slug = kernel_slug
        resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
    return [(f.file_name, f.url) for f in (resp.files or [])]


def _download_output(
    kernel_ref: str,
    work_dir: Path,
    kj_path: str | None = None,
    index_csv: Path | None = None,
    filtered_csv: Path | None = None,
    receptor_variant: str | None = None,
    engine: str = "unidock",
    retries: int = 5,
) -> Path:
    out = work_dir / "output"
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1 — get pre-signed URLs under the lock (brief API call only)
    file_urls: list = []
    for attempt in range(retries):
        try:
            with _KAGGLE_API_LOCK:
                if kj_path:
                    _set_account_creds(kj_path)
                import kaggle
                kaggle.api.authenticate()
                file_urls = _fetch_output_urls(kernel_ref)
            break
        except ImportError:
            # kagglesdk not available — fall back to the SDK helper
            with _KAGGLE_API_LOCK:
                if kj_path:
                    _set_account_creds(kj_path)
                import kaggle
                kaggle.api.authenticate()
                kaggle.api.kernels_output(kernel_ref, path=str(out), quiet=True)
            file_urls = []
            break
        except Exception as exc:
            # On the last retry, try the old SDK fallback before giving up —
            # kagglesdk returns 403 for some account/kernel combinations even
            # when the kernel is complete and the credentials are valid.
            if attempt == retries - 1:
                try:
                    with _KAGGLE_API_LOCK:
                        if kj_path:
                            _set_account_creds(kj_path)
                        import kaggle
                        kaggle.api.authenticate()
                        kaggle.api.kernels_output(kernel_ref, path=str(out), quiet=True)
                    file_urls = []
                    break
                except Exception:
                    raise exc
            wait = 2 ** (attempt + 1)
            console.print(f"  [yellow]Output list error (attempt {attempt + 1}/{retries}), retrying in {wait}s: {exc}[/yellow]")
            time.sleep(wait)

    # Phase 2 — download files in parallel; pre-signed URLs need no auth
    def _dl(name_url: tuple[str, str]) -> None:
        name, url = name_url
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = _req.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)

    if file_urls:
        with ThreadPoolExecutor(max_workers=min(8, len(file_urls))) as pool:
            list(pool.map(_dl, file_urls))

    # Unzip if the notebook bundled outputs into output.zip
    zip_path = out / "output.zip"
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract_zip(zf, out)
        zip_path.unlink()

    # Copy index.csv and filtered_gpu_size.csv so the merger can resolve
    # compound identity and assign score=0 to GPU-filtered molecules.
    import shutil
    if index_csv and index_csv.exists() and not (out / "index.csv").exists():
        shutil.copy2(index_csv, out / "index.csv")
    if filtered_csv and filtered_csv.exists() and not (out / "filtered_gpu_size.csv").exists():
        shutil.copy2(filtered_csv, out / "filtered_gpu_size.csv")

    _recover_scores(out)
    _enrich_scores_with_identity(out)
    _tag_scores_with_receptor_variant(out, receptor_variant)
    _tag_scores_with_engine(out, engine)
    _write_results_meta(out, engine)
    _write_single_unscored_reasons(out)
    return out


def run_screening_job(
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    notebook_path: Path,
    username: str,
    work_dir: Path,
    retry_limit: int = 3,
    accelerator: str = "nvidiaTeslaP100",
    engine: str = "unidock",
    kaggle_json_path: str | Path | None = None,
    job_observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: upload dataset → push kernel → poll → download output.
    Returns: {status, output_dir|None, error_type|None}
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    def observe(event: str, **details: Any) -> None:
        if job_observer is not None:
            job_observer(
                event,
                {
                    "run_id": run_id,
                    "work_dir": str(work_dir),
                    "notebook_path": str(notebook_path),
                    "engine": engine,
                    "kaggle_json_path": (
                        str(kaggle_json_path) if kaggle_json_path is not None else None
                    ),
                    **details,
                },
            )

    observe("preparing")

    # Persist paths so resume_failed_shards can find them later
    receptor_pdb = receptor_pdbqt.parent / "receptor_prep.pdb"
    resume_info = {
        "receptor_pdbqt": str(receptor_pdbqt),
        "receptor_pdb": str(receptor_pdb) if receptor_pdb.exists() else None,
        "shard_paths": [str(p) for p in shard_paths],
        "notebook_path": str(notebook_path),
        "username": username,
    }
    (work_dir / "resume.json").write_text(json.dumps(resume_info, indent=2))

    console.print(f"  [dim]Uploading run data ({len(shard_paths)} shard(s))...[/dim]")
    dataset_ref = upload_run_dataset(
        run_id=run_id,
        receptor_pdbqt=receptor_pdbqt,
        shard_paths=shard_paths,
        username=username,
        work_dir=work_dir,
    )
    console.print(f"  [dim]Dataset: {dataset_ref}[/dim]")

    # dataset_status() flips to "ready" before compute nodes consistently see
    # the uploaded files. Keep probing readiness before submitting the kernel.
    _wait_for_dataset(dataset_ref)
    _wait_for_dataset_sync(dataset_ref)

    console.print("  [dim]Submitting notebook to Kaggle...[/dim]")
    kernel_ref = push_kernel(
        run_id=run_id,
        notebook_path=notebook_path,
        dataset_ref=dataset_ref,
        username=username,
        work_dir=work_dir,
        accelerator=accelerator,
    )
    console.print(f"  [dim]Kernel: {kernel_ref}[/dim]")
    observe("submitted", kernel_ref=kernel_ref, retry_count=0)

    retry_count = 0
    while retry_count <= retry_limit:
        result = poll_until_done(
            kernel_ref=kernel_ref,
            run_id=run_id,
            retry_limit=retry_limit - retry_count,
            cred_lock=_KAGGLE_API_LOCK if kaggle_json_path is not None else None,
            kj_path=str(kaggle_json_path) if kaggle_json_path is not None else None,
        )

        if result["status"] == "complete":
            console.print("  [green]Run complete — downloading results...[/green]")
            output_dir = _download_output(
                kernel_ref,
                work_dir,
                kj_path=str(kaggle_json_path) if kaggle_json_path is not None else None,
                engine=engine,
            )
            observe(
                "complete",
                kernel_ref=kernel_ref,
                output_dir=str(output_dir),
                ended_epoch=time.time(),
            )
            return {
                "status": "complete",
                "output_dir": output_dir,
                "error_type": None,
                "kernel_ref": kernel_ref,
            }

        if result["status"] == "retry":
            retry_count = result["retry_count"]
            console.print(f"  [yellow]⟳ Resubmitting — retry {retry_count}/{retry_limit}[/yellow]")
            kernel_ref = push_kernel(
                run_id=f"{run_id}-r{retry_count}",
                notebook_path=notebook_path,
                dataset_ref=dataset_ref,
                username=username,
                work_dir=work_dir,
                accelerator=accelerator,
            )
            observe("submitted", kernel_ref=kernel_ref, retry_count=retry_count)
            continue

        # failed or timeout. Recheck under the owning account before giving up;
        # Kaggle can report transient/private-kernel errors even after completion.
        recovered_output = _download_if_kernel_completed_after_poll_failure(
            kernel_ref=kernel_ref,
            nb_work=work_dir,
            kj_path=kaggle_json_path,
            engine=engine,
        )
        if recovered_output is not None:
            observe(
                "complete",
                kernel_ref=kernel_ref,
                output_dir=str(recovered_output),
                ended_epoch=time.time(),
                recovered_after_poll_failure=True,
            )
            return {
                "status": "complete",
                "output_dir": recovered_output,
                "error_type": None,
                "recovered_after_poll_failure": True,
                "kernel_ref": kernel_ref,
            }
        observe(
            str(result["status"]),
            kernel_ref=kernel_ref,
            error_type=result["error_type"],
            ended_epoch=time.time(),
        )
        return {
            "status": result["status"],
            "output_dir": None,
            "error_type": result["error_type"],
            "kernel_ref": kernel_ref,
        }

    observe(
        "failed",
        kernel_ref=kernel_ref,
        error_type="max_retries_exceeded",
        ended_epoch=time.time(),
    )
    return {
        "status": "failed",
        "output_dir": None,
        "error_type": "max_retries_exceeded",
        "kernel_ref": kernel_ref,
    }


def _apply_account(account: dict | None) -> str | None:
    """Point kaggle API at a specific team account; returns original username env var."""
    if not account:
        return None
    kj_path = Path(account.get("kaggle_json_path", "")).expanduser()
    if not kj_path.exists():
        return None
    import json as _json
    data = _json.loads(kj_path.read_text())
    os.environ["KAGGLE_USERNAME"] = data.get("username", "")
    os.environ["KAGGLE_KEY"]      = data.get("key", "")
    return account.get("username")


def _resume_one_shard(
    run_id: str,
    shard_index: int,
    receptor_pdbqt: Path,
    shard_path: Path,
    notebook_path: Path,
    username: str,
    work_dir: Path,
    retry_limit: int,
    ck_lock: threading.Lock,
    account: dict | None = None,
    account_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    from ezscreen import checkpoint

    sub_run_id  = f"{run_id}-s{shard_index:03d}-resume"
    sub_work    = work_dir / f"resume_shard_{shard_index:03d}"
    effective_u = username
    try:
        # Switch to team account credentials if provided; hold account lock to
        # prevent two threads from simultaneously overwriting os.environ creds.
        lock_ctx = account_lock if account_lock is not None else threading.Lock()
        with lock_ctx:
            alt_user = _apply_account(account)
            if alt_user:
                effective_u = alt_user

        result = run_screening_job(
            run_id=sub_run_id,
            receptor_pdbqt=receptor_pdbqt,
            shard_paths=[shard_path],
            notebook_path=notebook_path,
            username=effective_u,
            work_dir=sub_work,
            retry_limit=retry_limit,
            kaggle_json_path=account.get("kaggle_json_path") if account else None,
        )
        new_status = "done" if result["status"] == "complete" else "failed"
        with ck_lock:
            checkpoint.update_shard(run_id, shard_index, new_status)
        return {"shard_index": shard_index, "result": result}
    except Exception as exc:
        with ck_lock:
            checkpoint.update_shard(run_id, shard_index, "failed", str(exc))
        return {"shard_index": shard_index, "result": {"status": "failed", "output_dir": None}}


def resume_failed_shards(
    run_id: str,
    work_dir: Path,
    retry_limit: int = 3,
) -> dict[str, Any]:
    from ezscreen import checkpoint
    from ezscreen.results.merger import merge_shard_results

    checkpoint.init_db()
    failed = checkpoint.get_failed_shards(run_id)
    if not failed:
        return {"status": "nothing_to_resume", "n_shards": 0, "n_succeeded": 0}

    resume_json = work_dir / "resume.json"
    if not resume_json.exists():
        return {"status": "failed", "error": "resume.json not found — run predates resume support"}

    info = json.loads(resume_json.read_text())
    receptor_pdbqt = Path(info["receptor_pdbqt"])
    notebook_path  = Path(info["notebook_path"])
    username       = info["username"]
    shard_dir      = work_dir / "shards"

    from ezscreen import auth as _auth
    accounts     = _auth.get_all_kaggle_accounts()
    acct_locks   = {a["name"]: threading.Lock() for a in accounts}

    ck_lock = threading.Lock()
    futures_map: dict = {}

    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_KERNELS) as pool:
        for i, shard in enumerate(failed):
            idx        = shard["shard_index"]
            shard_file = shard_dir / f"shard_{idx:03d}.pdbqt"
            if not shard_file.exists():
                console.print(f"  [yellow]Shard {idx}: file not found, skipping[/yellow]")
                continue
            # Round-robin account selection
            account     = accounts[i % len(accounts)] if len(accounts) > 1 else None
            acct_lock   = acct_locks[account["name"]] if account else None
            fut = pool.submit(
                _resume_one_shard,
                run_id, idx, receptor_pdbqt, shard_file,
                notebook_path, username, work_dir, retry_limit,
                ck_lock, account, acct_lock,
            )
            futures_map[fut] = idx

        outcomes = [fut.result() for fut in as_completed(futures_map)]

    # Merge new partial results with existing output
    completed_output_dirs = [
        work_dir / f"resume_shard_{o['shard_index']:03d}" / "output"
        for o in outcomes
        if o["result"]["status"] == "complete"
    ]
    existing_output = work_dir / "output"
    if completed_output_dirs:
        all_dirs = [existing_output] + completed_output_dirs
        merge_shard_results(all_dirs, existing_output)
        console.print(f"  [dim]Merged results from {len(completed_output_dirs)} resumed shard(s)[/dim]")

    n_succeeded = sum(1 for o in outcomes if o["result"]["status"] == "complete")
    overall     = "complete" if n_succeeded == len(failed) else ("partial" if n_succeeded else "failed")

    if overall in ("complete", "partial"):
        with ck_lock:
            checkpoint.mark_run_complete(run_id) if overall == "complete" else None

    return {"status": overall, "n_shards": len(failed), "n_succeeded": n_succeeded}


def _set_account_creds(kaggle_json_path: str | Path) -> str:
    """Switch Kaggle credentials; caller must hold _KAGGLE_API_LOCK.

    Sets env vars then immediately re-authenticates the SDK singleton so
    subsequent API calls use these credentials without a second authenticate().
    """
    import kaggle
    data = json.loads(Path(kaggle_json_path).expanduser().read_text())
    username = data["username"]
    key      = data["key"]
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"]      = key
    kaggle.api.authenticate()
    return username



def _kernel_status_with_account(
    kernel_ref: str,
    *,
    cred_lock: threading.Lock | None = None,
    kj_path: str | Path | None = None,
) -> tuple[str, str]:
    """Return normalized Kaggle kernel status using the account that owns it."""
    import kaggle

    def _call() -> tuple[str, str]:
        if kj_path is not None:
            _set_account_creds(kj_path)
        else:
            kaggle.api.authenticate()
        resp = kaggle.api.kernels_status(kernel_ref)
        status = str(getattr(resp, "status", "unknown")).split(".")[-1].lower()
        message = getattr(resp, "failure_message", "") or ""
        return status, message

    if cred_lock is not None:
        with cred_lock:
            return _call()
    return _call()


def _download_if_kernel_completed_after_poll_failure(
    *,
    kernel_ref: str,
    nb_work: Path,
    kj_path: str | Path | None = None,
    index_csv: Path | None = None,
    filtered_csv: Path | None = None,
    receptor_variant: str | None = None,
    engine: str = "unidock",
) -> Path | None:
    """Recover output when poller classified a kernel as failed but Kaggle now says complete."""
    try:
        status, message = _kernel_status_with_account(
            kernel_ref,
            cred_lock=_KAGGLE_API_LOCK if kj_path is not None else None,
            kj_path=kj_path,
        )
    except Exception as exc:
        console.print(f"  [dim]Recovery status check failed for {kernel_ref}: {exc}[/dim]")
        return None

    if status != "complete":
        detail = f" ({message})" if message else ""
        console.print(f"  [dim]Recovery status for {kernel_ref}: {status}{detail}[/dim]")
        return None

    console.print(f"  [yellow]Recovered completed kernel after poller failure: {kernel_ref}[/yellow]")
    try:
        return _download_output(
            kernel_ref,
            nb_work,
            kj_path=str(kj_path) if kj_path is not None else None,
            index_csv=index_csv,
            filtered_csv=filtered_csv,
            receptor_variant=receptor_variant,
            engine=engine,
        )
    except Exception as exc:
        console.print(f"  [yellow]Recovery download failed for {kernel_ref}: {exc}[/yellow]")
        return None

def render_vina_notebook(
    *,
    run_id: str,
    shard_filenames: list[str],
    notebook_index: int = 0,
    total_notebooks: int = 1,
    box_center: list[float] | None = None,
    box_size: list[float] | None = None,
    search_mode: str = "balance",
    exhaustiveness: int | None = None,
    ph: float = 7.4,
    prep_on_kaggle: bool = True,
    accelerator: str = "nvidiaTeslaP100",
    force_field: str | None = None,
    enumerate_opts: dict | None = None,
    prep_cfg: dict | None = None,
    results_cfg: dict | None = None,
    engine: str = "unidock",
    scoring: str = "vina",
    receptor_filename: str = "receptor.pdbqt",
    receptor_variant: str = "default",
    receptor_variants: list[dict[str, Any]] | None = None,
    receptor_pdb_filename: str = "receptor_fixed.pdb",
    ad4_maps_prefix: str | None = None,
) -> str:
    """Render the docking notebook to source text for a single kernel.

    Shared by the multi-account runner and the headless driver so the template var
    set lives in one place.  Loads [prep]/[results] config when not supplied.
    The template is chosen from the engine's registry profile (UniDock variants
    share vina_shard and switch on the `engine` template var).
    """
    import jinja2

    from ezscreen import __version__
    from ezscreen.backends import engines as _engines

    template_name = _engines.get(engine).kaggle_template or "vina_shard.ipynb.j2"

    if prep_cfg is None or results_cfg is None:
        try:
            from ezscreen import config as _cfg
            _full = _cfg.load()
            if prep_cfg is None:
                prep_cfg = _full.get("prep", {})
            if results_cfg is None:
                results_cfg = _full.get("results", {})
        except Exception:
            prep_cfg = prep_cfg or {}
            results_cfg = results_cfg or {}

    _eo = enumerate_opts or {}
    _enum_enabled = bool(_eo.get("enabled", prep_cfg.get("enumerate_enabled", False)))
    _variant_specs = receptor_variants or [{
        "name": receptor_variant,
        "filename": receptor_filename,
        "prepare_on_kaggle": receptor_variant != "default",
    }]
    _variant_specs = [
        {
            "name": str(v.get("name", "default")),
            "filename": str(v.get("filename", "receptor.pdbqt")),
            "prepare_on_kaggle": bool(v.get("prepare_on_kaggle", False)),
        }
        for v in _variant_specs
    ]

    template_path = Path(__file__).parent / "templates" / template_name
    j2_env = jinja2.Environment(
        variable_start_string="<<", variable_end_string=">>",
        block_start_string="<%",    block_end_string="%>",
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
    )
    return j2_env.get_template(template_path.name).render(
        ezscreen_version=__version__,
        run_id=run_id,
        engine=engine,
        scoring=scoring,
        mode="hybrid",
        box_center=box_center or [],
        box_size=box_size or [],
        notebook_index=notebook_index,
        total_notebooks=total_notebooks,
        shard_filenames=shard_filenames,
        ph=ph,
        search_mode=search_mode,
        exhaustiveness=exhaustiveness or 0,
        prep_on_kaggle=prep_on_kaggle,
        max_heavy_atoms=int(prep_cfg.get("max_heavy_atoms", 70)),
        max_mw=float(prep_cfg.get("max_mw", 700.0)),
        max_rotatable_bonds=int(prep_cfg.get("max_rotatable_bonds", 20)),
        mmff_max_iters=int(prep_cfg.get("mmff_max_iters", 0)),
        force_field=force_field or str(prep_cfg.get("force_field", "MMFF94")),
        normalization_profile=str(prep_cfg.get("normalization_profile", "legacy")),
        enumerate_enabled=_enum_enabled,
        enumerate_protonation=bool(_eo.get("protonation", prep_cfg.get("enumerate_protonation", True))),
        enumerate_tautomers=bool(_eo.get("tautomers", prep_cfg.get("enumerate_tautomers", True))),
        enumerate_stereo=bool(_eo.get("stereo", prep_cfg.get("enumerate_stereo", False))),
        enumerate_ring=bool(_eo.get("ring", prep_cfg.get("enumerate_ring", False))),
        max_variants_per_ligand=int(_eo.get("max_variants", prep_cfg.get("max_variants_per_ligand", 4))),
        poses_returned=int(results_cfg.get("poses_returned", 25)),
        receptor_filename=receptor_filename,
        receptor_variant=receptor_variant,
        receptor_variant_specs_py=repr(_variant_specs),
        receptor_prepare_on_kaggle=any(v.get("prepare_on_kaggle") for v in _variant_specs),
        receptor_pdb_filename=receptor_pdb_filename,
        ad4_maps_prefix=ad4_maps_prefix or "",
        gpu_ids="0,1" if accelerator == "nvidiaTeslaT4" else "",
    )



def _receptor_variant_specs(
    receptor_pdbqt: Path,
    receptor_pdbqts: list[Path] | None = None,
    receptor_variant_modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return receptor variant specs with stable dataset filenames."""
    if receptor_variant_modes:
        specs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for mode in receptor_variant_modes:
            if mode in seen:
                continue
            seen.add(mode)
            specs.append({
                "name": mode,
                "filename": "receptor.pdbqt",
                "path": receptor_pdbqt,
                "prepare_on_kaggle": mode != "default",
            })
        return specs

    import hashlib

    paths = [Path(p) for p in (receptor_pdbqts or [receptor_pdbqt])]
    if not paths:
        paths = [receptor_pdbqt]
    specs = []
    seen_hashes: set[str] = set()
    for path in paths:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        idx = len(specs)
        filename = "receptor.pdbqt" if idx == 0 else f"receptor_variant_{idx:03d}.pdbqt"
        name = "default" if idx == 0 else path.stem.replace("receptor_", "") or f"variant_{idx:03d}"
        specs.append({
            "name": name,
            "filename": filename,
            "path": path,
            "prepare_on_kaggle": False,
        })
    return specs

def run_multi_account_screening(
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    account_assignments: list[dict],
    work_dir: Path,
    receptor_pdbqts: list[Path] | None = None,
    receptor_variant_modes: list[str] | None = None,
    receptor_pdb: Path | None = None,
    box_center: list[float] | None = None,
    box_size: list[float] | None = None,
    search_mode: str = "balance",
    exhaustiveness: int | None = None,
    ph: float = 7.4,
    retry_limit: int = 3,
    accelerator: str = "nvidiaTeslaP100",
    prep_on_kaggle: bool = True,
    force_field: str | None = None,
    enumerate_opts: dict | None = None,
    engine: str = "unidock",
    scoring: str = "vina",
    extra_files: dict[str, Path] | None = None,
    ad4_maps_prefix: str | None = None,
) -> dict[str, Any]:
    """Submit all shards to multiple Kaggle accounts and merge results.

    account_assignments — list of {account: {name, kaggle_json_path, username}, shard_count: int}
    shard_count means notebooks per account (0 = 1).  All shard files are uploaded
    in one dataset per account; shards are distributed evenly across all notebooks.
    Kernels are submitted sequentially (env-var safety), then polled concurrently.
    """
    n_shards = len(shard_paths)

    # shard_count semantics: None = auto (1 notebook), 0 = excluded, N > 0 = explicit
    account_assignments = [a for a in account_assignments if a["shard_count"] != 0]
    if not account_assignments:
        return {"status": "failed", "error": "No active accounts — all accounts were set to 0 (excluded)"}

    notebook_counts = [a["shard_count"] or 1 for a in account_assignments]
    base_notebooks = sum(notebook_counts)
    receptor_variants = _receptor_variant_specs(
        receptor_pdbqt,
        receptor_pdbqts,
        receptor_variant_modes,
    )
    if scoring == "ad4":
        if engine != "unidock":
            return {"status": "failed", "error_type": "ad4_requires_unidock"}
        if not ad4_maps_prefix or not extra_files:
            return {"status": "failed", "error_type": "ad4_map_bundle_required"}
        if len(receptor_variants) != 1 or receptor_variants[0].get("name") != "default":
            return {"status": "failed", "error_type": "ad4_requires_single_default_receptor"}
        _extra_names = set(extra_files)
        _ad4_required = {f"{ad4_maps_prefix}.maps.fld", "ad4_maps_manifest.json"}
        if not _ad4_required <= _extra_names:
            return {"status": "failed", "error_type": "ad4_map_bundle_incomplete"}
    if any(v.get("prepare_on_kaggle") for v in receptor_variants):
        if receptor_pdb is None or not Path(receptor_pdb).exists():
            return {
                "status": "failed",
                "output_dir": None,
                "error_type": "receptor_pdb_required_for_kaggle_variant_prep",
            }
    total_notebooks = base_notebooks
    total_variant_jobs = base_notebooks * len(receptor_variants)

    console.print(
        f"  [dim]{n_shards} shard(s) -> {total_notebooks} notebook(s) across "
        f"{len(account_assignments)} active account(s); "
        f"{len(receptor_variants)} receptor variant(s) packed per notebook "
        f"({total_variant_jobs} docking job(s))[/dim]"
    )

    # Distribute shards in contiguous blocks across the user-requested notebook
    # groups. Each notebook loops over all receptor variants internally so the
    # user-selected Kaggle kernel count remains literal.
    base  = n_shards // base_notebooks
    extra = n_shards % base_notebooks
    nb_shard_groups: list[list[Path]] = []
    cursor = 0
    for i in range(base_notebooks):
        count = base + (1 if i < extra else 0)
        nb_shard_groups.append(shard_paths[cursor:cursor + count])
        cursor += count
    # Build per-account spec: collect the shards needed and notebook metadata
    account_specs: list[dict] = []
    nb_offset = 0
    for acct_i, (assignment, nb_count) in enumerate(zip(account_assignments, notebook_counts)):
        account = assignment["account"]
        notebooks: list[dict] = []
        acct_shard_paths: list[Path] = []
        seen: set = set()
        for nb_j in range(nb_count):
            nb_idx = nb_offset + nb_j
            group = nb_shard_groups[nb_idx]
            notebooks.append({
                "notebook_index": nb_idx,
                "shard_filenames": [p.name for p in group],
                "receptor_filename": receptor_variants[0]["filename"],
                "receptor_variant": receptor_variants[0]["name"],
                "receptor_variants": receptor_variants,
            })
            for p in group:
                if p not in seen:
                    acct_shard_paths.append(p)
                    seen.add(p)
        nb_offset += nb_count

        sub_run_id = f"{run_id}-acct{acct_i}"
        sub_work   = work_dir / f"acct_{acct_i}"
        sub_work.mkdir(parents=True, exist_ok=True)
        account_specs.append({
            "account":         account,
            "kj_path":         account["kaggle_json_path"],
            "shard_paths":     acct_shard_paths,
            "notebooks":       notebooks,
            "sub_run_id":      sub_run_id,
            "sub_work":        sub_work,
        })
        console.print(
            f"  [dim]  {account['name']}: {len(acct_shard_paths)} shard(s), "
            f"{nb_count} notebook(s)[/dim]"
        )

    # Upload one dataset per account (all its shards in one go)
    for spec in account_specs:
        kj_path    = spec["kj_path"]
        account    = spec["account"]
        sub_run_id = spec["sub_run_id"]
        sub_work   = spec["sub_work"]
        with _KAGGLE_API_LOCK:
            username = _set_account_creds(kj_path)
            spec["username"] = username
            console.print(f"  [dim]Uploading {len(spec['shard_paths'])} shard(s) for {account['name']}...[/dim]")
            dataset_ref = upload_run_dataset(
                run_id=sub_run_id,
                receptor_pdbqt=receptor_pdbqt,
                shard_paths=spec["shard_paths"],
                username=username,
                work_dir=sub_work,
                receptor_files={
                    v["filename"]: v["path"]
                    for v in receptor_variants
                    if not v.get("prepare_on_kaggle")
                },
                receptor_pdb=receptor_pdb,
                extra_files=extra_files,
            )
            _wait_for_dataset(dataset_ref)
        spec["dataset_ref"] = dataset_ref
        console.print(f"  [dim]Dataset ready: {dataset_ref}[/dim]")

    # Keep probing for a stable ready state across all uploaded datasets before
    # the sequential kernel submissions start.
    _wait_for_dataset_sync(
        [str(spec["dataset_ref"]) for spec in account_specs],
        label="all dataset files",
    )

    # Render and submit each notebook sequentially (env-var safety)
    try:
        from ezscreen import config as _cfg
        _full_cfg = _cfg.load()
        _prep_cfg = _full_cfg.get("prep", {})
        _results_cfg = _full_cfg.get("results", {})
    except Exception:
        _prep_cfg = {}
        _results_cfg = {}

    submissions: list[dict] = []
    for spec in account_specs:
        kj_path     = spec["kj_path"]
        account     = spec["account"]
        dataset_ref = spec["dataset_ref"]
        sub_work    = spec["sub_work"]
        username    = spec["username"]
        for nb_spec in spec["notebooks"]:
            nb_idx      = nb_spec["notebook_index"]
            filenames   = nb_spec["shard_filenames"]
            receptor_filename = nb_spec.get("receptor_filename", "receptor.pdbqt")
            receptor_variant = nb_spec.get("receptor_variant", "default")
            nb_run_id   = f"{spec['sub_run_id']}-nb{nb_idx}"
            nb_dir      = sub_work / f"nb_{nb_idx}"
            nb_dir.mkdir(parents=True, exist_ok=True)

            nb_src = render_vina_notebook(
                run_id=nb_run_id,
                shard_filenames=filenames,
                notebook_index=nb_idx,
                total_notebooks=total_notebooks,
                box_center=box_center,
                box_size=box_size,
                search_mode=search_mode,
                exhaustiveness=exhaustiveness,
                ph=ph,
                prep_on_kaggle=prep_on_kaggle,
                accelerator=accelerator,
                force_field=force_field,
                enumerate_opts=enumerate_opts,
                prep_cfg=_prep_cfg,
                results_cfg=_results_cfg,
                engine=engine,
                scoring=scoring,
                receptor_filename=receptor_filename,
                receptor_variant=receptor_variant,
                receptor_variants=nb_spec.get("receptor_variants"),
                receptor_pdb_filename="receptor_fixed.pdb",
                ad4_maps_prefix=ad4_maps_prefix,
            )
            nb_path = nb_dir / "notebook.ipynb"
            nb_path.write_text(nb_src)

            try:
                with _KAGGLE_API_LOCK:
                    _set_account_creds(kj_path)
                    kernel_ref = push_kernel(
                        run_id=nb_run_id,
                        notebook_path=nb_path,
                        dataset_ref=dataset_ref,
                        username=username,
                        work_dir=nb_dir,
                        accelerator=accelerator,
                    )
            except Exception as _push_exc:
                console.print(f"  [red]Kernel push failed for {nb_run_id}: {_push_exc}[/red]")
                continue
            console.print(
                f"  [dim]Kernel submitted: {kernel_ref}  "
                f"[{account['name']} nb{nb_idx + 1}/{total_notebooks}][/dim]"
            )
            submissions.append({
                "account":    account,
                "kj_path":    kj_path,
                "kernel_ref": kernel_ref,
                "nb_work":    nb_dir,
                "nb_run_id":  nb_run_id,
                "receptor_variant": receptor_variant,
            })

    # Poll all kernels concurrently
    def _poll_one(sub: dict) -> dict:
        result = poll_until_done(
            sub["kernel_ref"], sub["nb_run_id"], retry_limit,
            cred_lock=_KAGGLE_API_LOCK, kj_path=sub["kj_path"],
            show_live=False,
        )
        return {"sub": sub, "result": result}

    poll_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(submissions)) as pool:
        futures = [pool.submit(_poll_one, s) for s in submissions]
        for fut in as_completed(futures):
            try:
                poll_results.append(fut.result())
            except Exception as exc:
                console.print(f"  [red]Poller thread crashed: {exc}[/red]")

    # Log per-notebook outcome before downloading
    console.print("  [dim]Notebook results:[/dim]")
    for pr in poll_results:
        sub    = pr["sub"]
        result = pr["result"]
        icon   = "[green]✓[/green]" if result["status"] == "complete" else "[red]✗[/red]"
        console.print(
            f"  {icon} {sub['kernel_ref']}  "
            f"status={result['status']}  error={result.get('error_type')}"
        )

    # Download all completed notebooks concurrently
    _index_csv    = work_dir / "shards" / "index.csv"
    _filtered_csv = work_dir / "shards" / "filtered_gpu_size.csv"

    def _download_one(pr: dict) -> Path | None:
        sub = pr["sub"]
        if pr["result"]["status"] != "complete":
            return _download_if_kernel_completed_after_poll_failure(
                kernel_ref=sub["kernel_ref"],
                nb_work=sub["nb_work"],
                kj_path=sub["kj_path"],
                index_csv=_index_csv,
                filtered_csv=_filtered_csv,
                receptor_variant=sub.get("receptor_variant"),
                engine=engine,
            )
        try:
            return _download_output(
                sub["kernel_ref"], sub["nb_work"],
                kj_path=sub["kj_path"],
                index_csv=_index_csv,
                filtered_csv=_filtered_csv,
                receptor_variant=sub.get("receptor_variant"),
                engine=engine,
            )
        except Exception as exc:
            console.print(f"  [yellow]Download failed for {sub['kernel_ref']}: {exc}[/yellow]")
            return None

    console.print("  [dim]Downloading results...[/dim]")
    with ThreadPoolExecutor(max_workers=len(poll_results)) as pool:
        dl_results = list(pool.map(_download_one, poll_results))
    output_dirs = [p for p in dl_results if p is not None]

    if not output_dirs:
        return {"status": "failed", "output_dir": None, "error_type": "all_accounts_failed"}

    from ezscreen.results.merger import merge_shard_results
    main_output = work_dir / "output"
    merge_shard_results(output_dirs, main_output, engine=engine)
    console.print(
        f"  [green]Multi-account run complete — merged "
        f"{len(output_dirs)}/{len(submissions)} notebook result(s)[/green]"
    )

    status = "complete" if len(output_dirs) == len(submissions) else "partial"
    return {"status": status, "output_dir": main_output, "error_type": None}


def clean_run(run_id: str, username: str) -> None:
    """Delete all Kaggle artifacts for a run (dataset + kernel)."""
    from ezscreen.backends.kaggle.dataset import delete_run_dataset
    delete_run_dataset(run_id, username)
    delete_kernel(run_id, username)
    console.print(f"  [dim]Cleaned Kaggle artifacts for {run_id}[/dim]")
