from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console

from ezscreen.backends.kaggle.dataset import upload_run_dataset
from ezscreen.backends.kaggle.kernel import delete_kernel, push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done

console = Console()
_DATASET_POLL_INTERVAL = 10   # seconds
_DATASET_TIMEOUT = 300        # 5 minutes


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

    # Filter unphysical scores — UniDock GPU produces nonsensical values (<-15)
    # for very small/flexible molecules due to GPU scoring artifacts.
    # AutoDock Vina-family scores never go below -15 for real binding events.
    _SCORE_FLOOR = -15.0
    filtered = [r for r in rows if r["score"] >= _SCORE_FLOOR]
    n_artifacts = len(rows) - len(filtered)
    if n_artifacts:
        console.print(f"  [dim]Filtered {n_artifacts} artifact score(s) below {_SCORE_FLOOR} kcal/mol[/dim]")
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
    with scores_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    console.print(f"  [dim]Enriched scores.csv with compound identity ({enriched} rows)[/dim]")


def _download_output(kernel_ref: str, work_dir: Path, retries: int = 5) -> Path:
    import kaggle
    kaggle.api.authenticate()
    out = work_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            kaggle.api.kernels_output(kernel_ref, path=str(out))
            break
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            console.print(f"  [yellow]Download error (attempt {attempt + 1}/{retries}), retrying in {wait}s: {exc}[/yellow]")
            time.sleep(wait)
    _recover_scores(out)
    _enrich_scores_with_identity(out)
    return out


def run_screening_job(
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    notebook_path: Path,
    username: str,
    work_dir: Path,
    retry_limit: int = 3,
) -> dict[str, Any]:
    """
    Full pipeline: upload dataset → push kernel → poll → download output.
    Returns: {status, output_dir|None, error_type|None}
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"  [dim]Uploading run data ({len(shard_paths)} shard(s))...[/dim]")
    dataset_ref = upload_run_dataset(
        run_id=run_id,
        receptor_pdbqt=receptor_pdbqt,
        shard_paths=shard_paths,
        username=username,
        work_dir=work_dir,
    )
    console.print(f"  [dim]Dataset: {dataset_ref}[/dim]")

    # dataset_status() returns "ready" as soon as metadata is saved, but files
    # take ~60s to propagate to compute nodes.  Pushing the kernel too early
    # causes Cell 4's path assertions to fail with FileNotFoundError.
    _wait_for_dataset(dataset_ref)
    console.print("  [dim]Waiting 90s for dataset files to sync to compute nodes...[/dim]")
    time.sleep(90)

    console.print("  [dim]Submitting notebook to Kaggle...[/dim]")
    kernel_ref = push_kernel(
        run_id=run_id,
        notebook_path=notebook_path,
        dataset_ref=dataset_ref,
        username=username,
        work_dir=work_dir,
    )
    console.print(f"  [dim]Kernel: {kernel_ref}[/dim]")

    retry_count = 0
    while retry_count <= retry_limit:
        result = poll_until_done(
            kernel_ref=kernel_ref,
            run_id=run_id,
            retry_limit=retry_limit - retry_count,
        )

        if result["status"] == "complete":
            console.print("  [green]Run complete — downloading results...[/green]")
            output_dir = _download_output(kernel_ref, work_dir)
            return {"status": "complete", "output_dir": output_dir, "error_type": None}

        if result["status"] == "retry":
            retry_count = result["retry_count"]
            console.print(f"  [yellow]⟳ Resubmitting — retry {retry_count}/{retry_limit}[/yellow]")
            kernel_ref = push_kernel(
                run_id=f"{run_id}-r{retry_count}",
                notebook_path=notebook_path,
                dataset_ref=dataset_ref,
                username=username,
                work_dir=work_dir,
            )
            continue

        # failed or timeout
        return {"status": result["status"], "output_dir": None, "error_type": result["error_type"]}

    return {"status": "failed", "output_dir": None, "error_type": "max_retries_exceeded"}


def clean_run(run_id: str, username: str) -> None:
    """Delete all Kaggle artifacts for a run (dataset + kernel)."""
    from ezscreen.backends.kaggle.dataset import delete_run_dataset
    delete_run_dataset(run_id, username)
    delete_kernel(run_id, username)
    console.print(f"  [dim]Cleaned Kaggle artifacts for {run_id}[/dim]")
