from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console

from ezscreen.backends.kaggle.dataset import upload_run_dataset
from ezscreen.backends.kaggle.kernel import delete_kernel, push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done

console = Console()
_DATASET_POLL_INTERVAL = 10   # seconds
_DATASET_TIMEOUT = 300        # 5 minutes
_MAX_PARALLEL_KERNELS = 2
_KAGGLE_API_LOCK = threading.Lock()  # serialises env-var switching + authenticate()


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

    # Persist paths so resume_failed_shards can find them later
    resume_info = {
        "receptor_pdbqt": str(receptor_pdbqt),
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


def run_multi_account_screening(
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    account_assignments: list[dict],
    work_dir: Path,
    box_center: list[float] | None = None,
    box_size: list[float] | None = None,
    search_mode: str = "balance",
    ph: float = 7.4,
    retry_limit: int = 3,
) -> dict[str, Any]:
    """Submit all shards to multiple Kaggle accounts and merge results.

    account_assignments — list of {account: {name, kaggle_json_path, username}, shard_count: int}
    shard_count means notebooks per account (0 = 1).  All shard files are uploaded
    in one dataset per account; shards are distributed evenly across all notebooks.
    Kernels are submitted sequentially (env-var safety), then polled concurrently.
    """
    import jinja2

    from ezscreen import __version__

    n_shards = len(shard_paths)

    # Notebook count per account (0 → 1)
    notebook_counts = [a["shard_count"] or 1 for a in account_assignments]
    total_notebooks = sum(notebook_counts)

    console.print(
        f"  [dim]{n_shards} shard(s) → {total_notebooks} notebook(s) "
        f"across {len(account_assignments)} account(s)[/dim]"
    )

    # Distribute shards in contiguous blocks across all notebooks
    base  = n_shards // total_notebooks
    extra = n_shards % total_notebooks
    nb_shard_groups: list[list[Path]] = []
    cursor = 0
    for i in range(total_notebooks):
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
            group  = nb_shard_groups[nb_idx]
            notebooks.append({"notebook_index": nb_idx, "shard_filenames": [p.name for p in group]})
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
            )
            _wait_for_dataset(dataset_ref)
        spec["dataset_ref"] = dataset_ref
        console.print(f"  [dim]Dataset ready: {dataset_ref}[/dim]")

    # Single propagation wait after all uploads
    console.print("  [dim]Waiting 90s for all datasets to sync to compute nodes...[/dim]")
    time.sleep(90)

    # Render and submit each notebook sequentially (env-var safety)
    template_path = Path(__file__).parent / "templates" / "vina_shard.ipynb.j2"
    j2_env = jinja2.Environment(
        variable_start_string="<<", variable_end_string=">>",
        block_start_string="<%",    block_end_string="%>",
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
    )

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
            nb_run_id   = f"{spec['sub_run_id']}-nb{nb_idx}"
            nb_dir      = sub_work / f"nb_{nb_idx}"
            nb_dir.mkdir(parents=True, exist_ok=True)

            nb_src = j2_env.get_template(template_path.name).render(
                ezscreen_version=__version__,
                run_id=nb_run_id,
                engine="unidock",
                mode="hybrid",
                box_center=box_center or [],
                box_size=box_size or [],
                notebook_index=nb_idx,
                total_notebooks=total_notebooks,
                shard_filenames=filenames,
                ph=ph,
                search_mode=search_mode,
                enumerate_tautomers=False,
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

    # Download and merge results
    output_dirs: list[Path] = []
    for pr in poll_results:
        sub    = pr["sub"]
        result = pr["result"]
        if result["status"] == "complete":
            with _KAGGLE_API_LOCK:
                _set_account_creds(sub["kj_path"])
            out = _download_output(sub["kernel_ref"], sub["nb_work"])
            output_dirs.append(out)

    if not output_dirs:
        return {"status": "failed", "output_dir": None, "error_type": "all_accounts_failed"}

    from ezscreen.results.merger import merge_shard_results
    main_output = work_dir / "output"
    merge_shard_results(output_dirs, main_output)
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
