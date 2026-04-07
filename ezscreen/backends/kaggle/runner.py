from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console

from ezscreen.backends.kaggle.dataset import upload_run_dataset
from ezscreen.backends.kaggle.kernel import push_kernel, delete_kernel
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


def _download_output(kernel_ref: str, work_dir: Path) -> Path:
    import kaggle
    kaggle.api.authenticate()
    out = work_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    kaggle.api.kernels_output(kernel_ref, path=str(out))
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
