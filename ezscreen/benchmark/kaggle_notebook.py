from __future__ import annotations

from pathlib import Path
from typing import Any


def _healthy_account(account_name: str | None = None) -> tuple[dict[str, Any], str]:
    import kaggle

    from ezscreen import auth
    from ezscreen.backends.kaggle.runner import _KAGGLE_API_LOCK, _set_account_creds

    failures: list[tuple[str, str]] = []
    for account in auth.get_all_kaggle_accounts():
        label = str(account.get("name") or "unnamed")
        if account_name and label.lower() != account_name.lower():
            continue
        credential_path = Path(str(account["kaggle_json_path"])).expanduser()
        try:
            data = auth.validate_kaggle_json(credential_path)
            with _KAGGLE_API_LOCK:
                _set_account_creds(credential_path)
                kaggle.api.kernels_list(page=1, page_size=1, mine=True)
            return account, str(data["username"])
        except Exception as exc:
            failures.append((label, type(exc).__name__))
    if account_name and not failures:
        raise ValueError(f"Unknown Kaggle account label: {account_name}")
    raise RuntimeError(f"No healthy Kaggle account; failures={failures}")


def submit_notebook(
    campaign_dir: Path,
    notebook_path: Path,
    input_paths: list[Path],
    *,
    run_id: str,
    job_id: str | None = None,
    stage: str = "smoke",
    backend: str = "cpu",
    account_name: str | None = None,
    planned_gpu_hours: float = 0.0,
    resume_blocked: bool = False,
    dataset_ref: str | None = None,
    supersedes_job_ids: list[str] | None = None,
    finalizer: str | None = None,
    intended_csv: Path | None = None,
) -> dict[str, Any]:
    """Submit a custom benchmark notebook through the public user workflow."""
    from ezscreen.backends.kaggle.dataset import upload_files_dataset
    from ezscreen.backends.kaggle.kernel import accelerator_for_backend, push_kernel
    from ezscreen.backends.kaggle.runner import (
        _KAGGLE_API_LOCK,
        _set_account_creds,
        _wait_for_dataset,
        _wait_for_dataset_sync,
    )
    from ezscreen.benchmark import supervisor

    campaign_dir = campaign_dir.expanduser().resolve()
    notebook_path = notebook_path.expanduser().resolve()
    inputs = [path.expanduser().resolve() for path in input_paths]
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if not inputs:
        raise ValueError("At least one --input is required")
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Notebook input not found: {missing[0]}")
    reused_dataset_ref = str(dataset_ref or "").strip()
    if reused_dataset_ref and (
        reused_dataset_ref.count("/") != 1
        or any(char.isspace() for char in reused_dataset_ref)
    ):
        raise ValueError("--dataset-ref must be a Kaggle owner/slug reference")
    finalizer_key = str(finalizer or "").strip()
    if finalizer_key and finalizer_key not in {"bayesbind-vina", "bayesbind-gnina"}:
        raise ValueError("--finalizer must be bayesbind-vina or bayesbind-gnina")
    intended_path = intended_csv.expanduser().resolve() if intended_csv else None
    if finalizer_key and intended_path is None:
        raise ValueError("--intended-csv is required when --finalizer is set")
    if intended_path is not None and not intended_path.is_file():
        raise FileNotFoundError(f"Intended CSV not found: {intended_path}")
    accelerator = accelerator_for_backend(backend)
    compute_kind = "cpu" if backend.lower() == "cpu" else "gpu"
    planned = float(planned_gpu_hours)
    if compute_kind == "cpu" and planned != 0.0:
        raise ValueError("CPU backend must reserve zero GPU hours")
    if compute_kind == "gpu" and planned <= 0.0:
        raise ValueError("GPU backend requires positive planned GPU hours")
    if resume_blocked:
        supervisor.resume_blocked_campaign(
            campaign_dir,
            reason=f"authorized notebook repair: {run_id}",
        )

    account, username = _healthy_account(account_name)
    label = str(account.get("name") or "unnamed")
    credential_path = str(
        Path(str(account["kaggle_json_path"])).expanduser().resolve()
    )
    work_dir = campaign_dir / "work" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    job_key = job_id or run_id
    for superseded_job_id in supersedes_job_ids or []:
        supervisor.update_registered_job(
            campaign_dir,
            superseded_job_id,
            status="superseded",
            superseded_by=job_key,
        )
    job_payload: dict[str, Any] = {
        "job_id": job_key,
        "run_id": run_id,
        "stage": stage,
        "compute_kind": compute_kind,
        "planned_gpu_hours": planned,
        "account": label,
        "username": username,
        "kaggle_json_path": credential_path,
        "notebook_path": str(notebook_path),
        "work_dir": str(work_dir),
        "accelerator": accelerator,
        "backend": backend,
        "recovery_kind": "raw_kernel",
        "dataset_reused": bool(reused_dataset_ref),
    }
    if finalizer_key:
        job_payload["finalizer"] = finalizer_key
        job_payload["intended_csv"] = str(intended_path)
    supervisor.register_job(campaign_dir, job_payload)
    try:
        with _KAGGLE_API_LOCK:
            _set_account_creds(credential_path)
            if reused_dataset_ref:
                submitted_dataset_ref = reused_dataset_ref
            else:
                submitted_dataset_ref = upload_files_dataset(
                    run_id,
                    inputs,
                    username,
                    work_dir,
                    title=f"ezscreen {run_id}"[:50],
                )
            supervisor.update_registered_job(
                campaign_dir, job_key, dataset_ref=submitted_dataset_ref
            )
            if not reused_dataset_ref:
                _wait_for_dataset(submitted_dataset_ref)
                _wait_for_dataset_sync(
                    submitted_dataset_ref, label="notebook inputs"
                )
            kernel_ref = push_kernel(
                run_id,
                notebook_path,
                submitted_dataset_ref,
                username,
                work_dir,
                backend=backend,
            )
        supervisor.update_registered_job(
            campaign_dir,
            job_key,
            status="submitted",
            kernel_ref=kernel_ref,
        )
    except Exception as exc:
        supervisor.update_registered_job(
            campaign_dir,
            job_key,
            status="failed",
            submission_error=type(exc).__name__,
        )
        raise
    return {
        "status": "submitted",
        "job_id": job_key,
        "run_id": run_id,
        "kernel_ref": kernel_ref,
        "dataset_ref": submitted_dataset_ref,
        "dataset_reused": bool(reused_dataset_ref),
        "backend": backend,
        "compute_kind": compute_kind,
        "planned_gpu_hours": planned,
        "campaign_dir": str(campaign_dir),
    }
