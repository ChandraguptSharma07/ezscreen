"""Launch ADA preparation or docking on private Kaggle Tesla T4 kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ezscreen import auth
from ezscreen.backends.kaggle.dataset import (
    _bundle_fingerprint,
    dataset_slug,
    upload_files_dataset,
)
from ezscreen.backends.kaggle.kernel import push_kernel
from ezscreen.backends.kaggle.runner import (
    _KAGGLE_API_LOCK,
    _set_account_creds,
    _wait_for_dataset,
    _wait_for_dataset_sync,
)
from ezscreen.benchmark import supervisor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"
WORK = HERE / "work"
SOURCE_MANIFEST = (
    ROOT / "benchmarks/bench-20260723-vinardo-dud28/source_manifest.json"
)
PREPARED_BUNDLE = HERE / "inputs/prepared_bundle.zip"
PREPARED_MANIFEST = HERE / "prepared_input_manifest.json"
ACCELERATOR = "nvidiaTeslaT4"


def healthy_accounts() -> list[tuple[dict, str, Path]]:
    import kaggle

    found = []
    for account in auth.get_all_kaggle_accounts():
        credential = Path(account["kaggle_json_path"]).expanduser().resolve()
        try:
            data = auth.validate_kaggle_json(credential)
            with _KAGGLE_API_LOCK:
                _set_account_creds(credential)
                kaggle.api.kernels_list(page=1, page_size=1, mine=True)
            found.append((account, data["username"], credential))
        except Exception as exc:
            print(
                {
                    "account": account.get("name", "unnamed"),
                    "healthy": False,
                    "error": type(exc).__name__,
                }
            )
    return found


def dataset_for(
    username: str,
    credential: Path,
    files: list[Path],
    *,
    prefix: str,
    title: str,
) -> tuple[str, bool]:
    import kaggle

    slug = dataset_slug(f"{prefix}-{_bundle_fingerprint(files)}")
    ref = f"{username}/{slug}"
    with _KAGGLE_API_LOCK:
        _set_account_creds(credential)
        try:
            kaggle.api.dataset_status(ref)
            return ref, True
        except Exception:
            ref = upload_files_dataset(prefix, files, username, WORK, title=title)
            _wait_for_dataset(ref)
            _wait_for_dataset_sync(ref, label=f"{prefix} {username}")
            return ref, False


def active_jobs_for_account(state: dict, credential: Path) -> int:
    active = {"submitted", "queued", "running", "polling", "recovery_pending"}
    return sum(
        job.get("kaggle_json_path") == str(credential)
        and job.get("status") in active
        for job in state.get("jobs", [])
    )


def register_and_push(
    *,
    account: dict,
    username: str,
    credential: Path,
    run_id: str,
    stage: str,
    notebook: Path,
    dataset_ref: str,
    dataset_reused: bool,
    shard_index: int,
    shard_count: int,
    planned_gpu_hours: float,
) -> None:
    current = json.loads((HERE / "campaign_state.json").read_text())
    prior = next(
        (job for job in current.get("jobs", []) if job["job_id"] == run_id), None
    )
    if prior and prior.get("status") in {
        "submitted",
        "queued",
        "running",
        "complete",
    }:
        print({"run_id": run_id, "action": "skip", "status": prior["status"]})
        return
    if active_jobs_for_account(current, credential) >= 2:
        raise RuntimeError(f"two active kernels already assigned to {username}")
    job_work = WORK / run_id
    retry_count = int(prior.get("retry_count") or 0) + 1 if prior else 0
    supervisor.register_job(
        HERE,
        {
            "job_id": run_id,
            "run_id": run_id,
            "stage": stage,
            "target": "ada",
            "pdb_id": "1ndw",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "compute_kind": "gpu",
            "planned_gpu_hours": planned_gpu_hours,
            "account": str(account.get("name") or username),
            "username": username,
            "kaggle_json_path": str(credential),
            "notebook_path": str(notebook),
            "work_dir": str(job_work),
            "accelerator": ACCELERATOR,
            "required_gpu_model": "Tesla T4",
            "required_gpu_count": 2,
            "recovery_kind": "raw_kernel",
            "dataset_ref": dataset_ref,
            "dataset_reused": dataset_reused,
            "retry_count": retry_count,
        },
    )
    with _KAGGLE_API_LOCK:
        _set_account_creds(credential)
        kernel_ref = push_kernel(
            run_id,
            notebook,
            dataset_ref,
            username,
            job_work,
            accelerator=ACCELERATOR,
        )
    supervisor.update_registered_job(
        HERE, run_id, status="submitted", kernel_ref=kernel_ref
    )
    print({"run_id": run_id, "kernel_ref": kernel_ref, "status": "submitted"})


def launch_preparation(accounts: list[tuple[dict, str, Path]]) -> None:
    notebook = NOTEBOOKS / "ada_1ndw_prepare_t4.ipynb"
    files = [SOURCE_MANIFEST, HERE / "target_evidence.json", HERE / "campaign_protocol.json"]
    if not notebook.is_file() or any(not path.is_file() for path in files):
        raise FileNotFoundError("preparation notebook or manifest missing")
    account, username, credential = accounts[0]
    dataset_ref, reused = dataset_for(
        username,
        credential,
        files,
        prefix="ada-1ndw-prep-t4-inputs",
        title="ADA 1NDW DUD2 T4 preparation inputs",
    )
    register_and_push(
        account=account,
        username=username,
        credential=credential,
        run_id="ada-1ndw-prepare-t4",
        stage="preparation",
        notebook=notebook,
        dataset_ref=dataset_ref,
        dataset_reused=reused,
        shard_index=0,
        shard_count=1,
        planned_gpu_hours=1.0,
    )


def launch_docking(accounts: list[tuple[dict, str, Path]]) -> None:
    notebooks = sorted(NOTEBOOKS.glob("ada_1ndw_vina_t4_shard_*.ipynb"))
    if len(notebooks) != 12:
        raise RuntimeError(f"expected 12 docking notebooks, got {len(notebooks)}")
    files = [PREPARED_BUNDLE, PREPARED_MANIFEST]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("frozen prepared input bundle is missing")
    for account_index, (account, username, credential) in enumerate(accounts[:6]):
        dataset_ref, reused = dataset_for(
            username,
            credential,
            files,
            prefix="ada-1ndw-vina-t4-inputs",
            title="ADA 1NDW exact ADT-prepared Vina T4 inputs",
        )
        for shard in (account_index * 2, account_index * 2 + 1):
            register_and_push(
                account=account,
                username=username,
                credential=credential,
                run_id=f"ada-1ndw-vina-t4-s{shard:02d}",
                stage="docking",
                notebook=notebooks[shard],
                dataset_ref=dataset_ref,
                dataset_reused=reused,
                shard_index=shard,
                shard_count=12,
                planned_gpu_hours=1.5,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prep", "dock"))
    args = parser.parse_args()
    accounts = healthy_accounts()
    if len(accounts) < 6:
        raise RuntimeError(f"need six healthy accounts; found {len(accounts)}")
    WORK.mkdir(parents=True, exist_ok=True)
    supervisor.initialize_campaign_state(HERE)
    if args.stage == "prep":
        launch_preparation(accounts)
    else:
        launch_docking(accounts)
    state = json.loads((HERE / "campaign_state.json").read_text())
    print({"jobs": len(state.get("jobs", [])), "status": state.get("status")})


if __name__ == "__main__":
    main()
