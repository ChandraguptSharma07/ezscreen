"""Launch two ACE Vina T4 validation shards on each of six Kaggle accounts."""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"
WORK = HERE / "work"
SOURCE_MANIFEST = ROOT / "benchmarks/bench-20260723-vinardo-dud28/source_manifest.json"
EXACT_RECEPTOR = (
    ROOT
    / "benchmarks/bench-20260727-ace-ad4zn-probe/unidock_ad4_smoke_020/dataset_ace-1o86-unidock-ad4-smoke-020/receptor.pdbqt"
)
PLANNED_HOURS_PER_SHARD = 2.0
ACCELERATOR = "nvidiaTeslaT4"


def healthy_accounts() -> list[tuple[dict, str, Path]]:
    import kaggle

    found = []
    for account in auth.get_all_kaggle_accounts():
        path = Path(account["kaggle_json_path"]).expanduser().resolve()
        try:
            data = auth.validate_kaggle_json(path)
            with _KAGGLE_API_LOCK:
                _set_account_creds(path)
                kaggle.api.kernels_list(page=1, page_size=1, mine=True)
            found.append((account, data["username"], path))
        except Exception as exc:
            print(
                {
                    "account": account.get("name", "unnamed"),
                    "healthy": False,
                    "error": type(exc).__name__,
                }
            )
    return found


def dataset_for(username: str, credential: Path, files: list[Path]) -> tuple[str, bool]:
    import kaggle

    slug = dataset_slug(f"ace-vina-t4-inputs-{_bundle_fingerprint(files)}")
    ref = f"{username}/{slug}"
    with _KAGGLE_API_LOCK:
        _set_account_creds(credential)
        try:
            kaggle.api.dataset_status(ref)
            return ref, True
        except Exception:
            ref = upload_files_dataset(
                "ace-vina-t4-inputs",
                files,
                username,
                WORK,
                title="ACE 1O86 DUD2 Vina T4 validation inputs",
            )
            _wait_for_dataset(ref)
            _wait_for_dataset_sync(ref, label=f"ACE Vina T4 inputs {username}")
            return ref, False


def main() -> None:
    notebooks = sorted(NOTEBOOKS.glob("ace_vina_t4_shard_*.ipynb"))
    if len(notebooks) != 12:
        raise RuntimeError(f"expected 12 notebooks, got {len(notebooks)}")
    accounts = healthy_accounts()
    if len(accounts) < 6:
        raise RuntimeError(f"need six healthy accounts; found {len(accounts)}")
    files = [SOURCE_MANIFEST, EXACT_RECEPTOR]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    WORK.mkdir(parents=True, exist_ok=True)
    supervisor.initialize_campaign_state(HERE)

    for account_index, (account, username, credential) in enumerate(accounts[:6]):
        dataset_ref, reused = dataset_for(username, credential, files)
        for shard in (account_index * 2, account_index * 2 + 1):
            run_id = f"ace-1o86-vina-t4-s{shard:02d}"
            current = json.loads((HERE / "campaign_state.json").read_text())
            prior = next(
                (job for job in current["jobs"] if job["job_id"] == run_id), None
            )
            if prior and prior.get("status") in {
                "submitted",
                "queued",
                "running",
                "complete",
            }:
                print({"run_id": run_id, "action": "skip", "status": prior["status"]})
                continue
            job_work = WORK / run_id
            retry_count = int(prior.get("retry_count") or 0) + 1 if prior else 0
            supervisor.register_job(
                HERE,
                {
                    "job_id": run_id,
                    "run_id": run_id,
                    "stage": "full_vina_t4_validation",
                    "target": "ace",
                    "pdb_id": "1o86",
                    "shard_index": shard,
                    "shard_count": 12,
                    "compute_kind": "gpu",
                    "planned_gpu_hours": PLANNED_HOURS_PER_SHARD,
                    "account": str(account.get("name") or username),
                    "username": username,
                    "kaggle_json_path": str(credential),
                    "notebook_path": str(notebooks[shard]),
                    "work_dir": str(job_work),
                    "accelerator": ACCELERATOR,
                    "required_gpu_model": "Tesla T4",
                    "required_gpu_count": 2,
                    "recovery_kind": "raw_kernel",
                    "dataset_ref": dataset_ref,
                    "dataset_reused": reused,
                    "retry_count": retry_count,
                },
            )
            with _KAGGLE_API_LOCK:
                _set_account_creds(credential)
                kernel_ref = push_kernel(
                    run_id,
                    notebooks[shard],
                    dataset_ref,
                    username,
                    job_work,
                    accelerator=ACCELERATOR,
                )
            supervisor.update_registered_job(
                HERE, run_id, status="submitted", kernel_ref=kernel_ref
            )
            print({"run_id": run_id, "kernel_ref": kernel_ref, "status": "submitted"})
    state = json.loads((HERE / "campaign_state.json").read_text())
    print({"jobs": len(state["jobs"]), "status": state["status"]})


if __name__ == "__main__":
    main()
