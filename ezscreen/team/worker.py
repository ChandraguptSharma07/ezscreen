from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ezscreen.backends.kaggle.kernel import push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done
from ezscreen.team.client import TeamClient
from ezscreen.team.kaggle_upload_ticket import (
    UploadTicketUnavailable,
    commit_dataset_from_blob_tokens,
    mint_upload_ticket,
)

_DATASET_POLL_INTERVAL = 15
_DATASET_TIMEOUT = 10 * 60
_KERNEL_RETRY_LIMIT = 2


def _api():
    import kaggle

    kaggle.api.authenticate()
    return kaggle.api


def _select_job(client: TeamClient, worker_job_id: str | None = None) -> dict[str, Any]:
    jobs = client.request("GET", "/v1/worker/jobs").get("jobs", [])
    if not jobs:
        raise SystemExit("No active worker jobs are assigned to this token.")
    if worker_job_id:
        for job in jobs:
            if job.get("worker_job_id") == worker_job_id:
                return job
        raise SystemExit(f"Worker job is not assigned to this token: {worker_job_id}")
    if len(jobs) > 1:
        raise SystemExit("Multiple worker jobs are assigned; pass --worker-job-id.")
    return jobs[0]


def _kaggle_dataset_slug(team_job_id: str, worker_job_id: str) -> str:
    raw = f"ezscreen-{team_job_id}-{worker_job_id}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or "ezscreen-job"


def _kaggle_dataset_title(dataset_slug: str) -> str:
    return f"ezscreen {dataset_slug}"[:50]


def _kaggle_kernel_run_id(team_job_id: str, worker_job_id: str, attempt: int = 0) -> str:
    base = f"ezs-team-{team_job_id[-8:]}-{worker_job_id[-8:]}"
    if attempt:
        base = f"{base}-r{attempt}"
    return re.sub(r"[^a-z0-9-]+", "-", base.lower()).strip("-")[:50]


def _post_failed(client: TeamClient, worker_job_id: str, *, error_type: str, message: str | None = None, extra: dict[str, Any] | None = None) -> int:
    payload = {"error_type": error_type}
    if message:
        payload["message"] = message
    if extra:
        payload.update(extra)
    print(f"ezscreen worker failed: {payload}", file=sys.stderr)
    client.request(
        "POST",
        "/v1/worker/status",
        {
            "worker_job_id": worker_job_id,
            "status": "failed",
            "payload": payload,
        },
    )
    return 2


def _wait_for_dataset(dataset_ref: str) -> None:
    api = _api()
    elapsed = 0
    while elapsed < _DATASET_TIMEOUT:
        try:
            if api.dataset_status(dataset_ref) == "ready":
                return
        except Exception:
            pass
        import time
        time.sleep(_DATASET_POLL_INTERVAL)
        elapsed += _DATASET_POLL_INTERVAL


def _download_dataset_file(dataset_ref: str, filename: str, dest_dir: Path) -> Path:
    api = _api()
    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(dataset_ref, filename, path=str(dest_dir), force=True, quiet=True)
    direct = dest_dir / filename
    if direct.exists():
        return direct
    for candidate in dest_dir.rglob(filename):
        if candidate.is_file():
            return candidate
    zipped = dest_dir / f"{filename}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(dest_dir)
        if direct.exists():
            return direct
        for candidate in dest_dir.rglob(filename):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"dataset file not found after download: {filename}")


def _download_kernel_output_zip(kernel_ref: str, work_dir: Path) -> Path:
    api = _api()
    out_dir = work_dir / "kernel_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel_ref, path=str(out_dir), quiet=True)
    for candidate in out_dir.rglob("output.zip"):
        if candidate.is_file():
            return candidate
    zip_path = out_dir / "output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if not path.is_file() or path == zip_path:
                continue
            zf.write(path, path.relative_to(out_dir))
    return zip_path


def _finalize_kernel_run(client: TeamClient, job: dict[str, Any], *, kernel_ref: str, kernel_run_id: str) -> dict[str, Any]:
    result = poll_until_done(kernel_ref=kernel_ref, run_id=kernel_run_id, retry_limit=_KERNEL_RETRY_LIMIT, show_live=False)
    if result["status"] != "complete":
        return result
    with tempfile.TemporaryDirectory(prefix="ezscreen-team-worker-") as tmp:
        zip_path = _download_kernel_output_zip(kernel_ref, Path(tmp))
        client.upload_result_zip(worker_job_id=job["worker_job_id"], data=zip_path.read_bytes())
    return {"status": "complete", "retry_count": result.get("retry_count", 0), "error_type": None}


def prepare_upload(args: argparse.Namespace, client: TeamClient) -> int:
    job = _select_job(client, args.worker_job_id)
    try:
        tickets = []
        for file in job["required_files"]:
            ticket = mint_upload_ticket(file["filename"], int(file.get("size_bytes") or 0))
            tickets.append(ticket.as_dict())
    except UploadTicketUnavailable as exc:
        return _post_failed(
            client,
            job["worker_job_id"],
            error_type="upload_ticket_unavailable",
            message=str(exc),
        )
    client.request(
        "POST",
        "/v1/worker/upload-tickets",
        {"worker_job_id": job["worker_job_id"], "tickets": tickets},
    )
    return 0


def run_kaggle(args: argparse.Namespace, client: TeamClient) -> int:
    job = _select_job(client, args.worker_job_id)
    try:
        tickets = client.request(
            "GET",
            f"/v1/worker/upload-tickets?worker_job_id={job['worker_job_id']}",
        ).get("tickets", [])
        if not tickets:
            return _post_failed(client, job["worker_job_id"], error_type="upload_tickets_missing")
        files = [
            {"filename": ticket["filename"], "blob_token": ticket["blob_token"]}
            for ticket in tickets
        ]
        dataset_slug = _kaggle_dataset_slug(job["team_job_id"], job["worker_job_id"])
        dataset_ref = commit_dataset_from_blob_tokens(
            username=os.environ["KAGGLE_USERNAME"],
            slug=dataset_slug,
            title=_kaggle_dataset_title(dataset_slug),
            files=files,
        )
        _wait_for_dataset(dataset_ref)
        with tempfile.TemporaryDirectory(prefix="ezscreen-team-worker-") as tmp:
            work_dir = Path(tmp)
            notebook_path = _download_dataset_file(dataset_ref, "team_notebook.ipynb", work_dir)
            accelerator = str(job.get("payload", {}).get("accelerator") or "nvidiaTeslaP100")
            for attempt in range(_KERNEL_RETRY_LIMIT + 1):
                kernel_run_id = _kaggle_kernel_run_id(job["team_job_id"], job["worker_job_id"], attempt)
                kernel_ref = push_kernel(
                    run_id=kernel_run_id,
                    notebook_path=notebook_path,
                    dataset_ref=dataset_ref,
                    username=os.environ["KAGGLE_USERNAME"],
                    work_dir=work_dir,
                    accelerator=accelerator,
                )
                running_payload = {
                    "dataset_ref": dataset_ref,
                    "kernel_ref": kernel_ref,
                    "kernel_run_id": kernel_run_id,
                    "retry_attempt": attempt,
                }
                client.request(
                    "POST",
                    "/v1/worker/status",
                    {
                        "worker_job_id": job["worker_job_id"],
                        "status": "running",
                        "payload": running_payload,
                    },
                )
                result = _finalize_kernel_run(client, job, kernel_ref=kernel_ref, kernel_run_id=kernel_run_id)
                if result["status"] == "retry" and attempt < _KERNEL_RETRY_LIMIT:
                    continue
                if result["status"] == "complete":
                    return 0
                return _post_failed(
                    client,
                    job["worker_job_id"],
                    error_type=str(result.get("error_type") or "kernel_failed"),
                    message=f"Kaggle kernel {result['status']}",
                    extra=running_payload,
                )
    except UploadTicketUnavailable as exc:
        return _post_failed(
            client,
            job["worker_job_id"],
            error_type="upload_ticket_unavailable",
            message=str(exc),
        )
    except Exception as exc:
        return _post_failed(
            client,
            job["worker_job_id"],
            error_type="worker_runtime_error",
            message=str(exc),
        )
    return _post_failed(client, job["worker_job_id"], error_type="kernel_retry_exhausted")


def collect(args: argparse.Namespace, client: TeamClient) -> int:
    job = _select_job(client, args.worker_job_id)
    payload = dict(job.get("payload") or {})
    kernel_ref = str(payload.get("kernel_ref") or "").strip()
    kernel_run_id = str(payload.get("kernel_run_id") or "").strip()
    dataset_ref = str(payload.get("dataset_ref") or "").strip()
    try:
        if not kernel_ref:
            if not dataset_ref:
                return _post_failed(client, job["worker_job_id"], error_type="kernel_ref_missing")
            with tempfile.TemporaryDirectory(prefix="ezscreen-team-worker-") as tmp:
                work_dir = Path(tmp)
                notebook_path = _download_dataset_file(dataset_ref, "team_notebook.ipynb", work_dir)
                accelerator = str(job.get("payload", {}).get("accelerator") or "nvidiaTeslaP100")
                kernel_run_id = _kaggle_kernel_run_id(job["team_job_id"], job["worker_job_id"], int(payload.get("retry_attempt") or 0))
                kernel_ref = push_kernel(
                    run_id=kernel_run_id,
                    notebook_path=notebook_path,
                    dataset_ref=dataset_ref,
                    username=os.environ["KAGGLE_USERNAME"],
                    work_dir=work_dir,
                    accelerator=accelerator,
                )
                payload.update({"dataset_ref": dataset_ref, "kernel_ref": kernel_ref, "kernel_run_id": kernel_run_id})
        client.request(
            "POST",
            "/v1/worker/status",
            {"worker_job_id": job["worker_job_id"], "status": "collecting", "payload": payload},
        )
        result = _finalize_kernel_run(client, job, kernel_ref=kernel_ref, kernel_run_id=kernel_run_id or job["worker_job_id"])
    except Exception as exc:
        return _post_failed(
            client,
            job["worker_job_id"],
            error_type="collect_failed",
            message=str(exc),
            extra=payload,
        )
    if result["status"] == "complete":
        return 0
    return _post_failed(
        client,
        job["worker_job_id"],
        error_type=str(result.get("error_type") or "collect_failed"),
        message=f"Kaggle kernel {result['status']}",
        extra=payload,
    )


def auto(args: argparse.Namespace, client: TeamClient) -> int:
    jobs = client.request("GET", "/v1/worker/jobs").get("jobs", [])
    rc = 0
    for job in jobs:
        status = str(job.get("status") or "")
        job_args = argparse.Namespace(worker_job_id=job.get("worker_job_id"))
        if status == "needs_upload_tickets":
            rc = max(rc, prepare_upload(job_args, client))
        elif status == "queued_run":
            rc = max(rc, run_kaggle(job_args, client))
        elif status in {"running", "collecting"}:
            rc = max(rc, collect(job_args, client))
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ezscreen team GitHub Actions worker")
    parser.add_argument("--mode", choices=["auto", "prepare_upload", "run_kaggle", "collect"], default="auto")
    parser.add_argument("--worker-job-id", default=os.environ.get("EZSCREEN_WORKER_JOB_ID", ""))
    parser.add_argument("--server-url", default=os.environ.get("EZSCREEN_TEAM_SERVER_URL", ""))
    parser.add_argument("--token", default=os.environ.get("EZSCREEN_TEAM_TOKEN", ""))
    args = parser.parse_args(argv)
    if not args.server_url:
        raise SystemExit("EZSCREEN_TEAM_SERVER_URL is required.")
    if not args.token:
        raise SystemExit("EZSCREEN_TEAM_TOKEN is required.")
    args.worker_job_id = args.worker_job_id or None
    client = TeamClient(args.server_url, token=args.token)
    if args.mode == "auto":
        return auto(args, client)
    if args.mode == "prepare_upload":
        return prepare_upload(args, client)
    if args.mode == "run_kaggle":
        return run_kaggle(args, client)
    return collect(args, client)


if __name__ == "__main__":
    sys.exit(main())


