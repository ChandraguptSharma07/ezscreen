"""Fast account-grouped monitor and bounded raw-output downloader for ACE AD4."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ezscreen.backends.kaggle.runner import _KAGGLE_API_LOCK, _set_account_creds
from ezscreen.benchmark import supervisor

HERE = Path(__file__).resolve().parent
PATTERN = r"(^|/)(output\.zip|done\.flag|runtime\.json|.*\.log)$"


def normalized(status) -> str:
    value = getattr(status, "status", status)
    name = getattr(value, "name", str(value)).lower().replace("_", "")
    return {
        "running": "running",
        "queued": "queued",
        "complete": "complete",
        "error": "error",
        "cancelled": "cancelled",
    }.get(name, name)


def main() -> None:
    import kaggle

    state = json.loads((HERE / "campaign_state.json").read_text())
    grouped = defaultdict(list)
    for job in state["jobs"]:
        if job.get("status") in {
            "submitted",
            "queued",
            "running",
            "polling",
            "recovery_pending",
        }:
            grouped[job["kaggle_json_path"]].append(job)
    report = []
    for credential, jobs in grouped.items():
        with _KAGGLE_API_LOCK:
            _set_account_creds(credential)
            for job in jobs:
                response = kaggle.api.kernels_status(job["kernel_ref"])
                status = normalized(response)
                update = {"kernel_status": status}
                if status == "complete":
                    output = Path(job["work_dir"]) / "output"
                    output.mkdir(parents=True, exist_ok=True)
                    kaggle.api.kernels_output(
                        job["kernel_ref"],
                        str(output),
                        file_pattern=PATTERN,
                        force=False,
                        quiet=True,
                        page_size=100,
                    )
                    if (output / "output.zip").is_file() and (
                        output / "done.flag"
                    ).is_file():
                        update.update(status="complete", output_dir=str(output))
                    else:
                        update.update(status="recovery_pending")
                elif status == "error":
                    output = Path(job["work_dir"]) / "output"
                    output.mkdir(parents=True, exist_ok=True)
                    kaggle.api.kernels_output(
                        job["kernel_ref"],
                        str(output),
                        file_pattern=PATTERN,
                        force=False,
                        quiet=True,
                        page_size=100,
                    )
                    update.update(status="failed", output_dir=str(output))
                elif status in {"running", "queued"}:
                    update.update(status=status)
                supervisor.update_registered_job(HERE, job["job_id"], **update)
                report.append({"job_id": job["job_id"], **update})
    current = json.loads((HERE / "campaign_state.json").read_text())
    terminal = sum(
        j.get("status") in {"complete", "failed", "blocked", "cancelled"}
        for j in current["jobs"]
    )
    print(
        json.dumps(
            {"checked": len(report), "terminal": terminal, "statuses": report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
