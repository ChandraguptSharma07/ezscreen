"""Monitor and recover bounded ACE Vina T4 shard outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ezscreen.backends.kaggle.runner import _KAGGLE_API_LOCK, _set_account_creds
from ezscreen.benchmark import supervisor

HERE = Path(__file__).resolve().parent
PATTERN = (
    r"(^|/)(output\.zip|done\.flag|runtime\.json|hardware_manifest\.json|.*\.log)$"
)


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
                status = normalized(kaggle.api.kernels_status(job["kernel_ref"]))
                update = {"kernel_status": status}
                if status in {"complete", "error"}:
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
                    if (
                        status == "complete"
                        and (output / "output.zip").is_file()
                        and (output / "done.flag").is_file()
                    ):
                        update.update(status="complete", output_dir=str(output))
                    elif status == "complete":
                        update.update(status="recovery_pending", output_dir=str(output))
                    else:
                        update.update(status="failed", output_dir=str(output))
                elif status in {"running", "queued"}:
                    update.update(status=status)
                supervisor.update_registered_job(HERE, job["job_id"], **update)
                report.append({"job_id": job["job_id"], **update})
    current = json.loads((HERE / "campaign_state.json").read_text())
    terminal = sum(
        job.get("status") in {"complete", "failed", "blocked", "cancelled"}
        for job in current["jobs"]
    )
    print(
        json.dumps(
            {"checked": len(report), "terminal": terminal, "statuses": report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
