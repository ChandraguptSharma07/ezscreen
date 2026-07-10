from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ezscreen.results.merger import merge_shard_results
from ezscreen.team.client import TeamClient
from ezscreen.team.kaggle_upload_ticket import upload_file_to_signed_url
from ezscreen.team.state import extract_result_zip


def _file_spec(path: Path, role: str) -> dict[str, Any]:
    return {"filename": path.name, "size_bytes": path.stat().st_size, "role": role}


def start_team_job(
    *,
    client: TeamClient,
    user_id: str,
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    notebook_path: Path,
    engine: str,
    team_member_assignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    files = [_file_spec(receptor_pdbqt, "receptor"), _file_spec(notebook_path, "notebook")]
    files.extend(_file_spec(path, "ligand_shard") for path in shard_paths)
    return client.request(
        "POST",
        "/v1/jobs",
        {
            "user_id": user_id,
            "run_id": run_id,
            "payload": {
                "engine": engine,
                "files": files,
                "team_member_assignments": team_member_assignments or [],
            },
        },
    )


def upload_job_files(
    *,
    client: TeamClient,
    user_id: str,
    team_job_id: str,
    file_paths: list[Path],
    expected_ticket_count: int,
    poll_timeout_s: int = 300,
) -> list[dict[str, Any]]:
    by_name = {path.name: path for path in file_paths}
    deadline = time.time() + poll_timeout_s
    tickets: list[dict[str, Any]] = []
    while time.time() < deadline:
        tickets = client.request("GET", f"/v1/jobs/{team_job_id}/tickets?user_id={user_id}").get("tickets", [])
        if len(tickets) >= expected_ticket_count:
            break
        time.sleep(3)
    if len(tickets) < expected_ticket_count:
        raise TimeoutError(
            f"timed out waiting for worker upload tickets ({len(tickets)}/{expected_ticket_count} ready)"
        )
    for ticket in tickets:
        path = by_name.get(ticket["filename"])
        if path is None:
            continue
        upload_file_to_signed_url(path, ticket["upload_url"])
        client.request("POST", "/v1/uploads/ack", {"user_id": user_id, "ticket_id": ticket["ticket_id"]})
    return tickets


def merge_team_result_zips(result_zips: list[Path], output_dir: Path, *, engine: str | None = None) -> dict[str, Any]:
    shard_dirs = []
    extract_root = output_dir.parent / "team_worker_outputs"
    for index, zip_path in enumerate(result_zips):
        shard_dirs.append(extract_result_zip(zip_path, extract_root / f"worker_{index:03d}"))
    return merge_shard_results(shard_dirs, output_dir, engine=engine)
