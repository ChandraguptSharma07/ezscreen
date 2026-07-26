from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any

from ezscreen.results.merger import merge_shard_results
from ezscreen.team.client import TeamClient
from ezscreen.team.kaggle_upload_ticket import upload_file_to_signed_url
from ezscreen.team.state import extract_result_zip


def _file_spec(path: Path, role: str) -> dict[str, Any]:
    return {"filename": path.name, "size_bytes": path.stat().st_size, "role": role}


def stage_team_job_receptor_files(
    *,
    work_dir: Path,
    receptor_pdbqt: Path,
    receptor_pdbqts: list[Path] | None = None,
    receptor_variant_modes: list[str] | None = None,
    receptor_pdb: Path | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    from ezscreen.backends.kaggle.runner import _receptor_variant_specs

    staged_dir = work_dir / 'team_receptor_dataset'
    staged_dir.mkdir(parents=True, exist_ok=True)
    variant_specs = _receptor_variant_specs(
        Path(receptor_pdbqt),
        [Path(p) for p in (receptor_pdbqts or [receptor_pdbqt])],
        receptor_variant_modes,
    )
    staged_paths: list[Path] = []
    seen_names: set[str] = set()
    for spec in variant_specs:
        filename = str(spec['filename'])
        source = Path(spec['path'])
        target = staged_dir / filename
        if filename not in seen_names:
            shutil.copy2(source, target)
            staged_paths.append(target)
            seen_names.add(filename)
    if any(bool(spec.get('prepare_on_kaggle')) for spec in variant_specs):
        if receptor_pdb is None or not Path(receptor_pdb).exists():
            raise ValueError('receptor_pdb is required when Kaggle-side receptor variant prep is enabled')
        target = staged_dir / 'receptor_fixed.pdb'
        shutil.copy2(Path(receptor_pdb), target)
        staged_paths.append(target)
    return staged_paths, variant_specs


def start_team_job(
    *,
    client: TeamClient,
    user_id: str,
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    notebook_path: Path,
    engine: str,
    accelerator: str = "nvidiaTeslaP100",
    team_member_assignments: list[dict[str, Any]] | None = None,
    receptor_files: list[Path] | None = None,
) -> dict[str, Any]:
    upload_receptor_files = [Path(p) for p in (receptor_files or [receptor_pdbqt])]
    files = [
        _file_spec(path, 'receptor_pdb' if path.suffix.lower() == '.pdb' else 'receptor')
        for path in upload_receptor_files
    ]
    files.append(_file_spec(notebook_path, 'notebook'))
    files.extend(_file_spec(path, 'ligand_shard') for path in shard_paths)
    return client.request(
        'POST',
        '/v1/jobs',
        {
            'user_id': user_id,
            'run_id': run_id,
            'payload': {
                'engine': engine,
                'accelerator': accelerator,
                'files': files,
                'team_member_assignments': team_member_assignments or [],
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
        tickets = client.request('GET', f'/v1/jobs/{team_job_id}/tickets?user_id={user_id}').get('tickets', [])
        if len(tickets) >= expected_ticket_count:
            break
        time.sleep(3)
    if len(tickets) < expected_ticket_count:
        raise TimeoutError(
            f'timed out waiting for worker upload tickets ({len(tickets)}/{expected_ticket_count} ready)'
        )
    for ticket in tickets:
        path = by_name.get(ticket['filename'])
        if path is None:
            continue
        upload_file_to_signed_url(path, ticket['upload_url'])
        client.request('POST', '/v1/uploads/ack', {'user_id': user_id, 'ticket_id': ticket['ticket_id']})
    return tickets


def _dispatch_team_job_mode(
    *,
    client: TeamClient,
    user_id: str,
    team_job: dict[str, Any],
    mode: str,
) -> dict[str, list[dict[str, str]]]:
    return client.dispatch_team_job(user_id=user_id, team_job_id=team_job['team_job_id'], mode=mode)


def dispatch_team_job_workers(
    *,
    client: TeamClient,
    user_id: str,
    team_job: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    return _dispatch_team_job_mode(client=client, user_id=user_id, team_job=team_job, mode='prepare_upload')


def dispatch_team_job_runs(
    *,
    client: TeamClient,
    user_id: str,
    team_job: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    return _dispatch_team_job_mode(client=client, user_id=user_id, team_job=team_job, mode='run_kaggle')


def wait_for_team_job(
    *,
    client: TeamClient,
    user_id: str,
    team_job_id: str,
    poll_interval_s: int = 15,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = client.status(user_id)
        for job in status.get('jobs', []):
            if job.get('team_job_id') != team_job_id:
                continue
            if job.get('status') in {'complete', 'failed'}:
                return job
            break
        time.sleep(poll_interval_s)
    raise TimeoutError(f'timed out waiting for team job {team_job_id} to finish')


def download_team_result_zips(
    *,
    client: TeamClient,
    user_id: str,
    team_job: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []
    for worker_job in team_job.get('worker_jobs', []):
        worker_job_id = str(worker_job['worker_job_id'])
        zip_path = output_dir / f'{worker_job_id}.zip'
        zip_path.write_bytes(
            client.download_result_zip(
                user_id=user_id,
                team_job_id=team_job['team_job_id'],
                worker_job_id=worker_job_id,
            )
        )
        result_paths.append(zip_path)
    return result_paths


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _namespace_colliding_worker_outputs(shard_dirs: list[Path]) -> None:
    owners: dict[str, list[int]] = {}
    dir_maps: list[dict[str, str]] = []
    for shard_index, shard_dir in enumerate(shard_dirs):
        _, index_rows = _read_csv_rows(shard_dir / 'index.csv')
        rename_map: dict[str, str] = {}
        for row in index_rows:
            lig_id = str(row.get('ligand', '')).strip()
            if not lig_id:
                continue
            owners.setdefault(lig_id, []).append(shard_index)
        dir_maps.append(rename_map)

    for lig_id, shard_indexes in owners.items():
        if len(shard_indexes) <= 1:
            continue
        for shard_index in shard_indexes:
            dir_maps[shard_index][lig_id] = f'worker_{shard_index:03d}_{lig_id}'

    for shard_index, shard_dir in enumerate(shard_dirs):
        rename_map = dir_maps[shard_index]
        if not rename_map:
            continue
        for csv_name, id_col in (
            ('index.csv', 'ligand'),
            ('scores.csv', 'ligand'),
            ('unscored_reasons.csv', 'ligand'),
            ('cnn_scores.csv', 'lig_id'),
        ):
            path = shard_dir / csv_name
            fieldnames, rows = _read_csv_rows(path)
            if not fieldnames or not rows or id_col not in fieldnames:
                continue
            changed = False
            for row in rows:
                key = str(row.get(id_col, '')).strip()
                new_key = rename_map.get(key)
                if not new_key:
                    continue
                row[id_col] = new_key
                changed = True
            if changed:
                _write_csv_rows(path, fieldnames, rows)

        poses_path = shard_dir / 'poses.sdf'
        if poses_path.exists():
            text = poses_path.read_text(encoding='utf-8')
            for old, new in rename_map.items():
                text = text.replace(old, new)
            poses_path.write_text(text, encoding='utf-8')


def merge_team_result_zips(result_zips: list[Path], output_dir: Path, *, engine: str | None = None) -> dict[str, Any]:
    shard_dirs = []
    extract_root = output_dir.parent / 'team_worker_outputs'
    for index, zip_path in enumerate(result_zips):
        shard_dirs.append(extract_result_zip(zip_path, extract_root / f'worker_{index:03d}'))
    _namespace_colliding_worker_outputs(shard_dirs)
    return merge_shard_results(shard_dirs, output_dir, engine=engine)
