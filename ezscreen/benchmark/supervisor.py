from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

STATE_FILE = 'campaign_state.json'
EVENT_FILE = 'execution_events.jsonl'
QUOTA_FILE = 'quota_budget.json'
QUOTA_POINTER_FILE = 'quota_epoch_active.json'
TRANSIENT_ERRORS = {'preempted', 'gpu_oom', 'timeout', 'transient_network'}
TERMINAL_STATES = {'complete', 'failed', 'blocked', 'cancelled', 'superseded'}


@contextmanager
def _campaign_lock(campaign_dir: Path, timeout_seconds: float = 60.0):
    """Serialize cross-process campaign read/modify/write operations."""
    campaign_dir = campaign_dir.expanduser().resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    handle = (campaign_dir / 'campaign_state.lock').open('a+b')
    if handle.seek(0, 2) == 0:
        handle.write(b'0')
        handle.flush()
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt

            while not acquired:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('campaign state lock timed out')
                    time.sleep(0.1)
        else:
            import fcntl

            while not acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('campaign state lock timed out')
                    time.sleep(0.1)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _locked_campaign_operation(function):
    @wraps(function)
    def wrapped(campaign_dir: Path, *args, **kwargs):
        with _campaign_lock(Path(campaign_dir)):
            return function(campaign_dir, *args, **kwargs)

    return wrapped


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def _contains_secret(payload: Any, path: str = '') -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {'key', 'token', 'secret', 'password', 'api_key'}:
                if value not in ('', None, False):
                    return f'{path}.{key}'.lstrip('.')
            found = _contains_secret(value, f'{path}.{key}')
            if found:
                return found
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found = _contains_secret(value, f'{path}[{index}]')
            if found:
                return found
    return None


def _persist(path: Path, payload: dict) -> None:
    secret_path = _contains_secret(payload)
    if secret_path:
        raise ValueError(f'Refusing to persist secret field: {secret_path}')
    _write_json(path, payload)


def _event(campaign_dir: Path, event: str, **details: object) -> None:
    payload = {'timestamp': _now(), 'event': event, **details}
    secret_path = _contains_secret(payload)
    if secret_path:
        raise ValueError(f'Refusing to persist secret event field: {secret_path}')
    with (campaign_dir / EVENT_FILE).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def _active_quota_path(campaign_dir: Path) -> Path:
    pointer_path = campaign_dir / QUOTA_POINTER_FILE
    if not pointer_path.exists():
        return campaign_dir / QUOTA_FILE
    pointer = _load_json(pointer_path, {})
    quota_file = str(pointer.get('quota_file') or '').strip()
    active_epoch = int(pointer.get('active_epoch_id') or 0)
    if (
        active_epoch <= 0
        or not quota_file
        or Path(quota_file).name != quota_file
        or not quota_file.endswith('.json')
    ):
        raise ValueError('Invalid active quota epoch pointer')
    quota_path = campaign_dir / quota_file
    if not quota_path.is_file():
        raise FileNotFoundError(f'Active quota epoch file is missing: {quota_file}')
    return quota_path


def initialize_campaign_state(campaign_dir: Path) -> dict[str, object]:
    campaign_dir = campaign_dir.expanduser().resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / STATE_FILE
    quota_path = _active_quota_path(campaign_dir)
    created = not state_path.exists()
    state = _load_json(
        state_path,
        {
            'schema_version': 1,
            'campaign': campaign_dir.name,
            'status': 'pending',
            'created_at': _now(),
            'updated_at': _now(),
            'jobs': [],
            'deterministic_failure_counts': {},
        },
    )
    quota = _load_json(
        quota_path,
        {
            'schema_version': 1,
            'account_count': 6,
            'estimated_gpu_hours_per_account': 20.0,
            'default_usable_hours_per_account': 16.5,
            'maximum_planned_gpu_hours': 100.0,
            'reserve_gpu_hours': 20.0,
            'maximum_concurrent_notebooks_per_account': 2,
            'maximum_concurrent_gpu_notebooks_per_account': 2,
            'maximum_concurrent_cpu_notebooks_per_account': 10,
            'allocations_gpu_hours': {
                'smoke': 5.0,
                'baseline': 50.0,
                'calibration_and_repeats': 20.0,
                'first_ablation': 25.0,
            },
            'planned_gpu_hours': 0.0,
            'observed_gpu_hours': 0.0,
            'accounts': {},
            'updated_at': _now(),
        },
    )
    quota.setdefault(
        'maximum_concurrent_gpu_notebooks_per_account',
        int(quota.get('maximum_concurrent_notebooks_per_account') or 2),
    )
    quota.setdefault('maximum_concurrent_cpu_notebooks_per_account', 10)
    quota.setdefault('quota_epoch_id', 1)
    if int(quota['quota_epoch_id']) <= 0:
        raise ValueError('quota_epoch_id must be positive')
    if quota_path.name != QUOTA_FILE:
        pointer = _load_json(campaign_dir / QUOTA_POINTER_FILE, {})
        if int(pointer['active_epoch_id']) != int(quota['quota_epoch_id']):
            raise ValueError('Active quota pointer and quota epoch file disagree')
    _persist(state_path, state)
    _persist(quota_path, quota)
    if created:
        _event(campaign_dir, 'supervisor_initialized')
    return {
        'campaign_dir': campaign_dir,
        'state': state,
        'quota': quota,
        'quota_path': quota_path,
    }


@_locked_campaign_operation
def register_job(campaign_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Reserve quota and persist a restart-safe job before external submission."""
    initialized = initialize_campaign_state(campaign_dir)
    campaign_dir = initialized['campaign_dir']
    state = initialized['state']
    quota = initialized['quota']
    quota_path = initialized['quota_path']
    if state.get('status') == 'blocked':
        raise RuntimeError(
            f"Campaign is blocked: {state.get('blocked_reason') or 'unspecified'}"
        )
    jobs = state.get('jobs')
    if not isinstance(jobs, list):
        raise ValueError('campaign_state jobs must be a list')
    candidate = dict(job)
    job_id = str(candidate.get('job_id') or '').strip()
    if not job_id:
        raise ValueError('Registered job requires job_id')
    compute_kind = str(candidate.get('compute_kind') or 'gpu').strip().lower()
    if compute_kind not in {'cpu', 'gpu'}:
        raise ValueError('Registered job compute_kind must be `cpu` or `gpu`')
    planned = float(candidate.get('planned_gpu_hours') or 0.0)
    if compute_kind == 'gpu' and planned <= 0.0:
        raise ValueError('Registered GPU job requires positive planned_gpu_hours')
    if compute_kind == 'cpu' and planned != 0.0:
        raise ValueError('Registered CPU job must reserve zero planned_gpu_hours')
    candidate['compute_kind'] = compute_kind
    candidate['planned_gpu_hours'] = planned
    active_epoch = int(quota.get('quota_epoch_id') or 1)
    candidate_epoch = int(candidate.get('quota_epoch_id') or active_epoch)
    if candidate_epoch != active_epoch:
        raise ValueError('Registered job quota_epoch_id must match active quota epoch')
    candidate['quota_epoch_id'] = candidate_epoch
    candidate.setdefault('status', 'preparing')
    candidate.setdefault('retry_count', 0)
    candidate.setdefault('started_epoch', time.time())
    candidate.setdefault('registered_at', _now())
    candidate['updated_at'] = _now()
    secret_path = _contains_secret(candidate)
    if secret_path:
        raise ValueError(f'Refusing to persist secret field: {secret_path}')

    existing_index = next(
        (index for index, current in enumerate(jobs) if current.get('job_id') == job_id),
        None,
    )
    if (
        existing_index is not None
        and int(jobs[existing_index].get('quota_epoch_id') or 1) != candidate_epoch
    ):
        raise ValueError('Existing job belongs to a different quota epoch')
    proposed_jobs = [dict(current) for current in jobs]
    if existing_index is None:
        proposed_jobs.append(candidate)
    else:
        merged = dict(proposed_jobs[existing_index])
        merged.update(candidate)
        proposed_jobs[existing_index] = merged
        candidate = merged
    proposed_quota = json.loads(json.dumps(quota))
    violations = _update_quota(proposed_quota, proposed_jobs)
    if violations:
        _event(
            campaign_dir,
            'job_registration_refused',
            job_id=job_id,
            violations=violations,
        )
        raise RuntimeError('Quota guard refused job: ' + ', '.join(violations))

    state['jobs'] = proposed_jobs
    state['status'] = 'running'
    state['updated_at'] = _now()
    _persist(campaign_dir / STATE_FILE, state)
    _persist(quota_path, proposed_quota)
    _event(
        campaign_dir,
        'job_registered',
        job_id=job_id,
        account=candidate.get('account'),
        stage=candidate.get('stage'),
        planned_gpu_hours=planned,
    )
    return candidate


@_locked_campaign_operation
def resume_blocked_campaign(campaign_dir: Path, *, reason: str) -> dict[str, object]:
    """Resume an authorized repair while preserving the failed job evidence."""
    reason = str(reason or '').strip()
    if not reason:
        raise ValueError('Campaign resume requires a reason')
    initialized = initialize_campaign_state(campaign_dir)
    campaign_dir = initialized['campaign_dir']
    state = initialized['state']
    quota = initialized['quota']
    quota_path = initialized['quota_path']
    if state.get('status') != 'blocked':
        return {'state': state, 'quota': quota, 'changed': False}
    for job in state.get('jobs', []):
        if job.get('status') == 'blocked':
            job['status'] = 'superseded'
            job['superseded_at'] = _now()
    state['status'] = 'pending'
    state.pop('blocked_reason', None)
    state['updated_at'] = _now()
    _update_quota(quota, state.get('jobs', []))
    _persist(campaign_dir / STATE_FILE, state)
    _persist(quota_path, quota)
    _event(campaign_dir, 'campaign_resumed', reason=reason)
    return {'state': state, 'quota': quota, 'changed': True}


@_locked_campaign_operation
def update_registered_job(
    campaign_dir: Path,
    job_id: str,
    **updates: Any,
) -> dict[str, Any]:
    """Update one registered job without ever persisting credential contents."""
    initialized = initialize_campaign_state(campaign_dir)
    campaign_dir = initialized['campaign_dir']
    state = initialized['state']
    quota = initialized['quota']
    quota_path = initialized['quota_path']
    jobs = state.get('jobs')
    if not isinstance(jobs, list):
        raise ValueError('campaign_state jobs must be a list')
    secret_path = _contains_secret(updates)
    if secret_path:
        raise ValueError(f'Refusing to persist secret field: {secret_path}')
    target = next((job for job in jobs if job.get('job_id') == job_id), None)
    if target is None:
        raise KeyError(f'Unknown campaign job: {job_id}')
    target.update(updates)
    target['updated_at'] = _now()
    violations = _update_quota(quota, jobs)
    if violations:
        raise RuntimeError('Quota guard refused job update: ' + ', '.join(violations))
    effective_jobs = [
        job for job in jobs if job.get('status') != 'superseded'
    ]
    if any(job.get('status') == 'blocked' for job in effective_jobs):
        state['status'] = 'blocked'
        state['blocked_reason'] = 'job_blocked'
    elif effective_jobs and all(
        job.get('status') == 'complete' for job in effective_jobs
    ):
        state['status'] = 'complete'
    elif effective_jobs and all(
        job.get('status') in TERMINAL_STATES for job in effective_jobs
    ):
        state['status'] = 'failed'
    elif not effective_jobs:
        state['status'] = 'pending'
    else:
        state['status'] = 'running'
    state['updated_at'] = _now()
    _persist(campaign_dir / STATE_FILE, state)
    _persist(quota_path, quota)
    _event(
        campaign_dir,
        'job_updated',
        job_id=job_id,
        status=target.get('status'),
        kernel_ref=target.get('kernel_ref'),
    )
    return target


def _failure_fingerprint(error_type: str, message: str) -> str:
    compiler_signatures = re.findall(
        r'(?im)^([^\r\n]{0,300}(?:fatal\s+)?error:[^\r\n]{0,500})$',
        message,
    )
    signatures = re.findall(
        r'([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception):[^\\r\\n"{}]+)',
        message,
    )
    stable_message = (
        compiler_signatures[0]
        if compiler_signatures
        else signatures[-1] if signatures else message
    )
    normalized = ' '.join(stable_message.lower().split())
    return hashlib.sha256(
        f'{error_type}:{normalized}'.encode('utf-8')
    ).hexdigest()[:16]


def _recount_deterministic_failures(jobs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        fingerprint = str(job.get('failure_fingerprint') or '')
        if (
            fingerprint
            and job.get('status') in TERMINAL_STATES
            and job.get('status') != 'complete'
        ):
            counts[fingerprint] = counts.get(fingerprint, 0) + 1
    return counts


def _job_runtime_hours(job: dict, now_epoch: float) -> float:
    started = job.get('started_epoch')
    if not isinstance(started, (int, float)):
        return 0.0
    ended = job.get('ended_epoch')
    endpoint = float(ended) if isinstance(ended, (int, float)) else now_epoch
    return max(0.0, endpoint - float(started)) / 3600.0


def _job_gpu_runtime_hours(job: dict, now_epoch: float) -> float:
    if str(job.get('compute_kind') or 'gpu').strip().lower() != 'gpu':
        return 0.0
    return _job_runtime_hours(job, now_epoch)


def _update_quota(quota: dict, jobs: list[dict]) -> list[str]:
    now_epoch = time.time()
    active_epoch = int(quota.get('quota_epoch_id') or 1)
    account_usage: dict[str, dict[str, float | int]] = {}
    planned = observed = observed_cpu = 0.0
    cpu_jobs = 0
    for job in jobs:
        if str(job.get('status') or '').strip().lower() == 'superseded':
            continue
        job_epoch = int(job.get('quota_epoch_id') or 1)
        if job_epoch != active_epoch:
            continue
        account = str(job.get('account') or 'unassigned')
        usage = account_usage.setdefault(
            account, {'planned_gpu_hours': 0.0, 'observed_gpu_hours': 0.0}
        )
        job_planned = float(job.get('planned_gpu_hours') or 0.0)
        compute_kind = str(job.get('compute_kind') or 'gpu').strip().lower()
        job_observed = _job_gpu_runtime_hours(job, now_epoch)
        if compute_kind == 'cpu':
            cpu_jobs += 1
            observed_cpu += _job_runtime_hours(job, now_epoch)
        usage['planned_gpu_hours'] = float(usage['planned_gpu_hours']) + job_planned
        usage['observed_gpu_hours'] = float(usage['observed_gpu_hours']) + job_observed
        planned += job_planned
        observed += job_observed
    quota['planned_gpu_hours'] = round(planned, 6)
    quota['observed_gpu_hours'] = round(observed, 6)
    quota['planned_cpu_jobs'] = cpu_jobs
    quota['observed_cpu_hours'] = round(observed_cpu, 6)
    quota['accounts'] = account_usage
    quota['updated_at'] = _now()
    violations: list[str] = []
    if planned > float(quota['maximum_planned_gpu_hours']):
        violations.append('maximum_planned_gpu_hours_exceeded')
    usable = float(quota['default_usable_hours_per_account'])
    for account, usage in account_usage.items():
        if account != 'unassigned' and float(usage['planned_gpu_hours']) > usable:
            violations.append(f'account_usable_hours_exceeded:{account}')
    concurrency_limits = {
        'cpu': int(quota.get('maximum_concurrent_cpu_notebooks_per_account') or 10),
        'gpu': int(
            quota.get('maximum_concurrent_gpu_notebooks_per_account')
            or quota.get('maximum_concurrent_notebooks_per_account')
            or 2
        ),
    }
    concurrency_accounts = {
        str(job.get('account') or 'unassigned')
        for job in jobs
        if job.get('status') in {
            'preparing', 'submitted', 'queued', 'running', 'polling',
            'recovery_pending',
        }
    }
    for account in concurrency_accounts:
        for compute_kind, limit in concurrency_limits.items():
            running = sum(
                1
                for job in jobs
                if str(job.get('account') or 'unassigned') == account
                and str(job.get('compute_kind') or 'gpu').strip().lower() == compute_kind
                and job.get('status') in {
                    'preparing', 'submitted', 'queued', 'running', 'polling',
                    'recovery_pending',
                }
            )
            if account != 'unassigned' and running > limit:
                violations.append(
                    f'account_{compute_kind}_concurrency_exceeded:{account}'
                )
    quota['violations'] = violations
    return violations


def _kaggle_cli_env(job: dict) -> dict[str, str]:
    """Build a child-only Kaggle environment without exposing credential values."""
    env = os.environ.copy()
    credential_path = job.get('kaggle_json_path')
    if credential_path:
        credentials = json.loads(
            Path(str(credential_path)).expanduser().read_text(encoding='utf-8')
        )
        env['KAGGLE_USERNAME'] = str(credentials['username'])
        env['KAGGLE_KEY'] = str(credentials['key'])
    return env


def _query_kernel(job: dict) -> tuple[str, str]:
    executable = shutil.which('kaggle')
    if not executable:
        raise RuntimeError('Kaggle CLI is unavailable')
    process = subprocess.run(
        [executable, 'kernels', 'status', str(job['kernel_ref'])],
        env=_kaggle_cli_env(job),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    output = '\n'.join(part for part in (process.stdout, process.stderr) if part)
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', output)
    if process.returncode or not match:
        raise RuntimeError('Kaggle status query failed')
    return match.group(1).replace('_', '').lower(), ''


def _failure_diagnostics(job: dict) -> str:
    """Recover the real failed-notebook log for classification and fingerprinting."""
    root = (
        Path(str(job['work_dir'])).expanduser().resolve()
        / 'failure_diagnostics'
        / str(job.get('job_id') or 'job')
    )
    root.mkdir(parents=True, exist_ok=True)

    def read_bounded_logs(base: Path) -> str:
        candidates = [base / 'error.json']
        for pattern in ('*.log', '*/*.log', '*/logs/*.log', '*/*/logs/*.log'):
            candidates.extend(sorted(base.glob(pattern)))
        parts: list[str] = []
        total = 0
        unique = list(dict.fromkeys(candidates))
        unique.sort(
            key=lambda path: (
                0 if path.name == 'error.json' else
                1 if 'smina' in path.name.lower() else 2,
                str(path),
            )
        )
        markers = (
            'error:', 'exception:', 'fatal:', 'http error', 'traceback',
            'no such file', 'out of memory', 'preempt', 'timeout',
        )
        for log_path in unique:
            if not log_path.is_file():
                continue
            text = log_path.read_text(encoding='utf-8', errors='replace')[-200_000:]
            lines = text.splitlines()
            salient = [
                line for line in lines
                if any(marker in line.lower() for marker in markers)
            ][-80:]
            excerpt = '\n'.join(
                [f'[{log_path.name}]', *salient, *lines[-40:]]
            )[-30_000:]
            parts.append(excerpt)
            total += len(excerpt)
            if len(parts) >= 20 or total >= 100_000:
                break
        return '\n'.join(parts)[:100_000]

    existing = read_bounded_logs(root)
    if not existing:
        existing = read_bounded_logs(
            Path(str(job['work_dir'])).expanduser().resolve() / 'failed-output'
        )
    if existing:
        return existing
    try:
        executable = shutil.which('kaggle')
        if not executable:
            return ''
        subprocess.run(
            [
                executable,
                'kernels',
                'output',
                str(job['kernel_ref']),
                '-p',
                str(root),
                '-q',
            ],
            env=_kaggle_cli_env(job),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return read_bounded_logs(root)


def _recover_job(job: dict) -> Path | None:
    if str(job.get('recovery_kind') or '') == 'raw_kernel':
        return _recover_raw_kernel(job)
    from ezscreen.backends.kaggle.runner import (
        _download_if_kernel_completed_after_poll_failure,
    )

    work_dir = Path(str(job['work_dir'])).expanduser().resolve()
    nb_work = Path(str(job.get('nb_work') or work_dir)).expanduser().resolve()
    return _download_if_kernel_completed_after_poll_failure(
        kernel_ref=str(job['kernel_ref']),
        nb_work=nb_work,
        kj_path=job.get('kaggle_json_path'),
        index_csv=Path(str(job['index_csv'])) if job.get('index_csv') else None,
        filtered_csv=(
            Path(str(job['filtered_csv'])) if job.get('filtered_csv') else None
        ),
        receptor_variant=job.get('receptor_variant'),
        engine=str(job.get('engine') or 'unidock'),
    )


def _recover_raw_kernel(job: dict) -> Path | None:
    """Download every output from a custom kernel without screening transforms."""
    output_dir = Path(str(job['work_dir'])).expanduser().resolve() / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        executable = shutil.which('kaggle')
        if not executable:
            return None
        process = subprocess.run(
            [
                executable,
                'kernels',
                'output',
                str(job['kernel_ref']),
                '-p',
                str(output_dir),
                '-q',
            ],
            env=_kaggle_cli_env(job),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=900,
            check=False,
        )
        if process.returncode:
            return None
    except (OSError, subprocess.TimeoutExpired, KeyError, ValueError):
        return None
    return output_dir


def _finalize_recovered_output(job: dict, output_dir: Path) -> dict[str, Any] | None:
    finalizer = str(job.get('finalizer') or '').strip()
    if not finalizer:
        return None
    intended = str(job.get('intended_csv') or '').strip()
    if not intended:
        raise ValueError('Finalized recovery requires intended_csv')
    intended_csv = Path(intended).expanduser().resolve()
    if finalizer == 'bayesbind-vina':
        from ezscreen.benchmark.bayesbind_paper import finalize_vina_output

        return finalize_vina_output(output_dir, intended_csv)
    if finalizer == 'bayesbind-gnina':
        from ezscreen.benchmark.bayesbind_paper import finalize_gnina_output

        return finalize_gnina_output(output_dir, intended_csv)
    raise ValueError(f'Unknown recovery finalizer: {finalizer}')


def _resume_job(job: dict) -> dict:
    if str(job.get('recovery_kind') or '') == 'raw_kernel':
        from ezscreen.backends.kaggle.kernel import push_kernel
        from ezscreen.backends.kaggle.runner import (
            _KAGGLE_API_LOCK,
            _set_account_creds,
            _wait_for_dataset,
            _wait_for_dataset_sync,
        )

        required = ('run_id', 'notebook_path', 'dataset_ref', 'username', 'work_dir')
        if any(not job.get(field) for field in required):
            return {'status': 'failed'}
        with _KAGGLE_API_LOCK:
            if job.get('kaggle_json_path'):
                _set_account_creds(job['kaggle_json_path'])
            if not job.get('dataset_reused'):
                _wait_for_dataset(str(job['dataset_ref']))
                _wait_for_dataset_sync(
                    str(job['dataset_ref']), label='raw kernel inputs'
                )
            kernel_ref = push_kernel(
                str(job['run_id']),
                Path(str(job['notebook_path'])),
                str(job['dataset_ref']),
                str(job['username']),
                Path(str(job['work_dir'])),
                accelerator=str(job.get('accelerator') or 'none'),
            )
        job['kernel_ref'] = kernel_ref
        return {'status': 'submitted', 'kernel_ref': kernel_ref}
    from ezscreen.backends.kaggle.runner import resume_failed_shards

    return resume_failed_shards(
        str(job['run_id']),
        Path(str(job['work_dir'])).expanduser().resolve(),
        retry_limit=2,
    )


@_locked_campaign_operation
def supervise_once(campaign_dir: Path) -> dict[str, object]:
    initialized = initialize_campaign_state(campaign_dir)
    campaign_dir = initialized['campaign_dir']
    state = initialized['state']
    quota = initialized['quota']
    quota_path = initialized['quota_path']
    jobs = state.get('jobs')
    if not isinstance(jobs, list):
        raise ValueError('campaign_state jobs must be a list')
    violations = _update_quota(quota, jobs)
    if violations:
        state['status'] = 'blocked'
        state['blocked_reason'] = 'quota_guard'
        _event(campaign_dir, 'quota_guard_blocked', violations=violations)
        _persist(campaign_dir / STATE_FILE, state)
        _persist(quota_path, quota)
        return {'state': state, 'quota': quota, 'changed': True}

    changed = False
    credential_failures: set[str] = set()
    accounts_seen: set[str] = set()
    for job in jobs:
        status = str(job.get('status') or 'pending')
        if status not in {'submitted', 'queued', 'running', 'polling', 'recovery_pending'}:
            continue
        account = str(job.get('account') or 'unassigned')
        accounts_seen.add(account)
        try:
            kernel_status, message = _query_kernel(job)
        except Exception as exc:
            credential_failures.add(account)
            job['last_poll_error'] = type(exc).__name__
            _event(
                campaign_dir,
                'kernel_poll_error',
                job_id=job.get('job_id'),
                account=account,
                error_type=type(exc).__name__,
            )
            continue
        job['last_polled_at'] = _now()
        job['kernel_status'] = kernel_status
        if kernel_status == 'complete':
            if not job.get('work_dir'):
                job['status'] = 'blocked'
                job['blocked_reason'] = 'recovery_metadata_missing'
            else:
                recovered = _recover_job(job)
                if recovered is None:
                    job['status'] = 'recovery_pending'
                else:
                    try:
                        final_metrics = _finalize_recovered_output(job, recovered)
                    except Exception as exc:
                        job['status'] = 'blocked'
                        job['blocked_reason'] = 'finalization_failed'
                        job['finalization_error'] = type(exc).__name__
                        job['output_dir'] = str(recovered)
                        job['ended_epoch'] = time.time()
                        _event(
                            campaign_dir,
                            'kernel_finalization_failed',
                            job_id=job.get('job_id'),
                            finalizer=job.get('finalizer'),
                            error_type=type(exc).__name__,
                        )
                    else:
                        job['status'] = 'complete'
                        job['output_dir'] = str(recovered)
                        job['ended_epoch'] = time.time()
                        if final_metrics:
                            for key in (
                                'metrics_path',
                                'ranking_path',
                                'target_id',
                                'coverage',
                                'ef1',
                                'ef5',
                                'ef20',
                                'roc_auc',
                                'bedroc20',
                                'first_active_rank',
                                'pose_retention',
                            ):
                                if key in final_metrics:
                                    job[key] = final_metrics[key]
                            job['finalized_at'] = _now()
                        _event(
                            campaign_dir,
                            'kernel_recovered',
                            job_id=job.get('job_id'),
                            kernel_ref=job.get('kernel_ref'),
                            finalizer=job.get('finalizer'),
                        )
            changed = True
            continue
        if kernel_status not in {'error', 'cancelacknowledged'}:
            if status != 'running':
                job['status'] = 'running'
                changed = True
            continue

        from ezscreen.backends.kaggle.poller import _classify_error

        diagnostics = _failure_diagnostics(job)
        failure_message = diagnostics or message
        error_type = _classify_error(failure_message)
        fingerprint = _failure_fingerprint(error_type, failure_message)
        job['failure_fingerprint'] = fingerprint
        job['error_type'] = error_type
        if error_type in TRANSIENT_ERRORS and int(job.get('retry_count') or 0) < 2:
            job['retry_count'] = int(job.get('retry_count') or 0) + 1
            outcome = _resume_job(job) if job.get('run_id') and job.get('work_dir') else {
                'status': 'failed'
            }
            job['status'] = str(outcome.get('status') or 'failed')
            _event(
                campaign_dir,
                'transient_retry',
                job_id=job.get('job_id'),
                error_type=error_type,
                retry_count=job['retry_count'],
            )
        else:
            attempts = 1 + sum(
                1
                for other in jobs
                if other is not job
                and other.get('failure_fingerprint') == fingerprint
                and other.get('status') in TERMINAL_STATES
                and other.get('status') != 'complete'
            )
            job['status'] = (
                'blocked' if attempts >= 3 else 'failed'
            )
            job['ended_epoch'] = time.time()
            _event(
                campaign_dir,
                'kernel_failed',
                job_id=job.get('job_id'),
                error_type=error_type,
                fingerprint=fingerprint,
                attempts=attempts,
            )
        changed = True

    effective_jobs = [
        job for job in jobs if job.get('status') != 'superseded'
    ]
    if accounts_seen and credential_failures == accounts_seen:
        state['status'] = 'blocked'
        state['blocked_reason'] = 'all_credentials_failed'
    elif not jobs:
        state['status'] = 'pending'
    elif any(job.get('status') == 'blocked' for job in effective_jobs):
        state['status'] = 'blocked'
        state['blocked_reason'] = 'job_blocked'
    elif effective_jobs and all(
        job.get('status') == 'complete' for job in effective_jobs
    ):
        state['status'] = 'complete'
    elif effective_jobs and all(
        job.get('status') in TERMINAL_STATES for job in effective_jobs
    ):
        state['status'] = 'failed'
    elif not effective_jobs:
        state['status'] = 'pending'
    else:
        state['status'] = 'running'
    state['updated_at'] = _now()
    state['deterministic_failure_counts'] = _recount_deterministic_failures(jobs)
    _update_quota(quota, jobs)
    _persist(campaign_dir / STATE_FILE, state)
    _persist(quota_path, quota)
    return {'state': state, 'quota': quota, 'changed': changed}


def supervise_campaign(
    campaign_dir: Path,
    *,
    once: bool = False,
    poll_seconds: int = 30,
) -> dict[str, object]:
    while True:
        result = supervise_once(campaign_dir)
        status = str(result['state'].get('status'))
        if once or status in TERMINAL_STATES:
            return result
        time.sleep(max(1, poll_seconds))
