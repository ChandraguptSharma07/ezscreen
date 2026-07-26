import json
from pathlib import Path

import pytest

from ezscreen.benchmark import supervisor


def test_supervisor_initializes_restart_safe_quota_artifacts(tmp_path: Path) -> None:
    result = supervisor.supervise_once(tmp_path / 'campaign')
    campaign = tmp_path / 'campaign'
    assert result['state']['status'] == 'pending'
    assert (campaign / 'campaign_state.json').exists()
    assert (campaign / 'execution_events.jsonl').exists()
    quota = json.loads((campaign / 'quota_budget.json').read_text())
    assert quota['maximum_planned_gpu_hours'] == 100.0
    assert quota['reserve_gpu_hours'] == 20.0
    assert quota['maximum_concurrent_notebooks_per_account'] == 2
    assert quota['maximum_concurrent_gpu_notebooks_per_account'] == 2
    assert quota['maximum_concurrent_cpu_notebooks_per_account'] == 10


def test_supervisor_enforces_distinct_cpu_and_gpu_concurrency(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    for index in range(10):
        supervisor.register_job(
            campaign,
            {
                'job_id': f'cpu-{index}',
                'account': 'acct1',
                'compute_kind': 'cpu',
                'planned_gpu_hours': 0.0,
            },
        )
    with pytest.raises(RuntimeError, match='account_cpu_concurrency_exceeded:acct1'):
        supervisor.register_job(
            campaign,
            {
                'job_id': 'cpu-over',
                'account': 'acct1',
                'compute_kind': 'cpu',
                'planned_gpu_hours': 0.0,
            },
        )

    gpu_campaign = tmp_path / 'gpu-campaign'
    for index in range(2):
        supervisor.register_job(
            gpu_campaign,
            {
                'job_id': f'gpu-{index}',
                'account': 'acct1',
                'compute_kind': 'gpu',
                'planned_gpu_hours': 1.0,
            },
        )
    with pytest.raises(RuntimeError, match='account_gpu_concurrency_exceeded:acct1'):
        supervisor.register_job(
            gpu_campaign,
            {
                'job_id': 'gpu-over',
                'account': 'acct1',
                'compute_kind': 'gpu',
                'planned_gpu_hours': 1.0,
            },
        )


def test_supervisor_blocks_before_reserve_is_consumed(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.initialize_campaign_state(campaign)
    state_path = campaign / 'campaign_state.json'
    state = json.loads(state_path.read_text())
    state['jobs'] = [
        {
            'job_id': 'too-large',
            'status': 'pending',
            'account': 'acct1',
            'planned_gpu_hours': 101.0,
        }
    ]
    state_path.write_text(json.dumps(state), encoding='utf-8')
    result = supervisor.supervise_once(campaign)
    assert result['state']['status'] == 'blocked'
    assert 'maximum_planned_gpu_hours_exceeded' in result['quota']['violations']


def test_supervisor_recovers_completed_kernel(tmp_path: Path, monkeypatch) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.initialize_campaign_state(campaign)
    state_path = campaign / 'campaign_state.json'
    state = json.loads(state_path.read_text())
    state['jobs'] = [
        {
            'job_id': 'j1',
            'status': 'running',
            'account': 'acct1',
            'kernel_ref': 'acct/kernel',
            'work_dir': str(tmp_path / 'work'),
            'planned_gpu_hours': 1.0,
        }
    ]
    state_path.write_text(json.dumps(state), encoding='utf-8')
    monkeypatch.setattr(supervisor, '_query_kernel', lambda job: ('complete', ''))
    output = tmp_path / 'work' / 'output'
    monkeypatch.setattr(supervisor, '_recover_job', lambda job: output)
    result = supervisor.supervise_once(campaign)
    assert result['state']['status'] == 'complete'
    assert result['state']['jobs'][0]['output_dir'] == str(output)


def test_supervisor_refuses_secret_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Refusing to persist secret'):
        supervisor._persist(tmp_path / 'state.json', {'api_key': 'do-not-write'})


def test_register_job_reserves_quota_idempotently(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    job = {
        'job_id': 'baseline-t1',
        'account': 'acct1',
        'stage': 'baseline',
        'planned_gpu_hours': 4.0,
        'work_dir': str(tmp_path / 'work'),
    }
    supervisor.register_job(campaign, job)
    supervisor.register_job(campaign, job)
    state = json.loads((campaign / 'campaign_state.json').read_text())
    quota = json.loads((campaign / 'quota_budget.json').read_text())
    assert len(state['jobs']) == 1
    assert state['jobs'][0]['status'] == 'preparing'
    assert quota['planned_gpu_hours'] == 4.0


def test_register_cpu_job_does_not_consume_gpu_quota(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'prepare-inputs',
            'account': 'acct1',
            'stage': 'input_preparation',
            'compute_kind': 'cpu',
            'planned_gpu_hours': 0.0,
        },
    )
    state = json.loads((campaign / 'campaign_state.json').read_text())
    quota = json.loads((campaign / 'quota_budget.json').read_text())
    assert state['jobs'][0]['compute_kind'] == 'cpu'
    assert quota['planned_gpu_hours'] == 0.0
    assert quota['planned_cpu_jobs'] == 1


def test_register_cpu_job_rejects_gpu_reservation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='CPU job must reserve zero'):
        supervisor.register_job(
            tmp_path / 'campaign',
            {
                'job_id': 'bad-cpu',
                'account': 'acct1',
                'compute_kind': 'cpu',
                'planned_gpu_hours': 1.0,
            },
        )


def test_resume_blocked_campaign_preserves_failed_job(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.initialize_campaign_state(campaign)
    state_path = campaign / 'campaign_state.json'
    state = json.loads(state_path.read_text())
    state['status'] = 'blocked'
    state['blocked_reason'] = 'job_blocked'
    state['jobs'] = [{'job_id': 'old', 'status': 'blocked'}]
    state_path.write_text(json.dumps(state))

    result = supervisor.resume_blocked_campaign(
        campaign, reason='authorized repair'
    )

    assert result['changed'] is True
    assert result['state']['status'] == 'pending'
    assert result['state']['jobs'][0]['status'] == 'superseded'
    assert 'blocked_reason' not in result['state']
    assert 'campaign_resumed' in (campaign / 'execution_events.jsonl').read_text()


def test_register_job_refuses_account_overage_before_submission(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    with pytest.raises(RuntimeError, match='account_usable_hours_exceeded'):
        supervisor.register_job(
            campaign,
            {
                'job_id': 'oversize',
                'account': 'acct1',
                'planned_gpu_hours': 17.0,
            },
        )
    state = json.loads((campaign / 'campaign_state.json').read_text())
    assert state['jobs'] == []


def test_update_registered_job_persists_kernel_and_completion(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'j1',
            'account': 'acct1',
            'planned_gpu_hours': 1.0,
        },
    )
    supervisor.update_registered_job(
        campaign,
        'j1',
        status='submitted',
        kernel_ref='owner/kernel',
    )
    updated = supervisor.update_registered_job(
        campaign,
        'j1',
        status='complete',
        output_dir=str(tmp_path / 'output'),
        ended_epoch=1.0,
    )
    state = json.loads((campaign / 'campaign_state.json').read_text())
    assert updated['kernel_ref'] == 'owner/kernel'
    assert state['status'] == 'complete'


def test_recover_job_dispatches_raw_kernel(tmp_path: Path, monkeypatch) -> None:
    expected = tmp_path / 'output'
    monkeypatch.setattr(supervisor, '_recover_raw_kernel', lambda job: expected)
    assert supervisor._recover_job({'recovery_kind': 'raw_kernel'}) == expected


def test_resume_raw_kernel_requires_restart_metadata() -> None:
    assert supervisor._resume_job({'recovery_kind': 'raw_kernel'}) == {
        'status': 'failed'
    }


def test_query_kernel_uses_bounded_cli_and_normalizes_status(
    tmp_path: Path, monkeypatch
) -> None:
    credential_path = tmp_path / 'kaggle.json'
    credential_path.write_text(
        json.dumps({'username': 'owner', 'key': 'not-persisted'})
    )
    monkeypatch.setattr(supervisor.shutil, 'which', lambda name: 'kaggle')
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return type(
            'Result',
            (),
            {
                'returncode': 0,
                'stdout': 'owner/kernel has status "KernelWorkerStatus.ERROR"',
                'stderr': '',
            },
        )()

    monkeypatch.setattr(supervisor.subprocess, 'run', fake_run)

    status, message = supervisor._query_kernel(
        {
            'kernel_ref': 'owner/kernel',
            'kaggle_json_path': str(credential_path),
        }
    )

    assert (status, message) == ('error', '')
    assert captured['timeout'] == 45
    assert captured['env']['KAGGLE_USERNAME'] == 'owner'


def test_campaign_lock_serializes_state_operations(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    with supervisor._campaign_lock(campaign):
        with pytest.raises(TimeoutError, match='campaign state lock timed out'):
            with supervisor._campaign_lock(campaign, timeout_seconds=0.01):
                pass


def test_supervisor_retries_transient_network_from_downloaded_log(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'net',
            'run_id': 'net',
            'account': 'acct1',
            'compute_kind': 'cpu',
            'planned_gpu_hours': 0.0,
            'status': 'submitted',
            'kernel_ref': 'owner/kernel',
            'work_dir': str(tmp_path / 'work'),
        },
    )
    monkeypatch.setattr(supervisor, '_query_kernel', lambda job: ('error', ''))
    monkeypatch.setattr(
        supervisor,
        '_failure_diagnostics',
        lambda job: 'HTTPError: HTTP Error 502: Bad Gateway',
    )
    monkeypatch.setattr(
        supervisor,
        '_resume_job',
        lambda job: {'status': 'submitted', 'kernel_ref': 'owner/kernel'},
    )

    result = supervisor.supervise_once(campaign)

    job = result['state']['jobs'][0]
    assert job['status'] == 'submitted'
    assert job['error_type'] == 'transient_network'
    assert job['retry_count'] == 1


def test_supervisor_does_not_retry_compiler_error_containing_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'build',
            'account': 'acct1',
            'compute_kind': 'cpu',
            'planned_gpu_hours': 0.0,
            'status': 'submitted',
            'kernel_ref': 'owner/kernel',
            'work_dir': str(tmp_path / 'work'),
        },
    )
    monkeypatch.setattr(supervisor, '_query_kernel', lambda job: ('error', ''))
    monkeypatch.setattr(
        supervisor,
        '_failure_diagnostics',
        lambda job: 'timeout=7200\nmake: *** [main.o] Error 1',
    )
    monkeypatch.setattr(
        supervisor,
        '_resume_job',
        lambda job: pytest.fail('deterministic build failure must not retry'),
    )

    result = supervisor.supervise_once(campaign)

    job = result['state']['jobs'][0]
    assert job['status'] == 'failed'
    assert job['error_type'] == 'build_failure'
    assert job['retry_count'] == 0


def test_build_log_compiler_error_precedes_incidental_prep_text() -> None:
    from ezscreen.backends.kaggle.poller import _classify_error

    message = (
        '[smina_build.log]\n'
        'error: no type named tag\n'
        'MGLTools prep completed earlier'
    )
    assert _classify_error(message) == 'build_failure'


def test_compiler_fingerprints_distinguish_concrete_errors() -> None:
    first = supervisor._failure_fingerprint(
        'build_failure',
        "smina.cpp:8: error: version is not a class template\n"
        'RuntimeError: smina_build.log failed',
    )
    second = supervisor._failure_fingerprint(
        'build_failure',
        "optional.hpp:60: error: no type named tag\n"
        'RuntimeError: smina_build.log failed',
    )
    assert first != second


def test_failure_recount_counts_each_terminal_job_once() -> None:
    jobs = [
        {'status': 'failed', 'failure_fingerprint': 'a'},
        {'status': 'failed', 'failure_fingerprint': 'a'},
        {'status': 'running', 'failure_fingerprint': 'a'},
        {'status': 'complete', 'failure_fingerprint': 'b'},
    ]
    assert supervisor._recount_deterministic_failures(jobs) == {'a': 2}


def test_superseded_job_does_not_make_replacement_campaign_fail(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    for job_id in ('old', 'replacement'):
        supervisor.register_job(
            campaign,
            {
                'job_id': job_id,
                'account': 'acct1',
                'compute_kind': 'cpu',
                'planned_gpu_hours': 0.0,
            },
        )
    supervisor.update_registered_job(campaign, 'old', status='superseded')
    supervisor.update_registered_job(
        campaign, 'replacement', status='complete'
    )

    state = json.loads((campaign / 'campaign_state.json').read_text())
    assert state['status'] == 'complete'


def test_superseded_gpu_job_does_not_consume_quota(tmp_path: Path) -> None:
    campaign = tmp_path / 'campaign'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'old-gpu',
            'account': 'acct1',
            'compute_kind': 'gpu',
            'planned_gpu_hours': 4.0,
        },
    )
    supervisor.update_registered_job(campaign, 'old-gpu', status='superseded')
    quota = json.loads((campaign / 'quota_budget.json').read_text())
    assert quota['planned_gpu_hours'] == 0.0
    assert quota['observed_gpu_hours'] == 0.0


def test_supervisor_finalizes_recovered_bayesbind_vina_kernel(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = tmp_path / 'campaign'
    intended = tmp_path / 'intended.csv'
    intended.write_text('ligand,label\nL1,1\n', encoding='utf-8')
    output = tmp_path / 'work' / 'output'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'vina-finalize',
            'run_id': 'vina-finalize',
            'account': 'acct1',
            'compute_kind': 'cpu',
            'planned_gpu_hours': 0.0,
            'status': 'submitted',
            'kernel_ref': 'owner/kernel',
            'work_dir': str(tmp_path / 'work'),
            'recovery_kind': 'raw_kernel',
            'finalizer': 'bayesbind-vina',
            'intended_csv': str(intended),
        },
    )
    monkeypatch.setattr(supervisor, '_query_kernel', lambda job: ('complete', ''))
    monkeypatch.setattr(supervisor, '_recover_job', lambda job: output)

    from ezscreen.benchmark import bayesbind_paper

    def fake_finalize(recovered: Path, intended_csv: Path) -> dict:
        assert recovered == output
        assert intended_csv == intended.resolve()
        return {
            'metrics_path': str(output / 'target_metrics.csv'),
            'ranking_path': str(output / 'compound_ranking.csv'),
            'target_id': 'AL1A1_HUMAN_4_501_0',
            'coverage': 1.0,
            'ef1': 0.0,
            'roc_auc': 0.5,
        }

    monkeypatch.setattr(bayesbind_paper, 'finalize_vina_output', fake_finalize)

    result = supervisor.supervise_once(campaign)

    job = result['state']['jobs'][0]
    assert job['status'] == 'complete'
    assert job['output_dir'] == str(output)
    assert job['metrics_path'].endswith('target_metrics.csv')
    assert job['ranking_path'].endswith('compound_ranking.csv')
    assert job['target_id'] == 'AL1A1_HUMAN_4_501_0'
    assert job['coverage'] == 1.0
    assert 'finalized_at' in job


def test_supervisor_blocks_recovered_kernel_when_finalization_fails(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = tmp_path / 'campaign'
    intended = tmp_path / 'intended.csv'
    intended.write_text('ligand,label\nL1,1\n', encoding='utf-8')
    output = tmp_path / 'work' / 'output'
    supervisor.register_job(
        campaign,
        {
            'job_id': 'vina-bad-finalize',
            'run_id': 'vina-bad-finalize',
            'account': 'acct1',
            'compute_kind': 'cpu',
            'planned_gpu_hours': 0.0,
            'status': 'submitted',
            'kernel_ref': 'owner/kernel',
            'work_dir': str(tmp_path / 'work'),
            'recovery_kind': 'raw_kernel',
            'finalizer': 'bayesbind-vina',
            'intended_csv': str(intended),
        },
    )
    monkeypatch.setattr(supervisor, '_query_kernel', lambda job: ('complete', ''))
    monkeypatch.setattr(supervisor, '_recover_job', lambda job: output)

    from ezscreen.benchmark import bayesbind_paper

    def fake_finalize(_recovered: Path, _intended_csv: Path) -> dict:
        raise ValueError('bad counts')

    monkeypatch.setattr(bayesbind_paper, 'finalize_vina_output', fake_finalize)

    result = supervisor.supervise_once(campaign)

    job = result['state']['jobs'][0]
    assert result['state']['status'] == 'blocked'
    assert job['status'] == 'blocked'
    assert job['blocked_reason'] == 'finalization_failed'
    assert job['finalization_error'] == 'ValueError'
    assert job['output_dir'] == str(output)
