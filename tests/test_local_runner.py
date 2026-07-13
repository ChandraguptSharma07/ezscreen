from __future__ import annotations

from pathlib import Path

from ezscreen.backends.local import runner


def test_run_vina_appends_seed_argument(monkeypatch, tmp_path):
    calls = []

    class Result:
        returncode = 0
        stderr = ''
        stdout = ''

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        out_path = Path(cmd[cmd.index('--out') + 1])
        out_path.write_text('REMARK VINA RESULT: -8.1\n', encoding='utf-8')
        return Result()

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)

    out = tmp_path / 'pose_out.pdbqt'
    text = runner._run_vina(
        vina=tmp_path / 'vina',
        receptor=tmp_path / 'receptor.pdbqt',
        ligand=tmp_path / 'ligand.pdbqt',
        out_path=out,
        center=[1.0, 2.0, 3.0],
        size=[10.0, 11.0, 12.0],
        exhaustiveness=4,
        num_modes=3,
        seed=12345,
    )

    assert text == 'REMARK VINA RESULT: -8.1\n'
    assert '--seed' in calls[0]
    assert calls[0][calls[0].index('--seed') + 1] == '12345'


def test_run_local_screening_derives_stable_seed_per_ligand(monkeypatch, tmp_path):
    shard = tmp_path / 'shard_0000.pdbqt'
    shard.write_text('REMARK lig_id ligA\nTORSDOF 0\n', encoding='utf-8')
    receptor = tmp_path / 'receptor.pdbqt'
    receptor.write_text('RECEPTOR\n', encoding='utf-8')

    monkeypatch.setattr(runner, 'get_vina_binary', lambda: tmp_path / 'vina')
    monkeypatch.setattr(
        runner,
        '_split_pdbqt_shard',
        lambda _shard, out_dir: [('ligA', out_dir / 'ligA.pdbqt'), ('ligB', out_dir / 'ligB.pdbqt')],
    )

    seeds = []

    def fake_run_vina(*args, **kwargs):
        seeds.append(kwargs['seed'])
        return 'REMARK VINA RESULT: -7.5\n'

    monkeypatch.setattr(runner, '_run_vina', fake_run_vina)

    result = runner.run_local_screening(
        run_id='ezs-local',
        receptor_pdbqt=receptor,
        shard_paths=[shard],
        box_center=[1.0, 2.0, 3.0],
        box_size=[10.0, 11.0, 12.0],
        work_dir=tmp_path / 'run',
    )

    assert result['status'] == 'complete'
    assert seeds == [
        runner._stable_vina_seed('ezs-local', 'ligA'),
        runner._stable_vina_seed('ezs-local', 'ligB'),
    ]
