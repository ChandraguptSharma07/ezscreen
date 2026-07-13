from __future__ import annotations


def test_stage_team_job_receptor_files_stages_ensemble_with_stable_kaggle_names(tmp_path):
    from ezscreen.team.coordinator import stage_team_job_receptor_files

    primary = tmp_path / 'receptor_fixed.pdbqt'
    conformer_a = tmp_path / 'receptor_conformer_003_aligned_1M52_A.pdbqt'
    conformer_b = tmp_path / 'receptor_conformer_004_aligned_7DT2_B.pdbqt'
    primary.write_text('primary', encoding='utf-8')
    conformer_a.write_text('conf-a', encoding='utf-8')
    conformer_b.write_text('conf-b', encoding='utf-8')

    staged_files, variant_specs = stage_team_job_receptor_files(
        work_dir=tmp_path,
        receptor_pdbqt=primary,
        receptor_pdbqts=[primary, conformer_a, conformer_b],
    )

    assert [path.name for path in staged_files] == [
        'receptor.pdbqt',
        'receptor_variant_001.pdbqt',
        'receptor_variant_002.pdbqt',
    ]
    assert [spec['name'] for spec in variant_specs] == [
        'default',
        'conformer_003_aligned_1M52_A',
        'conformer_004_aligned_7DT2_B',
    ]
    assert [spec['filename'] for spec in variant_specs] == [
        'receptor.pdbqt',
        'receptor_variant_001.pdbqt',
        'receptor_variant_002.pdbqt',
    ]



def test_stage_team_job_receptor_files_keeps_receptor_fixed_pdb_for_kaggle_prep(tmp_path):
    from ezscreen.team.coordinator import stage_team_job_receptor_files

    primary = tmp_path / 'receptor_fixed.pdbqt'
    receptor_pdb = tmp_path / 'receptor_fixed.pdb'
    primary.write_text('primary', encoding='utf-8')
    receptor_pdb.write_text('pdb', encoding='utf-8')

    staged_files, variant_specs = stage_team_job_receptor_files(
        work_dir=tmp_path,
        receptor_pdbqt=primary,
        receptor_variant_modes=['default', 'propka_ph'],
        receptor_pdb=receptor_pdb,
    )

    assert [path.name for path in staged_files] == ['receptor.pdbqt', 'receptor_fixed.pdb']
    assert [spec['name'] for spec in variant_specs] == ['default', 'propka_ph']
    assert all(spec['filename'] == 'receptor.pdbqt' for spec in variant_specs)



def test_start_team_job_includes_all_staged_receptor_files(tmp_path):
    from ezscreen.team.coordinator import start_team_job

    receptor = tmp_path / 'receptor.pdbqt'
    conformer = tmp_path / 'receptor_variant_001.pdbqt'
    receptor_pdb = tmp_path / 'receptor_fixed.pdb'
    notebook = tmp_path / 'team_notebook.ipynb'
    shard = tmp_path / 'shard_000.smi'
    for path in (receptor, conformer, receptor_pdb, notebook, shard):
        path.write_text(path.name, encoding='utf-8')
    calls = []

    class FakeClient:
        def request(self, method, path, payload):
            calls.append((method, path, payload))
            return {'team_job_id': 'tjob_1', 'worker_jobs': []}

    start_team_job(
        client=FakeClient(),
        user_id='u1',
        run_id='ezs-test',
        receptor_pdbqt=receptor,
        receptor_files=[receptor, conformer, receptor_pdb],
        shard_paths=[shard],
        notebook_path=notebook,
        engine='unidock',
        accelerator='nvidiaTeslaT4',
    )

    files = calls[0][2]['payload']['files']
    assert [item['filename'] for item in files] == [
        'receptor.pdbqt',
        'receptor_variant_001.pdbqt',
        'receptor_fixed.pdb',
        'team_notebook.ipynb',
        'shard_000.smi',
    ]
    assert [item['role'] for item in files] == [
        'receptor',
        'receptor',
        'receptor_pdb',
        'notebook',
        'ligand_shard',
    ]
