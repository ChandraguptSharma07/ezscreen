from __future__ import annotations

import sys
import types

from ezscreen.backends import engines


def test_default_engine_is_implemented():
    prof = engines.get(engines.DEFAULT_ENGINE)
    assert prof.implemented
    assert prof.requires_box


def test_unknown_engine_falls_back_to_default():
    assert engines.get("does-not-exist").key == engines.DEFAULT_ENGINE


def test_gnina_supports_cnn_and_vina_family():
    g = engines.get("gnina")
    assert g.supports_cnn
    # ranks by Vina affinity (CNN columns are supplementary), so the merger is unchanged
    assert g.native_score_type == "vina_kcal_mol"
    assert "vinardo" in g.scoring_functions


def test_unidock_is_vina_family_without_cnn():
    u = engines.get("unidock")
    assert not u.supports_cnn
    assert u.native_score_type == "vina_kcal_mol"
    assert engines.default_scoring("unidock") == "vina"
    assert engines.supports_scoring("unidock", "ad4")


def test_implemented_engines_are_a_subset():
    impl = {e.key for e in engines.implemented_engines()}
    everything = {e.key for e in engines.all_engines()}
    # UniDock-Pro's structure-based docking == UniDock, so it is not a distinct
    # engine; its value (LBVS/hybrid) is a future feature.
    assert impl == {"unidock", "gnina", "diffdock", "vina-local"}
    assert impl < everything  # smina / unidock-pro declared but not offered


def test_dockable_engines_have_a_template():
    dockable = {e.key for e in engines.dockable_engines()}
    assert dockable == {"unidock", "gnina", "diffdock"}
    for e in engines.dockable_engines():
        assert e.kaggle_template or e.key == "diffdock"


def test_diffdock_is_box_free():
    d = engines.get("diffdock")
    assert not d.requires_box
    assert d.scoring_functions == ()


def test_cpu_vina_is_a_registered_backend():
    cpu = engines.get("vina-local")
    assert cpu.implemented
    assert not cpu.requires_gpu
    assert cpu.requires_box
    assert cpu.default_scoring == "vina"


def test_scoring_function_membership():
    assert engines.supports_scoring("unidock", "vinardo")
    assert not engines.supports_scoring("unidock", "cnn")
    assert not engines.supports_scoring("autodock-gpu", "vina")


def test_ad4_is_offered_only_through_unidock_map_path():
    assert engines.supports_scoring("unidock", "ad4")
    assert not engines.supports_scoring("gnina", "ad4")
    assert not engines.supports_scoring("vina-local", "ad4")


def test_scoring_flag_threaded_into_docking_command():
    import json

    from ezscreen.backends.kaggle import runner

    for eng in ("unidock", "gnina"):
        src = runner.render_vina_notebook(
            run_id="ezs-x",
            shard_filenames=["s0.smi"],
            box_center=[1, 2, 3],
            box_size=[20, 20, 20],
            engine=eng,
            scoring="vinardo",
        )
        joined = "\n".join("".join(c["source"]) for c in json.loads(src)["cells"])
        assert "SCORING = 'vinardo'" in joined, eng
        if eng == "unidock":
            assert "'--scoring', SCORING" in joined
        else:
            assert "'--scoring', 'vinardo'" in joined


def test_kaggle_prep_notebook_loads_openbabel_ligand_formats():
    import json

    from ezscreen.backends.kaggle import runner

    src = runner.render_vina_notebook(
        run_id="ezs-x",
        shard_filenames=["raw_0000_ligand.mol2", "raw_0001_ligand.pdb"],
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
        prep_on_kaggle=True,
    )
    joined = "\n".join("".join(c["source"]) for c in json.loads(src)["cells"])

    assert "def _convert_with_obabel" in joined
    assert "if _suffix in ('.mol2', '.mol', '.pdb')" in joined
    assert "subprocess.run(['obabel', str(_path), '-O', str(_out)]" in joined
    assert "def _largest_fragment_mol" in joined
    assert "failed_prep.csv" in joined
    assert "'input_smiles'" in joined
    assert "'prep_note'" in joined



def test_team_worker_notebook_accepts_local_shard_subset():
    import json

    from ezscreen.backends.kaggle import runner

    src = runner.render_vina_notebook(
        run_id="ezs-team",
        shard_filenames=["shard_000.smi", "shard_001.smi"],
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
        prep_on_kaggle=True,
    )
    joined = "\n".join("".join(c["source"]) for c in json.loads(src)["cells"])

    assert "_missing_shards = []" in joined
    assert "if not _candidates:" in joined
    assert "_missing_shards.append(_fn)" in joined
    assert "assert LIGAND_PATHS" in joined
    assert "Ignoring shard filenames not present in this dataset" in joined


def test_team_worker_notebook_uses_shard_prefix_for_ligand_ids():
    import json

    from ezscreen.backends.kaggle import runner

    src = runner.render_vina_notebook(
        run_id="ezs-team",
        shard_filenames=["shard_000.smi", "shard_001.smi"],
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
        prep_on_kaggle=True,
    )
    joined = "\n".join("".join(c["source"]) for c in json.loads(src)["cells"])

    assert "_shard_prefix = re.sub(r'[^A-Za-z0-9]+', '_', _shard_path.stem).strip('_')" in joined
    assert "_lig_id = f'{_lig_prefix}_lig_{_prep_passed:05d}'" in joined


def test_team_worker_notebook_uses_packaged_unidock_for_stock_grid_cap():
    import json

    from ezscreen.backends.kaggle import runner

    src = runner.render_vina_notebook(
        run_id="ezs-team",
        shard_filenames=["shard_000.smi"],
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
        engine="unidock",
        prep_on_kaggle=True,
    )
    joined = "\n".join("".join(c["source"]) for c in json.loads(src)["cells"])

    assert "_PACKAGED_BIN = _sh.which('unidock')" in joined
    assert "_FALLBACK_URL = 'https://github.com/dptech-corp/Uni-Dock/releases/download/1.1.0/unidock-1.1.0-cuda120-linux-x86_64'" in joined
    assert "_nvcc_bin = _sh.which('nvcc')" in joined
    assert "if _pl.Path(_BIN).exists() and _os.path.getsize(_BIN) > 500_000:" in joined
    assert "elif _REQUIRED_GRID_CAP > _STOCK_GRID_CAP:" in joined
    assert "elif _PACKAGED_BIN:" in joined
    assert "Downloading UniDock fallback binary" in joined
    assert "if _pl.Path(_BIN).exists() and _os.path.getsize(_BIN) > 500_000:" in joined
    assert "elif (not _PACKAGED_BIN) and _pl.Path(_FALLBACK_BIN).exists()" in joined


def test_kaggle_receptor_variant_notebook_prepares_propka_receptor():
    import json

    from ezscreen.backends.kaggle import runner

    src = runner.render_vina_notebook(
        run_id="ezs-x",
        shard_filenames=["s0.smi"],
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
        receptor_variant="propka_ph",
        receptor_pdb_filename="receptor_fixed.pdb",
    )
    nb = json.loads(src)
    cells = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(cells)

    assert "pip install rdkit==2026.3.5 meeko==0.7.1 gemmi prody" in joined
    assert "pip install pdb2pqr propka" in joined
    assert "RECEPTOR_PDB_FILENAME = 'receptor_fixed.pdb'" in joined
    assert "RECEPTOR_VARIANT_SPECS" in joined
    assert "RECEPTOR_PDBQTS = {}" in joined
    assert "for RECEPTOR_VARIANT, RECEPTOR_PDBQT in RECEPTOR_PDBQTS.items()" in joined
    assert "RECEPTOR_DOCKED_DIRS" in joined
    assert "out_pdbqt_items" in joined
    assert "--titration-state-method=propka" in joined
    assert "--keep-chain" in joined
    assert "_name.upper() == 'CA'" in joined
    assert "return 'C'" in joined
    assert "def _apply_pqr_residue_names_to_source" in joined
    assert "Meeko receptor prep produced a truncated PDBQT" in joined
    assert "pdbqt_atoms" in joined
    assert "mk_prepare_receptor" in joined
    assert "meeko.cli.mk_prepare_receptor" in joined
    assert "Meeko receptor prep failed" in joined
    assert "receptor_prep" in joined

    for cell in cells:
        if cell.strip():
            compile(cell, "rendered_vina_shard_cell.py", "exec")




def test_receptor_variant_specs_accept_ensemble_pdbqts(tmp_path):
    from ezscreen.backends.kaggle import runner

    primary = tmp_path / "receptor.pdbqt"
    primary.write_text("PRIMARY\n")
    conf = tmp_path / "receptor_conformer_001_open.pdbqt"
    conf.write_text("CONF\n")

    specs = runner._receptor_variant_specs(primary, [primary, conf], None)

    assert [s["name"] for s in specs] == ["default", "conformer_001_open"]
    assert [s["filename"] for s in specs] == [
        "receptor.pdbqt",
        "receptor_variant_001.pdbqt",
    ]
    assert all(not s["prepare_on_kaggle"] for s in specs)

def test_multi_account_runner_packs_receptor_variants_into_selected_kernel_count(tmp_path, monkeypatch):
    from pathlib import Path

    from ezscreen.backends.kaggle import runner

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("RECEPTOR\n")
    receptor_pdb = tmp_path / "receptor_fixed.pdb"
    receptor_pdb.write_text("ATOM\nEND\n")
    shards = []
    for i in range(4):
        shard = tmp_path / f"shard_{i:04d}.pdbqt"
        shard.write_text(f"REMARK shard {i}\n")
        shards.append(shard)

    accounts = [
        {
            "account": {
                "name": f"acct{i}",
                "kaggle_json_path": tmp_path / f"kaggle_{i}.json",
            },
            "shard_count": 1,
        }
        for i in range(4)
    ]

    pushed = []
    rendered = []

    monkeypatch.setattr(runner, "_set_account_creds", lambda _path: "user")
    monkeypatch.setattr(runner, "_wait_for_dataset", lambda _ref: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        runner,
        "upload_run_dataset",
        lambda **kwargs: f"dataset/{kwargs['run_id']}",
    )

    real_render = runner.render_vina_notebook

    def fake_render_vina_notebook(**kwargs):
        rendered.append(kwargs)
        return real_render(**kwargs)

    def fake_push_kernel(**kwargs):
        pushed.append(kwargs)
        return f"user/{kwargs['run_id']}"

    def fake_poll_until_done(*args, **kwargs):
        return {"status": "complete", "error_type": None}

    def fake_download_output(kernel_ref, nb_work, **kwargs):
        out = Path(nb_work) / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "scores.csv").write_text(
            "ligand,score,receptor_variant\nlig_00000,-1.0,default\nlig_00000,-2.0,propka_ph\n"
        )
        return out

    def fake_merge_shard_results(output_dirs, main_output, **kwargs):
        main_output.mkdir(parents=True, exist_ok=True)
        (main_output / "scores.csv").write_text("ligand,score,receptor_variant\n")
        return {}

    monkeypatch.setattr(runner, "render_vina_notebook", fake_render_vina_notebook)
    monkeypatch.setattr(runner, "push_kernel", fake_push_kernel)
    monkeypatch.setattr(runner, "poll_until_done", fake_poll_until_done)
    monkeypatch.setattr(runner, "_download_output", fake_download_output)
    monkeypatch.setattr(runner, "_wait_for_dataset_sync", lambda _refs, **_kwargs: None)

    import ezscreen.results.merger as merger

    monkeypatch.setattr(merger, "merge_shard_results", fake_merge_shard_results)

    result = runner.run_multi_account_screening(
        run_id="ezs-pack",
        receptor_pdbqt=receptor,
        shard_paths=shards,
        account_assignments=accounts,
        work_dir=tmp_path / "run",
        receptor_variant_modes=["default", "propka_ph"],
        receptor_pdb=receptor_pdb,
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
    )

    assert result["status"] == "complete"
    assert len(pushed) == 4
    assert len(rendered) == 4
    assert all(len(call["receptor_variants"]) == 2 for call in rendered)
    assert [call["notebook_index"] for call in rendered] == [0, 1, 2, 3]
    assert all(call["total_notebooks"] == 4 for call in rendered)


def test_backend_adapters_expose_engine_profile(tmp_path):
    from ezscreen.backends.kaggle.runner import KaggleVinaBackend
    from ezscreen.backends.local.runner import LocalVinaBackend

    gnina = KaggleVinaBackend("gnina")
    local = LocalVinaBackend()

    assert gnina.engine_key == "gnina"
    assert gnina.requires_gpu
    assert gnina.requires_box
    assert gnina.score_type == engines.native_score_type("gnina")
    assert local.engine_key == "vina-local"
    assert not local.requires_gpu
    assert local.requires_box


def test_local_backend_adapter_calls_legacy_runner(tmp_path, monkeypatch):
    from pathlib import Path

    from ezscreen.backends.base import DockingBox
    from ezscreen.backends.local import runner as local_runner

    calls = []

    def fake_run_local_screening(**kwargs):
        calls.append(kwargs)
        return {"status": "complete", "output_dir": kwargs["work_dir"] / "output"}

    monkeypatch.setattr(local_runner, "run_local_screening", fake_run_local_screening)

    result = local_runner.LocalVinaBackend().run(
        ligands=[tmp_path / "shard_0000.pdbqt"],
        receptor=tmp_path / "receptor.pdbqt",
        box=DockingBox(center=(1.0, 2.0, 3.0), size=(20.0, 21.0, 22.0)),
        config={"run_id": "ezs-local", "work_dir": tmp_path / "run", "exhaustiveness": 8},
    )

    assert result["status"] == "complete"
    assert calls[0]["run_id"] == "ezs-local"
    assert calls[0]["box_center"] == [1.0, 2.0, 3.0]
    assert calls[0]["box_size"] == [20.0, 21.0, 22.0]
    assert calls[0]["exhaustiveness"] == 8
    assert calls[0]["shard_paths"] == [Path(tmp_path / "shard_0000.pdbqt")]




def test_wait_for_dataset_sync_requires_stable_ready_streak(monkeypatch):
    from ezscreen.backends.kaggle import runner

    calls: list[int] = []
    statuses = iter(["ready", "updating", "ready", "ready", "ready"])

    class FakeApi:
        def authenticate(self):
            return None

        def dataset_status(self, _dataset_ref):
            return next(statuses)

    monkeypatch.setitem(sys.modules, "kaggle", types.SimpleNamespace(api=FakeApi()))
    monkeypatch.setattr(runner, "_DATASET_SYNC_POLL_INTERVAL", 1)
    monkeypatch.setattr(runner, "_DATASET_SYNC_TIMEOUT", 10)
    monkeypatch.setattr(runner, "_DATASET_SYNC_READY_STREAK", 3)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: calls.append(seconds))

    runner._wait_for_dataset_sync("user/dataset")

    assert calls == [1, 1, 1, 1]


def test_run_screening_job_waits_for_sync_probe_before_submit(tmp_path, monkeypatch):
    from pathlib import Path

    from ezscreen.backends.kaggle import runner

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("RECEPTOR\n")
    (tmp_path / "receptor_prep.pdb").write_text("ATOM\nEND\n")
    shard = tmp_path / "shard_0000.smi"
    shard.write_text("CCO ligand\n")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}")
    work_dir = tmp_path / "run"

    steps: list[str] = []

    monkeypatch.setattr(runner, "upload_run_dataset", lambda **_kwargs: "user/ezs-run")
    monkeypatch.setattr(runner, "_wait_for_dataset", lambda _ref: steps.append("ready"))
    monkeypatch.setattr(runner, "_wait_for_dataset_sync", lambda _ref, **_kwargs: steps.append("sync"))
    monkeypatch.setattr(runner, "push_kernel", lambda **_kwargs: steps.append("push") or "user/kernel")
    monkeypatch.setattr(
        runner,
        "poll_until_done",
        lambda **_kwargs: {"status": "complete", "error_type": None},
    )
    monkeypatch.setattr(
        runner,
        "_download_output",
        lambda _kernel_ref, job_work_dir, **_kwargs: Path(job_work_dir) / "output",
    )

    result = runner.run_screening_job(
        run_id="ezs-run",
        receptor_pdbqt=receptor,
        shard_paths=[shard],
        notebook_path=notebook,
        username="user",
        work_dir=work_dir,
    )

    assert result["status"] == "complete"
    assert steps == ["ready", "sync", "push"]


def test_run_screening_job_observer_registers_before_push(tmp_path, monkeypatch):
    from ezscreen.backends.kaggle import runner

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("RECEPTOR\n")
    shard = tmp_path / "shard_0000.smi"
    shard.write_text("CCO ligand\n")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}")
    events = []

    monkeypatch.setattr(runner, "upload_run_dataset", lambda **_kwargs: "user/data")
    monkeypatch.setattr(runner, "_wait_for_dataset", lambda _ref: None)
    monkeypatch.setattr(runner, "_wait_for_dataset_sync", lambda _ref: None)
    monkeypatch.setattr(
        runner,
        "push_kernel",
        lambda **_kwargs: events.append(("push", {})) or "user/kernel",
    )
    monkeypatch.setattr(
        runner,
        "poll_until_done",
        lambda **_kwargs: {"status": "complete", "error_type": None},
    )
    monkeypatch.setattr(
        runner,
        "_download_output",
        lambda _ref, work_dir, **_kwargs: work_dir / "output",
    )

    runner.run_screening_job(
        run_id="ezs-observed",
        receptor_pdbqt=receptor,
        shard_paths=[shard],
        notebook_path=notebook,
        username="user",
        work_dir=tmp_path / "run",
        job_observer=lambda event, details: events.append((event, details)),
    )

    assert [event for event, _ in events] == [
        "preparing",
        "push",
        "submitted",
        "complete",
    ]
    assert events[2][1]["kernel_ref"] == "user/kernel"


def test_multi_account_recovers_completed_kernel_after_poll_failure(tmp_path, monkeypatch):
    from ezscreen.backends.kaggle import runner

    out_dir = tmp_path / "downloaded"
    calls: list[tuple[str, str | None]] = []

    def fake_status(kernel_ref, *, cred_lock=None, kj_path=None):
        calls.append((kernel_ref, kj_path))
        return "complete", "Kernel status: complete"

    def fake_download(kernel_ref, work_dir, **kwargs):
        calls.append((f"download:{kernel_ref}", kwargs.get("kj_path")))
        out_dir.mkdir()
        return out_dir

    monkeypatch.setattr(runner, "_kernel_status_with_account", fake_status)
    monkeypatch.setattr(runner, "_download_output", fake_download)

    recovered = runner._download_if_kernel_completed_after_poll_failure(
        kernel_ref="owner/kernel",
        nb_work=tmp_path,
        engine="unidock",
        kj_path="acct/kaggle.json",
    )

    assert recovered == out_dir
    assert calls == [
        ("owner/kernel", "acct/kaggle.json"),
        ("download:owner/kernel", "acct/kaggle.json"),
    ]


def test_run_screening_job_recovers_completed_kernel_after_poll_failure(
    tmp_path, monkeypatch
):
    from ezscreen.backends.kaggle import runner

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("RECEPTOR\n")
    (tmp_path / "receptor_prep.pdb").write_text("ATOM\nEND\n")
    shard = tmp_path / "shard_0000.smi"
    shard.write_text("CCO ligand\n")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}")
    work_dir = tmp_path / "run"
    recovered = tmp_path / "recovered-output"
    recovered.mkdir()

    monkeypatch.setattr(runner, "upload_run_dataset", lambda **_kwargs: "user/ezs-run")
    monkeypatch.setattr(runner, "_wait_for_dataset", lambda _ref: None)
    monkeypatch.setattr(runner, "_wait_for_dataset_sync", lambda _ref: None)
    monkeypatch.setattr(runner, "push_kernel", lambda **_kwargs: "user/kernel")
    monkeypatch.setattr(
        runner,
        "poll_until_done",
        lambda **_kwargs: {
            "status": "timeout",
            "error_type": "transient",
            "message": "poll failed",
        },
    )
    monkeypatch.setattr(
        runner,
        "_download_if_kernel_completed_after_poll_failure",
        lambda **_kwargs: recovered,
    )

    result = runner.run_screening_job(
        run_id="ezs-run",
        receptor_pdbqt=receptor,
        shard_paths=[shard],
        notebook_path=notebook,
        username="user",
        work_dir=work_dir,
        kaggle_json_path="acct/kaggle.json",
    )

    assert result["status"] == "complete"
    assert result["output_dir"] == recovered
    assert result["error_type"] is None
    assert result["recovered_after_poll_failure"] is True

def test_ad4_notebook_requires_validated_maps_and_uses_literal_exhaustiveness():
    import json

    from ezscreen.backends.kaggle import runner

    src = runner.render_vina_notebook(
        run_id="ezs-ad4",
        shard_filenames=["s0.pdbqt"],
        box_center=[1, 2, 3],
        box_size=[20, 20, 20],
        engine="unidock",
        scoring="ad4",
        ad4_maps_prefix="ace_receptor_TZ",
        exhaustiveness=32,
    )
    joined = "\n".join("".join(c["source"]) for c in json.loads(src)["cells"])

    assert "AD4 map receptor hash differs from selected receptor" in joined
    assert "*(['--maps', str(AD4_MAPS_PREFIX)]" in joined
    assert "cmd.extend([\x27--exhaustiveness\x27, str(EXHAUSTIVENESS)])" in joined
    assert "ad4_map_runtime.json" in joined
