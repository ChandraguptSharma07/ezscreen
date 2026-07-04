from __future__ import annotations

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


def test_implemented_engines_are_a_subset():
    impl = {e.key for e in engines.implemented_engines()}
    everything = {e.key for e in engines.all_engines()}
    # UniDock-Pro's structure-based docking == UniDock, so it is not a distinct
    # engine; its value (LBVS/hybrid) is a future feature.
    assert impl == {"unidock", "gnina"}
    assert impl < everything  # smina / diffdock / unidock-pro declared but not offered


def test_dockable_engines_have_a_template():
    dockable = {e.key for e in engines.dockable_engines()}
    assert dockable == {"unidock", "gnina"}
    for e in engines.dockable_engines():
        assert e.kaggle_template


def test_diffdock_is_box_free():
    d = engines.get("diffdock")
    assert not d.requires_box
    assert d.scoring_functions == ()


def test_scoring_function_membership():
    assert engines.supports_scoring("unidock", "vinardo")
    assert not engines.supports_scoring("unidock", "cnn")
    assert not engines.supports_scoring("autodock-gpu", "vina")


def test_offered_engines_exclude_ad4_scoring():
    # ad4 needs precomputed affinity maps, so it is not a drop-in scoring choice.
    for e in engines.dockable_engines():
        assert "ad4" not in e.scoring_functions
        assert set(e.scoring_functions) == {"vina", "vinardo"}


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
        assert "'--scoring', 'vinardo'" in joined, eng


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

    assert "pip install meeko gemmi prody" in joined
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

    def fake_merge_shard_results(output_dirs, main_output):
        main_output.mkdir(parents=True, exist_ok=True)
        (main_output / "scores.csv").write_text("ligand,score,receptor_variant\n")
        return {}

    monkeypatch.setattr(runner, "render_vina_notebook", fake_render_vina_notebook)
    monkeypatch.setattr(runner, "push_kernel", fake_push_kernel)
    monkeypatch.setattr(runner, "poll_until_done", fake_poll_until_done)
    monkeypatch.setattr(runner, "_download_output", fake_download_output)

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
