from __future__ import annotations

import inspect
import json

from ezscreen.benchmark.vina_baseline import (
    render_vina_full_redock_notebook,
    run_multi_account_vina_baseline,
)


def test_render_vina_full_redock_notebook_contract_is_valid_json() -> None:
    rendered = render_vina_full_redock_notebook(
        run_id="ezs-test-vina-baseline",
        shard_filenames=["shard_000.smi"],
        box_center=[1.0, 2.0, 3.0],
        box_size=[20.0, 21.0, 22.0],
        prep_cfg={
            "enumerate_enabled": False,
            "max_heavy_atoms": 70,
            "max_mw": 700.0,
            "max_rotatable_bonds": 20,
        },
        exhaustiveness=4,
        num_modes=1,
        cpu=4,
        seed=42,
    )

    notebook = json.loads(rendered)
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert "AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64" in source
    assert "pip install meeko gemmi" in source
    assert "scores.csv" in source
    assert "failed_docking.csv" in source
    assert "failed_prep.csv" in source
    assert "poses.sdf" in source
    assert "shutil.rmtree(DOCKED" in source
    assert "shutil.rmtree(pathlib.Path(\"/kaggle/working/ligands\")" in source
    assert "done.flag" in source
    assert '"engine": "vina-local"' in source



def test_render_vina_full_redock_notebook_supports_released_sdf_contract() -> None:
    rendered = render_vina_full_redock_notebook(
        run_id="ezs-test-vina-exact",
        shard_filenames=["exact_sdf_manifest.csv"],
        box_center=[1.0, 2.0, 3.0],
        box_size=[20.0, 21.0, 22.0],
        prep_cfg={"enumerate_enabled": False, "enable_gpu_size_filter": False},
        exhaustiveness=8,
        num_modes=9,
        cpu=1,
        seed=42,
        receptor_prep="openbabel",
        center_ligand=True,
        ligand_box_padding=3.0,
        ligand_box_lexicographic=True,
        ligand_input_mode="sdf_manifest",
        meeko_hydrate=False,
        vina_use_executable_defaults=True,
        sdf_archive_filenames=["exact_sdfs.tar"],
        ligand_workers=3,
    )

    notebook = json.loads(rendered)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{cell.get('id', '')}", "exec")
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "LIGAND_INPUT_MODE = 'sdf_manifest'" in source
    assert "MEEKO_HYDRATE = False" in source
    assert "VINA_USE_EXECUTABLE_DEFAULTS = True" in source
    assert "AddHs(mol, addCoords=True)" in source
    assert "MoleculePreparation(hydrate=MEEKO_HYDRATE)" in source
    assert "if not VINA_USE_EXECUTABLE_DEFAULTS" in source
    assert "vina_search_options_provenance" in source
    assert "exact_sdf_sha256" in source
    assert "list(max(tuple(ligand_box), tuple(BOX_SIZE)))" in source
    assert "receptor_prep_manifest.json" in source
    assert "receptor_prep_stdout.log" in source
    assert "receptor_prep_stderr.log" in source
    assert "openbabel-wheel==3.1.1.22" in source
    assert "output_sha256" in source
    assert "SDF_ARCHIVE_FILENAMES = ['exact_sdfs.tar']" in source
    assert "EXACT_SDF_ROOT" in source
    assert 'tarfile.open(archive_path, "r:*")' in source
    assert "SDF_EXTRACTED_ROOTS" in source
    assert "extracted directory match violation" in source
    assert "SDF hash drift" in source
    assert "LIGAND_WORKERS = 3" in source
    assert "ThreadPoolExecutor" in source
    assert '"--cpu", str(CPU)' in source
    assert "parallelism_provenance" in source


def test_vina_baseline_recovers_completed_kernel_after_poll_failure() -> None:
    source = inspect.getsource(run_multi_account_vina_baseline)

    assert "_download_if_kernel_completed_after_poll_failure" in source
    assert "engine=\"vina-local\"" in source
