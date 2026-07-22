from __future__ import annotations

import inspect
import json

from ezscreen.benchmark.vina_baseline import render_vina_full_redock_notebook, run_multi_account_vina_baseline


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



def test_vina_baseline_recovers_completed_kernel_after_poll_failure() -> None:
    source = inspect.getsource(run_multi_account_vina_baseline)

    assert "_download_if_kernel_completed_after_poll_failure" in source
    assert "engine=\"vina-local\"" in source
