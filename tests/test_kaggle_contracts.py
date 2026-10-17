from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ezscreen.backends.kaggle import gnina_runner, plip_runner, runner


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _joined_notebook_source(text: str) -> str:
    notebook = json.loads(text)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_render_vina_notebook_contract_is_valid_json() -> None:
    rendered = runner.render_vina_notebook(
        run_id="ezs-contract",
        shard_filenames=["shard_000.smi", "shard_001.smi"],
        notebook_index=1,
        total_notebooks=2,
        box_center=[1.0, 2.0, 3.0],
        box_size=[20.0, 21.0, 22.0],
        prep_on_kaggle=True,
        engine="unidock",
        scoring="vina",
        unidock_grid_point_cap=2097152,
        receptor_pdb_filename="receptor_fixed.pdb",
        receptor_variants=[
            {"name": "default", "filename": "receptor.pdbqt", "prepare_on_kaggle": False},
            {"name": "hip", "filename": "receptor_variant_001.pdbqt", "prepare_on_kaggle": True},
        ],
        prep_cfg={"enumerate_enabled": True},
        results_cfg={"poses_returned": 12},
    )

    joined = _joined_notebook_source(rendered)

    assert "shard_000.smi" in joined
    assert "shard_001.smi" in joined
    assert "receptor_fixed.pdb" in joined
    assert "receptor_variant_001.pdbqt" in joined
    assert "/kaggle/working/scores.csv" in joined
    assert "/kaggle/working/poses.sdf" in joined
    assert "/kaggle/working/failed_prep.csv" in joined
    assert "/kaggle/working/failed_docking.csv" in joined
    assert "missing_output_pdbqt" in joined
    assert "poses_raw_pdbqt.zip" in joined
    assert "pose_export_status.json" in joined
    assert "['obabel', '-ipdbqt'" in joined
    assert "Pose export contract failed" in joined
    assert "import json, os, pathlib, time, traceback" in joined
    assert "_REQUIRED_GRID_CAP = 2097152" in joined
    assert "capacity_only_change" in joined
    assert "cuda-nvcc-12-8" in joined
    assert "patched UniDock kernel.h hash mismatch" in joined
    assert "source_build_manifest.json" in joined
    assert "/kaggle/working/error.json" in joined
    assert "filtered_gpu_size.csv" in joined
    assert "output.zip" in joined


def test_render_vina_notebook_uses_generalized_fragment_contract() -> None:
    rendered = runner.render_vina_notebook(
        run_id="ezs-normalization",
        shard_filenames=["shard_000.smi"],
        prep_on_kaggle=True,
        prep_cfg={"normalization_profile": "generalized_v1"},
    )

    joined = _joined_notebook_source(rendered)

    assert "_normalize_ligand" in joined
    assert "ambiguous_fragments" in joined
    assert "removed_counterions" in joined
    assert "_largest_fragment_mol" not in joined


def test_render_vina_notebook_pins_prep_and_controls_macrocycles() -> None:
    rendered = runner.render_vina_notebook(
        run_id="ezs-rigid-rings",
        shard_filenames=["shard_000.smi"],
        prep_on_kaggle=True,
        prep_cfg={
            "normalization_profile": "generalized_v1",
            "rigid_macrocycles": True,
            "kaggle_rdkit_version": "2026.3.5",
            "kaggle_meeko_version": "0.7.1",
        },
    )

    joined = _joined_notebook_source(rendered)

    assert "rdkit==2026.3.5 meeko==0.7.1" in joined
    assert "MoleculePreparation(rigid_macrocycles=True)" in joined
    assert "macrocycle_candidate" in joined
    assert "macrocycle_policy" in joined
    assert "ligand_prep_manifest.json" in joined
    assert "pdbqt_with_glue_pseudoatoms" in joined
    assert "'rdkit': '2026.3.5'" in joined
    assert "'meeko': '0.7.1'" in joined


def test_render_vina_notebook_rejects_unsafe_prep_version() -> None:
    with pytest.raises(ValueError, match="Invalid pinned Meeko version"):
        runner.render_vina_notebook(
            run_id="ezs-unsafe-version",
            shard_filenames=["shard_000.smi"],
            prep_cfg={"kaggle_meeko_version": "0.7.1'; import os; #"},
        )


def test_render_gnina_notebook_contract_is_valid_json(tmp_path: Path) -> None:
    out_path = tmp_path / "gnina.ipynb"
    gnina_runner._render_gnina_notebook(
        run_id="ezs-gnina",
        top_n=25,
        dataset_slug="owner/ezscreen-gnina-ezs-gnina",
        template_dir=Path("ezscreen/backends/kaggle/templates"),
        out_path=out_path,
    )

    joined = _joined_notebook_source(out_path.read_text())

    assert "/kaggle/input" in joined
    assert "gnina_poses.sdf" in joined
    assert "scores_top_n.csv" in joined
    assert "cnn_scores.csv" in joined
    assert "gnina_output.zip" in joined


def test_render_plip_notebook_contract_is_valid_json(tmp_path: Path) -> None:
    out_path = tmp_path / "plip.ipynb"
    plip_runner._render_plip_notebook(
        run_id="ezs-plip",
        top_n=15,
        dataset_slug="owner/ezscreen-plip-ezs-plip",
        template_dir=Path("ezscreen/backends/kaggle/templates"),
        out_path=out_path,
    )

    joined = _joined_notebook_source(out_path.read_text())

    assert "/kaggle/input" in joined
    assert "plip_poses.sdf" in joined
    assert "scores_top_n.csv" in joined
    assert "interactions.json" in joined
    assert "plip_output.zip" in joined


def test_single_output_contract_writers_emit_results_meta_and_unscored_reasons(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    _write_csv(
        output_dir / "scores.csv",
        ["ligand", "docking_score", "name", "smiles"],
        [{"ligand": "lig_0001", "docking_score": "-8.2", "name": "Mol A", "smiles": "CCO"}],
    )
    _write_csv(
        output_dir / "index.csv",
        ["ligand", "name", "smiles"],
        [
            {"ligand": "lig_0001", "name": "Mol A", "smiles": "CCO"},
            {"ligand": "lig_fail", "name": "Mol Fail", "smiles": "CCN"},
            {"ligand": "lig_missing", "name": "Mol Missing", "smiles": "CCC"},
        ],
    )
    _write_csv(
        output_dir / "filtered_gpu_size.csv",
        ["name", "smiles"],
        [{"name": "Too big", "smiles": "CCCC"}],
    )
    _write_csv(
        output_dir / "failed_prep.csv",
        ["name", "smiles", "reason"],
        [{"name": "Prep fail", "smiles": "CCCl", "reason": "bad_valence"}],
    )
    _write_csv(
        output_dir / "failed_docking.csv",
        ["ligand", "raw_score", "reason"],
        [{"ligand": "lig_fail", "raw_score": "0.0", "reason": "no_pose"}],
    )

    runner._write_results_meta(output_dir, engine="unidock")
    runner._write_single_unscored_reasons(output_dir)

    meta = json.loads((output_dir / "results_meta.json").read_text(encoding="utf-8"))
    reasons = list(csv.DictReader((output_dir / "unscored_reasons.csv").open(newline="", encoding="utf-8")))

    assert meta["engine"] == "unidock"
    assert meta["score_col"] == "docking_score"
    assert meta["score_type"]
    assert {row["reason"] for row in reasons} == {
        "gpu_size_filter",
        "bad_valence",
        "no_pose",
        "missing_output_pdbqt",
    }
    docking_row = next(row for row in reasons if row["ligand"] == "lig_fail")
    assert docking_row["name"] == "Mol Fail"
    assert docking_row["smiles"] == "CCN"
    missing_row = next(row for row in reasons if row["ligand"] == "lig_missing")
    assert missing_row["name"] == "Mol Missing"
    failed = list(
        csv.DictReader(
            (output_dir / "failed_docking.csv").open(newline="", encoding="utf-8")
        )
    )
    assert {(row["ligand"], row["reason"]) for row in failed} == {
        ("lig_fail", "no_pose"),
        ("lig_missing", "missing_output_pdbqt"),
    }


def test_single_output_contract_rejects_empty_pose_export(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_csv(
        output_dir / "scores.csv",
        ["ligand", "score"],
        [{"ligand": "lig_0001", "score": "-8.2"}],
    )
    (output_dir / "poses.sdf").write_text("", encoding="utf-8")
    (output_dir / "pose_export_status.json").write_text(
        json.dumps({"requested": 1, "written": 0}), encoding="utf-8"
    )

    assert runner._single_output_contract_error(output_dir) == "pose_export_failed"

    (output_dir / "pose_export_status.json").write_text(
        json.dumps({"requested": 1, "written": 1}), encoding="utf-8"
    )
    assert runner._single_output_contract_error(output_dir) is None


def test_single_output_contract_rejects_prepared_but_unscored_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_csv(output_dir / "scores.csv", ["ligand", "score"], [])
    _write_csv(
        output_dir / "index.csv",
        ["ligand", "name"],
        [{"ligand": "lig_0001", "name": "Mol A"}],
    )

    assert runner._single_output_contract_error(output_dir) == "no_scored_poses"
