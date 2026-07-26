import csv
import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from typer.testing import CliRunner

from ezscreen.benchmark import bayesbind_paper
from ezscreen.benchmark.bayesbind_paper import freeze_bayesbind

runner = CliRunner()


def _write_target(path: Path, *, conflict: bool = False) -> None:
    path.mkdir(parents=True)
    common = {
        "pocket_center_x": 1.0,
        "pocket_center_y": 2.0,
        "pocket_center_z": 3.0,
        "pocket_size_x": 20.0,
        "pocket_size_y": 21.0,
        "pocket_size_z": 22.0,
    }
    pd.DataFrame(
        [
            {"lig_smiles": "CCO", "lig_file": f"chembl_structures/{path.name}_active0.sdf", "pchembl_value": 6.0, **common},
            {"lig_smiles": "CCN", "lig_file": f"chembl_structures/{path.name}_inactive.sdf", "pchembl_value": 4.0, **common},
        ]
    ).to_csv(path / "actives.csv", index=False)
    random_smiles = ["O", "N"] if not conflict else ["CCO", "N"]
    pd.DataFrame(
        [
            {"lig_file": f"chembl_structures/{path.name}_random{idx}.sdf", "lig_smiles": smiles, **common}
            for idx, smiles in enumerate(random_smiles)
        ]
    ).to_csv(path / "random.csv", index=False)
    for name in ("rec.pdb", "rec_hs.pdb"):
        (path / name).write_text("END\n", encoding="utf-8")


def test_freeze_bayesbind_resolves_cross_label_conflicts(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted" / "BayesBindV1.5"
    for split in ("val", "test"):
        (extracted / split).mkdir(parents=True)
    names = [f"T{i:02d}_HUMAN_1_2_0" for i in range(25)]
    for index, name in enumerate(names):
        split = "val" if index < 14 else "test"
        _write_target(
            extracted / split / name,
            conflict=index == 0,
        )
    campaign = tmp_path / "benchmarks" / "campaign"

    result = freeze_bayesbind(
        campaign,
        extracted,
        random_limit=2,
        active_cutoff=5.0,
    )

    assert result["target_count"] == 25
    assert result["paper_intended_rows"] == 75
    assert result["source_compounds"] == 74
    assert result["cross_label_conflicts"] == 1
    targets = json.loads((campaign / "targets.json").read_text())
    assert [row["target_id"] for row in targets] == names
    assert targets[0]["split"] == "development"
    library_lines = (campaign / "paper_inputs" / names[0] / "library.smi").read_text(encoding="utf-8").splitlines()
    assert all("\t" in line for line in library_lines)
    assert targets[1]["split"] == "held_out_validation"
    rows = list(
        csv.DictReader(
            (campaign / "paper_inputs" / names[0] / "source_compounds.csv").open()
        )
    )
    ethanol = next(row for row in rows if row["canonical_smiles"] == "CCO")
    assert ethanol["label"] == "1"
    assert ethanol["cross_label_conflict_resolved"] == "1"

    binary = campaign / "inputs" / bayesbind_paper.VINA_BINARY_FILENAME
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fake-vina")
    expected = hashlib.sha256(binary.read_bytes()).hexdigest()
    original = bayesbind_paper.VINA_BINARY_SHA256
    bayesbind_paper.VINA_BINARY_SHA256 = expected
    try:
        staged = bayesbind_paper.stage_vina_notebook(
            campaign,
            names[0],
            campaign / "notebooks" / "smoke.ipynb",
            run_id="bayesbind-smoke-test",
            sample_size=2,
        )
    finally:
        bayesbind_paper.VINA_BINARY_SHA256 = original
    library = Path(staged["inputs"][0])
    library_lines = library.read_text(encoding="utf-8").splitlines()
    assert len(library_lines) == 2
    assert all("\t" in line for line in library_lines)
    assert all(" " not in line.split("\t", 1)[1] for line in library_lines)
    assert staged["sample_size"] == 2


def _valid_sdf_bytes(smiles: str = "CCO") -> bytes:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return (Chem.MolToMolBlock(mol) + "\n$$$$\n").encode("utf-8")


def _write_plain_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            import io

            archive.addfile(info, io.BytesIO(data))


def _write_inventory(path: Path, members: dict[str, bytes]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, data in members.items():
            handle.write(
                json.dumps(
                    {"name": name, "kind": "file", "size": len(data), "mtime": "2026-07-24T00:00:00+00:00"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def test_exact_ligand_member_resolution_and_extraction(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted" / "BayesBindV1.5"
    for split in ("val", "test"):
        (extracted / split).mkdir(parents=True)
    names = [f"T{i:02d}_HUMAN_1_2_0" for i in range(25)]
    for index, name in enumerate(names):
        split = "val" if index < 14 else "test"
        _write_target(extracted / split / name)
    campaign = tmp_path / "campaign"
    freeze_bayesbind(campaign, extracted, random_limit=2, active_cutoff=5.0)

    lig_files = sorted(
        row["lig_file"]
        for target_dir in (campaign / "paper_inputs").iterdir()
        for row in csv.DictReader((target_dir / "paper_rows.csv").open())
    )
    members = {f"BigBindV1.5/{lig_file}": _valid_sdf_bytes() for lig_file in lig_files}
    archive = tmp_path / "BigBindV1.5.tar.gz"
    inventory = tmp_path / "BigBindV1.5.members.jsonl"
    _write_plain_tar(archive, members)
    _write_inventory(inventory, members)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()

    result = bayesbind_paper.extract_exact_bayesbind_ligands(
        campaign,
        archive_path=archive,
        inventory_path=inventory,
        expected_archive_sha256=archive_hash,
    )

    assert result["paper_row_count"] == 75
    assert result["unique_lig_file_count"] == 75
    assert result["extracted_ligand_count"] == 75
    assert Path(result["manifest_path"]).exists()
    first = result["ligands"][0]
    assert Path(first["output_path"]).exists()
    assert first["sha256"] == hashlib.sha256(_valid_sdf_bytes()).hexdigest()

    binary = campaign / "inputs" / bayesbind_paper.VINA_BINARY_FILENAME
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fake-vina")
    expected = hashlib.sha256(binary.read_bytes()).hexdigest()
    original = bayesbind_paper.VINA_BINARY_SHA256
    bayesbind_paper.VINA_BINARY_SHA256 = expected
    try:
        staged = bayesbind_paper.stage_vina_notebook(
            campaign,
            names[0],
            campaign / "notebooks" / "exact.ipynb",
            run_id="bayesbind-exact-test",
            sample_size=2,
            use_exact_sdf=True,
            ligand_workers=2,
        )
    finally:
        bayesbind_paper.VINA_BINARY_SHA256 = original
    manifest = json.loads(Path(staged["manifest"]).read_text())
    assert manifest["ligand_input_mode"] == "sdf_manifest"
    assert manifest["declared_deviation"] == "none: exact BigBind SDF bytes staged"
    assert manifest["paper_protocol"]["vina_defaults"] == "executable defaults"
    assert manifest["paper_protocol"]["ligand_workers"] == 2
    assert manifest["exact_sdf_archive_member_count"] == 2
    assert manifest["exact_sdf_archive"].endswith("_sdfs.tar")
    assert any(path.endswith("_sdf_manifest.csv") for path in staged["inputs"])
    assert any(path.endswith("_sdfs.tar") for path in staged["inputs"])
    assert sum(path.endswith(".sdf") for path in staged["inputs"]) == 0

    gnina = bayesbind_paper.stage_gnina_notebook(
        campaign,
        names[0],
        campaign / "notebooks" / "gnina.ipynb",
        run_id="bayesbind-gnina-test",
        sample_size=2,
        gnina_workers=1,
    )
    gnina_manifest = json.loads(Path(gnina["manifest"]).read_text())
    gnina_notebook = json.loads(Path(gnina["notebook"]).read_text())
    gnina_source = "".join("".join(cell.get("source", [])) for cell in gnina_notebook["cells"])
    assert gnina_manifest["engine"] == "gnina"
    assert gnina_manifest["paper_protocol"]["receptor_prep"] == "raw rec.pdb"
    assert gnina_manifest["paper_protocol"]["cnn_model"] == "crossdock_default2018"
    assert gnina_manifest["exact_sdf_archive_member_count"] == 2
    assert any(path.endswith("_sdfs.tar") for path in gnina["inputs"])
    assert sum(path.endswith(".sdf") for path in gnina["inputs"]) == 0
    assert "GNINA_URL" in gnina_source
    assert "gnina.cuda12.8.static" in gnina_source
    assert "--cnn" in gnina_source
    assert "CNN_MODEL" in gnina_source
    assert "crossdock_default2018" in gnina_source
    assert "raw_rec_pdb" in gnina_source
    assert "_stage_exact_sdfs" in gnina_source
    assert "kaggle_extracted_directory" in gnina_source
    assert "sdf_source_kind" in gnina_source
    assert "ThreadPoolExecutor" in gnina_source
    assert "minimizedAffinity" in gnina_source
    assert "CNNaffinity" in gnina_source


def test_finalize_gnina_output_adds_gnina_provenance(tmp_path: Path) -> None:
    intended = tmp_path / "intended.csv"
    intended.write_text(
        "source_compound_id,target_id,label,label_source,canonical_smiles,lig_file,cross_label_conflict_resolved\n"
        "S1,T,1,measured,CCO,chembl_structures/a.sdf,0\n"
        "S2,T,0,random,CCN,chembl_structures/b.sdf,0\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "index.csv").write_text("ligand,name\nS1,S1\nS2,S2\n", encoding="utf-8")
    (output / "scores.csv").write_text(
        "ligand,score,name,smiles,engine,seconds,pose_retained,CNNscore,CNNaffinity\n"
        "S1,-8.0,S1,CCO,gnina-local,1.0,1,0.9,7.0\n",
        encoding="utf-8",
    )
    result = bayesbind_paper.finalize_gnina_output(output, intended)
    metrics = json.loads(Path(result["metrics_path"]).read_text())
    assert metrics["engine"] == "gnina"
    assert metrics["score_provenance"] == "GNINA minimizedAffinity; lower is better"
    assert metrics["cnn_model"] == "crossdock_default2018"
    assert metrics["counts"]["before_prep"] == 2
    assert metrics["counts"]["scored"] == 1


def test_exact_ligand_resolution_fails_closed_on_missing_and_duplicate(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    paper_dir = campaign / "paper_inputs" / "TARGET"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper_rows.csv").write_text(
        "paper_row_id,target_id,paper_class,paper_label,source_index,lig_file\n"
        "r1,TARGET,active,1,0,chembl_structures/missing.sdf\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "members.jsonl"
    inventory.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing BigBind members"):
        bayesbind_paper.resolve_exact_bayesbind_ligand_members(campaign, inventory_path=inventory)

    row = json.dumps({"name": "BigBindV1.5/chembl_structures/missing.sdf", "kind": "file", "size": 1})
    inventory.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate BigBind members"):
        bayesbind_paper.resolve_exact_bayesbind_ligand_members(campaign, inventory_path=inventory)


def test_exact_ligand_extraction_fails_on_malformed_sdf(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    paper_dir = campaign / "paper_inputs" / "TARGET"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper_rows.csv").write_text(
        "paper_row_id,target_id,paper_class,paper_label,source_index,lig_file\n"
        "r1,TARGET,active,1,0,chembl_structures/bad.sdf\n",
        encoding="utf-8",
    )
    member_name = "BigBindV1.5/chembl_structures/bad.sdf"
    members = {member_name: b"not sdf\n$$$$\n"}
    archive = tmp_path / "BigBindV1.5.tar.gz"
    inventory = tmp_path / "members.jsonl"
    _write_plain_tar(archive, members)
    _write_inventory(inventory, members)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Malformed SDF"):
        bayesbind_paper.extract_exact_bayesbind_ligands(
            campaign,
            archive_path=archive,
            inventory_path=inventory,
            expected_archive_sha256=archive_hash,
        )


def test_recover_official_bayesbind_vina_reference_metrics(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    target = "TGT_HUMAN_1_2_0"
    paper_dir = campaign / "paper_inputs" / target
    paper_dir.mkdir(parents=True)
    bayesbind_paper._write_csv(
        paper_dir / "source_compounds.csv",
        [
            {
                "source_compound_id": f"{target}__active",
                "target_id": target,
                "label": "1",
                "label_source": "paper_active_set",
                "canonical_smiles": "CCO",
                "lig_file": "chembl_structures/active.sdf",
                "cross_label_conflict_resolved": "0",
            },
            {
                "source_compound_id": f"{target}__random",
                "target_id": target,
                "label": "0",
                "label_source": "paper_random_set",
                "canonical_smiles": "CCC",
                "lig_file": "chembl_structures/random.sdf",
                "cross_label_conflict_resolved": "0",
            },
        ],
        [
            "source_compound_id",
            "target_id",
            "label",
            "label_source",
            "canonical_smiles",
            "lig_file",
            "cross_label_conflict_resolved",
        ],
    )
    members = {
        f"BayesBindBaselinesV1.5/vina/test/{target}/actives_results/results.csv": b"smiles,filename,score\nCCO,0.sdf,8.0\n",
        f"BayesBindBaselinesV1.5/vina/test/{target}/actives_results/0.sdf": b"pose\n",
        f"BayesBindBaselinesV1.5/vina/test/{target}/random_results/results.csv": b"smiles,filename,score\nCCC,0.sdf,6.0\n",
        f"BayesBindBaselinesV1.5/vina/test/{target}/random_results/0.sdf": b"pose\n",
    }
    archive = tmp_path / "BayesBindBaselinesV1.5.tar.gz"
    inventory = tmp_path / "BayesBindBaselinesV1.5.members.jsonl"
    _write_plain_tar(archive, members)
    _write_inventory(inventory, members)

    result = bayesbind_paper.recover_official_bayesbind_baseline(
        campaign,
        method="vina",
        archive_path=archive,
        inventory_path=inventory,
        target_ids=[target],
        expected_archive_sha256=None,
    )

    assert result["status"] == "ready"
    assert result["totals"]["official_result_rows"] == 2
    assert result["totals"]["source_unmatched"] == 0
    assert result["totals"]["pose_members_missing"] == 0
    metrics = list(csv.DictReader(Path(result["outputs"]["official_target_metrics_csv"]).open()))
    assert metrics[0]["official_scored"] == "2"
    assert metrics[0]["intended_pose_retention"] == "1.0"
    assert metrics[0]["roc_auc"] == "1.0"
    rows = list(csv.DictReader(Path(result["outputs"]["official_reference_rows_csv"]).open()))
    assert {row["metric_score_lower_is_better"] for row in rows} == {"-8.0", "-6.0"}





def test_compare_bayesbind_rerun_to_official_subset(tmp_path: Path) -> None:
    target = "TGT_HUMAN_1_2_0"
    intended = tmp_path / "intended.csv"
    bayesbind_paper._write_csv(
        intended,
        [
            {
                "source_compound_id": "S1",
                "target_id": target,
                "label": 1,
                "label_source": "measured_activity_pchembl_gt_cutoff",
                "canonical_smiles": "CCO",
                "lig_file": "a.sdf",
                "cross_label_conflict_resolved": 0,
            },
            {
                "source_compound_id": "S2",
                "target_id": target,
                "label": 0,
                "label_source": "paper_random_set",
                "canonical_smiles": "CCC",
                "lig_file": "b.sdf",
                "cross_label_conflict_resolved": 0,
            },
        ],
        [
            "source_compound_id",
            "target_id",
            "label",
            "label_source",
            "canonical_smiles",
            "lig_file",
            "cross_label_conflict_resolved",
        ],
    )
    output = tmp_path / "rerun"
    output.mkdir()
    bayesbind_paper._write_json(
        output / "target_metrics.json",
        {
            "target_id": target,
            "ef1": 1.0,
            "ef5": 1.0,
            "ef20": 1.0,
            "roc_auc": 1.0,
            "bedroc": 0.5,
            "coverage": 1.0,
            "first_active_rank": 1,
            "pose_retention": 1.0,
            "counts": {"before_prep": 2, "scored": 2},
        },
    )
    bayesbind_paper._write_csv(
        output / "source_ranking.csv",
        [
            {
                "source_compound_id": "S1",
                "target_id": target,
                "label": 1,
                "score": -8.0,
                "scored": 1,
                "failed": 0,
                "pose_retained": 1,
            },
            {
                "source_compound_id": "S2",
                "target_id": target,
                "label": 0,
                "score": -6.0,
                "scored": 1,
                "failed": 0,
                "pose_retained": 1,
            },
        ],
        [
            "source_compound_id",
            "target_id",
            "label",
            "score",
            "scored",
            "failed",
            "pose_retained",
        ],
    )
    official = tmp_path / "official"
    official.mkdir()
    bayesbind_paper._write_json(
        official / "official_baseline_manifest.json",
        {"status": "ready", "method": "vina"},
    )
    bayesbind_paper._write_csv(
        official / "official_target_metrics.csv",
        [
            {
                "target_id": target,
                "method": "vina",
                "roc_auc": 1.0,
                "coverage": 1.0,
                "ef1": 1.0,
            }
        ],
        ["target_id", "method", "roc_auc", "coverage", "ef1"],
    )
    bayesbind_paper._write_csv(
        official / "official_source_ranking.csv",
        [
            {
                "target_id": target,
                "source_compound_id": "S1",
                "label": 1,
                "label_source": "paper_active_set",
                "canonical_smiles": "CCO",
                "official_score": 8.0,
                "metric_score_lower_is_better": -8.0,
                "official_scored": 1,
                "official_failed": 0,
                "pose_member_found": 1,
            },
            {
                "target_id": target,
                "source_compound_id": "S2",
                "label": 0,
                "label_source": "paper_random_set",
                "canonical_smiles": "CCC",
                "official_score": 6.0,
                "metric_score_lower_is_better": -6.0,
                "official_scored": 1,
                "official_failed": 0,
                "pose_member_found": 1,
            },
        ],
        [
            "target_id",
            "source_compound_id",
            "label",
            "label_source",
            "canonical_smiles",
            "official_score",
            "metric_score_lower_is_better",
            "official_scored",
            "official_failed",
            "pose_member_found",
        ],
    )

    result = bayesbind_paper.compare_bayesbind_rerun_to_official(
        output, intended, official, method="vina"
    )

    assert result["status"] == "ready"
    assert result["counts"]["paired_scored"] == 2
    assert result["metric_comparison"]["roc_auc"]["delta"] == 0.0
    assert result["score_summary"]["paired_lower_is_better"]["pearson"] == 1.0
    assert Path(result["paired_scores_csv"]).exists()


def _write_fake_plan_inputs(tmp_path: Path, *, calibration_target_split: str = "development") -> tuple[Path, Path, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    targets = [
        {
            "position": 1,
            "target_id": "DEV_HUMAN_1_2_0",
            "split": calibration_target_split,
            "source_compounds": 1000,
        },
        {
            "position": 2,
            "target_id": "HOLD_HUMAN_1_2_0",
            "split": "held_out_validation",
            "source_compounds": 1000,
        },
    ]
    bayesbind_paper._write_json(campaign / "targets.json", targets)
    bayesbind_paper._write_json(
        campaign / "quota_budget.json",
        {
            "account_count": 2,
            "maximum_concurrent_notebooks_per_account": 2,
            "maximum_planned_gpu_hours": 100.0,
            "reserve_gpu_hours": 20.0,
            "planned_gpu_hours": 0.0,
        },
    )
    output = tmp_path / "calibration_output"
    output.mkdir()
    bayesbind_paper._write_json(
        output / "target_metrics.json",
        {
            "target_id": "DEV_HUMAN_1_2_0",
            "coverage": 1.0,
            "counts": {"before_prep": 100, "scored": 100, "failed": 0},
        },
    )
    bayesbind_paper._write_json(
        output / "vina_baseline_summary.json",
        {"runtime_seconds": 200.0, "scored": 100},
    )
    bayesbind_paper._write_json(
        output / "gnina_baseline_summary.json",
        {"runtime_seconds": 7200.0, "scored": 100},
    )
    stage = tmp_path / "stage_manifest.json"
    bayesbind_paper._write_json(
        stage,
        {"target_id": "DEV_HUMAN_1_2_0", "sample_size": 100, "sample_seed": 20260724},
    )
    return campaign, output, stage


def test_freeze_bayesbind_full_run_plan_from_development_calibration(tmp_path: Path) -> None:
    campaign, output, stage = _write_fake_plan_inputs(tmp_path)

    result = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="vina",
        safety_factor=1.25,
    )

    assert result["status"] == "ready"
    assert result["compute_kind"] == "cpu"
    assert result["calibration"]["intended_compounds"] == 100
    assert result["calibration"]["seconds_per_compound"] == 2.0
    assert result["quota_gate"]["planned_gpu_hours"] == 0.0
    assert result["planning"]["target_count"] == 2
    assert result["planning"]["split_counts"] == {"development": 1, "held_out_validation": 1}
    assert result["planning"]["uses_held_out_for_tuning"] is False
    assert result["targets"][0]["total_notebooks"] == 2
    assert result["targets"][0]["shards"][0]["source_row_start"] == 0
    assert Path(result["plan_path"]).exists()


def test_freeze_bayesbind_plan_uses_compute_specific_concurrency(tmp_path: Path) -> None:
    campaign, output, stage = _write_fake_plan_inputs(tmp_path)
    quota_path = campaign / "quota_budget.json"
    quota = json.loads(quota_path.read_text(encoding="utf-8"))
    quota["maximum_concurrent_cpu_notebooks_per_account"] = 10
    quota["maximum_concurrent_gpu_notebooks_per_account"] = 2
    bayesbind_paper._write_json(quota_path, quota)

    vina = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="vina",
        safety_factor=1.25,
    )
    gnina = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="gnina",
        safety_factor=1.25,
    )

    assert vina["planning"]["maximum_concurrent_notebooks_per_account"] == 10
    assert gnina["planning"]["maximum_concurrent_notebooks_per_account"] == 2


def test_freeze_bayesbind_full_run_plan_applies_runtime_cap(tmp_path: Path) -> None:
    campaign, output, stage = _write_fake_plan_inputs(tmp_path)
    result = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="vina",
        safety_factor=1.0,
        max_shard_wall_hours=0.25,
    )

    assert result["status"] == "ready"
    assert result["runtime_gate"]["runtime_cap_safe"] is True
    assert result["targets"][0]["count_based_notebooks"] == 2
    assert result["targets"][0]["runtime_cap_notebooks"] == 3
    assert result["targets"][0]["total_notebooks"] == 3
    assert max(
        shard["estimated_wall_seconds"]
        for shard in result["targets"][0]["shards"]
    ) <= result["runtime_gate"]["max_shard_wall_seconds"]


def test_freeze_bayesbind_full_run_plan_blocks_gpu_quota_risk(tmp_path: Path) -> None:
    campaign, output, stage = _write_fake_plan_inputs(tmp_path)
    quota = json.loads((campaign / "quota_budget.json").read_text(encoding="utf-8"))
    quota["maximum_planned_gpu_hours"] = 25.0
    quota["reserve_gpu_hours"] = 20.0
    bayesbind_paper._write_json(campaign / "quota_budget.json", quota)

    result = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="gnina",
        safety_factor=1.25,
    )

    assert result["status"] == "blocked"
    assert result["compute_kind"] == "gpu"
    assert result["quota_gate"]["quota_safe"] is False
    assert result["quota_gate"]["planned_gpu_hours"] > result["quota_gate"]["available_gpu_hours_after_reserve"]


def test_freeze_bayesbind_full_run_plan_rejects_held_out_calibration(tmp_path: Path) -> None:
    campaign, output, stage = _write_fake_plan_inputs(
        tmp_path,
        calibration_target_split="held_out_validation",
    )

    with pytest.raises(ValueError, match="development split"):
        bayesbind_paper.freeze_bayesbind_full_run_plan(campaign, output, stage)


def _write_exact_single_target_campaign(tmp_path: Path) -> tuple[Path, str]:
    extracted = tmp_path / "extracted" / "BayesBindV1.5"
    (extracted / "val").mkdir(parents=True)
    (extracted / "test").mkdir(parents=True)
    names = [f"PLAN{i:02d}_HUMAN_1_2_0" for i in range(25)]
    for index, name in enumerate(names):
        split = "val" if index < 14 else "test"
        _write_target(extracted / split / name)
    target = names[0]
    campaign = tmp_path / "campaign_exact"
    freeze_bayesbind(campaign, extracted, random_limit=2, active_cutoff=5.0)
    lig_files = sorted(
        row["lig_file"]
        for target_dir in (campaign / "paper_inputs").iterdir()
        for row in csv.DictReader((target_dir / "paper_rows.csv").open())
    )
    members = {f"BigBindV1.5/{lig_file}": _valid_sdf_bytes() for lig_file in lig_files}
    archive = tmp_path / "BigBindV1.5.tar.gz"
    inventory = tmp_path / "BigBindV1.5.members.jsonl"
    _write_plain_tar(archive, members)
    _write_inventory(inventory, members)
    bayesbind_paper.extract_exact_bayesbind_ligands(
        campaign,
        archive_path=archive,
        inventory_path=inventory,
        expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    binary = campaign / "inputs" / bayesbind_paper.VINA_BINARY_FILENAME
    binary.write_bytes(b"fake-vina")
    return campaign, target


def _write_tiny_calibration_for_plan(tmp_path: Path, target: str) -> tuple[Path, Path]:
    output = tmp_path / "tiny_calibration_output"
    output.mkdir()
    bayesbind_paper._write_json(
        output / "target_metrics.json",
        {
            "target_id": target,
            "coverage": 1.0,
            "counts": {"before_prep": 3, "scored": 3, "failed": 0},
        },
    )
    bayesbind_paper._write_json(output / "vina_baseline_summary.json", {"runtime_seconds": 30.0})
    bayesbind_paper._write_json(output / "gnina_baseline_summary.json", {"runtime_seconds": 30.0})
    stage = tmp_path / "tiny_stage_manifest.json"
    bayesbind_paper._write_json(
        stage,
        {
            "target_id": target,
            "sample_size": 3,
            "sample_seed": 20260724,
            "paper_protocol": {"ligand_workers": 2, "gnina_workers": 1},
        },
    )
    return output, stage


def test_stage_bayesbind_plan_shard_renders_exact_vina_row_range(tmp_path: Path) -> None:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    original = bayesbind_paper.VINA_BINARY_SHA256
    bayesbind_paper.VINA_BINARY_SHA256 = hashlib.sha256(
        (campaign / "inputs" / bayesbind_paper.VINA_BINARY_FILENAME).read_bytes()
    ).hexdigest()
    try:
        plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
            campaign,
            output,
            stage,
            method="vina",
            minimum_calibration_compounds=2,
            maximum_calibration_compounds=3,
        )
        staged = bayesbind_paper.stage_bayesbind_plan_shard(
            campaign,
            Path(plan["plan_path"]),
            target,
            1,
        )
    finally:
        bayesbind_paper.VINA_BINARY_SHA256 = original

    manifest = json.loads(Path(staged["manifest"]).read_text(encoding="utf-8"))
    source_rows = list(
        csv.DictReader(
            (campaign / "paper_inputs" / target / "source_compounds.csv").open()
        )
    )
    intended = list(
        csv.DictReader(
            (Path(staged["stage_dir"]) / f"{staged['run_id']}_intended_compounds.csv").open()
        )
    )
    assert manifest["row_selection"]["mode"] == "source_row_range"
    assert manifest["row_selection"]["source_row_start"] == 0
    assert manifest["row_selection"]["source_row_end_exclusive"] == 2
    assert manifest["full_run_plan"]["shard_index"] == 1
    assert manifest["paper_protocol"]["ligand_workers"] == 2
    assert manifest["exact_sdf_archive_member_count"] == 2
    assert {row["source_compound_id"] for row in intended} == {
        row["source_compound_id"] for row in source_rows[:2]
    }


def test_stage_bayesbind_plan_shard_renders_exact_gnina_row_range(tmp_path: Path) -> None:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="gnina",
        minimum_calibration_compounds=2,
        maximum_calibration_compounds=3,
    )

    staged = bayesbind_paper.stage_bayesbind_plan_shard(
        campaign,
        Path(plan["plan_path"]),
        target,
        2,
    )

    manifest = json.loads(Path(staged["manifest"]).read_text(encoding="utf-8"))
    assert staged["method"] == "gnina"
    assert manifest["engine"] == "gnina"
    assert manifest["row_selection"]["source_row_start"] == 2
    assert manifest["row_selection"]["source_row_end_exclusive"] == 3
    assert manifest["full_run_plan"]["shard_count"] == 1
    assert manifest["exact_sdf_archive_member_count"] == 1


def test_stage_bayesbind_plan_shard_rejects_blocked_plan(tmp_path: Path) -> None:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="vina",
        minimum_calibration_compounds=2,
        maximum_calibration_compounds=3,
    )
    plan_path = Path(plan["plan_path"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["status"] = "blocked"
    bayesbind_paper._write_json(plan_path, payload)

    with pytest.raises(ValueError, match="not ready"):
        bayesbind_paper.stage_bayesbind_plan_shard(campaign, plan_path, target, 1)


def test_submit_bayesbind_plan_shards_registers_vina_finalizer(tmp_path: Path, monkeypatch) -> None:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    original = bayesbind_paper.VINA_BINARY_SHA256
    bayesbind_paper.VINA_BINARY_SHA256 = hashlib.sha256(
        (campaign / "inputs" / bayesbind_paper.VINA_BINARY_FILENAME).read_bytes()
    ).hexdigest()
    calls = []

    def fake_submit(campaign_dir, notebook_path, input_paths, **kwargs):
        calls.append((campaign_dir, notebook_path, input_paths, kwargs))
        return {
            "status": "submitted",
            "job_id": kwargs["job_id"],
            "run_id": kwargs["run_id"],
            "kernel_ref": "owner/kernel",
            "dataset_ref": "owner/data",
            "backend": kwargs["backend"],
            "compute_kind": "cpu",
            "planned_gpu_hours": kwargs["planned_gpu_hours"],
        }

    monkeypatch.setattr("ezscreen.benchmark.kaggle_notebook.submit_notebook", fake_submit)
    try:
        plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
            campaign,
            output,
            stage,
            method="vina",
            minimum_calibration_compounds=2,
            maximum_calibration_compounds=3,
        )
        result = bayesbind_paper.submit_bayesbind_plan_shards(
            campaign,
            Path(plan["plan_path"]),
            target_ids=[target],
            shard_indices=[1],
            max_submissions=1,
        )
    finally:
        bayesbind_paper.VINA_BINARY_SHA256 = original

    assert result["status"] == "submitted"
    assert result["method"] == "vina"
    assert result["backend"] == "cpu"
    assert result["submitted_count"] == 1
    kwargs = calls[0][3]
    assert kwargs["finalizer"] == "bayesbind-vina"
    assert kwargs["planned_gpu_hours"] == 0.0
    assert kwargs["backend"] == "cpu"
    assert kwargs["intended_csv"].name.endswith("_intended_compounds.csv")
    assert Path(kwargs["intended_csv"]).exists()
    assert all(Path(path).is_file() for path in calls[0][2])


def test_submit_bayesbind_plan_shards_uses_gnina_gpu_plan_hours(tmp_path: Path, monkeypatch) -> None:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    calls = []

    def fake_submit(_campaign_dir, _notebook_path, _input_paths, **kwargs):
        calls.append(kwargs)
        return {
            "status": "submitted",
            "job_id": kwargs["job_id"],
            "run_id": kwargs["run_id"],
            "kernel_ref": "owner/kernel",
            "dataset_ref": "owner/data",
            "backend": kwargs["backend"],
            "compute_kind": "gpu",
            "planned_gpu_hours": kwargs["planned_gpu_hours"],
        }

    monkeypatch.setattr("ezscreen.benchmark.kaggle_notebook.submit_notebook", fake_submit)
    plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="gnina",
        minimum_calibration_compounds=2,
        maximum_calibration_compounds=3,
    )

    result = bayesbind_paper.submit_bayesbind_plan_shards(
        campaign,
        Path(plan["plan_path"]),
        target_ids=[target],
        shard_indices=[1],
        max_submissions=1,
    )

    assert result["method"] == "gnina"
    assert result["backend"] == "t4x2"
    assert result["planned_gpu_hours"] > 0.0
    assert calls[0]["finalizer"] == "bayesbind-gnina"
    assert calls[0]["backend"] == "t4x2"
    assert calls[0]["planned_gpu_hours"] == result["planned_gpu_hours"]


def test_submit_bayesbind_plan_shards_rejects_blocked_plan_before_submit(tmp_path: Path, monkeypatch) -> None:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
        campaign,
        output,
        stage,
        method="gnina",
        minimum_calibration_compounds=2,
        maximum_calibration_compounds=3,
    )
    plan_path = Path(plan["plan_path"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["status"] = "blocked"
    bayesbind_paper._write_json(plan_path, payload)
    monkeypatch.setattr(
        "ezscreen.benchmark.kaggle_notebook.submit_notebook",
        lambda *_args, **_kwargs: pytest.fail("submit should not be called"),
    )

    with pytest.raises(ValueError, match="not ready"):
        bayesbind_paper.submit_bayesbind_plan_shards(
            campaign,
            plan_path,
            target_ids=[target],
            shard_indices=[1],
        )


def _write_completed_plan_shards(
    campaign: Path,
    plan: dict,
    target: str,
    *,
    duplicate: bool = False,
    missing_job: bool = False,
    missing_output_dir: bool = False,
) -> None:
    source_rows = list(
        csv.DictReader(
            (campaign / "paper_inputs" / target / "source_compounds.csv").open()
        )
    )
    jobs = []
    for shard in next(row for row in plan["targets"] if row["target_id"] == target)["shards"]:
        run_id = shard["run_id"]
        output = campaign / "work" / run_id / "output"
        output.mkdir(parents=True)
        start = int(shard["source_row_start"])
        end = int(shard["source_row_end_exclusive"])
        rows = source_rows[start:end]
        ranking_rows = []
        for offset, intended in enumerate(rows):
            source_id = intended["source_compound_id"]
            if duplicate and int(shard["shard_index"]) == 2:
                source_id = source_rows[0]["source_compound_id"]
            scored = not (int(shard["shard_index"]) == 2 and offset == 0)
            ranking_rows.append(
                {
                    "source_compound_id": source_id,
                    "target_id": target,
                    "label": intended["label"],
                    "score": -8.0 + start + offset if scored else 0.0,
                    "scored": int(scored),
                    "failed": int(not scored),
                    "pose_retained": int(scored),
                }
            )
        bayesbind_paper._write_csv(
            output / "source_ranking.csv",
            ranking_rows,
            ["source_compound_id", "target_id", "label", "score", "scored", "failed", "pose_retained"],
        )
        bayesbind_paper._write_json(
            output / "target_metrics.json",
            {
                "target_id": target,
                "counts": {
                    "before_prep": len(rows),
                    "after_prep": len(rows),
                    "after_aggregation": len(rows),
                    "scored": sum(int(row["scored"]) for row in ranking_rows),
                    "failed": sum(int(row["failed"]) for row in ranking_rows),
                    "filtered": 0,
                    "unmatched": 0,
                },
            },
        )
        if not (missing_job and int(shard["shard_index"]) == 2):
            job = {"run_id": run_id, "job_id": run_id, "status": "complete", "output_dir": str(output)}
            if missing_output_dir and int(shard["shard_index"]) == 2:
                job.pop("output_dir")
            jobs.append(job)
    bayesbind_paper._write_json(
        campaign / "campaign_state.json",
        {"schema_version": 1, "status": "complete", "jobs": jobs},
    )


def _freeze_tiny_ready_plan(tmp_path: Path, method: str = "vina") -> tuple[Path, str, dict]:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    output, stage = _write_tiny_calibration_for_plan(tmp_path, target)
    original = bayesbind_paper.VINA_BINARY_SHA256
    bayesbind_paper.VINA_BINARY_SHA256 = hashlib.sha256(
        (campaign / "inputs" / bayesbind_paper.VINA_BINARY_FILENAME).read_bytes()
    ).hexdigest()
    try:
        plan = bayesbind_paper.freeze_bayesbind_full_run_plan(
            campaign,
            output,
            stage,
            method=method,
            target_ids=[target],
            minimum_calibration_compounds=2,
            maximum_calibration_compounds=3,
        )
    finally:
        bayesbind_paper.VINA_BINARY_SHA256 = original
    return campaign, target, plan


def test_finalize_bayesbind_plan_target_merges_completed_shards(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target)

    result = bayesbind_paper.finalize_bayesbind_plan_target(
        campaign,
        Path(plan["plan_path"]),
        target,
    )

    assert result["target_id"] == target
    assert result["method"] == "vina"
    assert result["counts"]["before_prep"] == 3
    assert result["counts"]["after_aggregation"] == 3
    assert result["counts"]["scored"] == 2
    assert result["counts"]["failed"] == 1
    assert result["counts"]["planned_shards"] == 2
    assert result["counts"]["completed_shards"] == 2
    assert Path(result["metrics_path"]).exists()
    assert Path(result["ranking_path"]).exists()
    ranking = list(csv.DictReader(Path(result["ranking_path"]).open()))
    failed_scores = [float(row["score"]) for row in ranking if row["scored"] == "0"]
    scored_scores = [float(row["score"]) for row in ranking if row["scored"] == "1"]
    assert failed_scores and failed_scores[0] > max(scored_scores)
    assert Path(result["outputs"]["shard_manifest_csv"]).exists()
    state = json.loads((campaign / "campaign_state.json").read_text(encoding="utf-8"))
    registered = state["finalized_targets"]["vina"][target]
    assert registered["metrics_path"] == result["metrics_path"]
    assert registered["ranking_path"] == result["ranking_path"]
    assert registered["counts"]["after_aggregation"] == 3


def test_finalize_bayesbind_plan_target_requires_all_shards_complete(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target, missing_job=True)

    with pytest.raises(ValueError, match="not complete"):
        bayesbind_paper.finalize_bayesbind_plan_target(campaign, Path(plan["plan_path"]), target)


def test_finalize_bayesbind_plan_target_rejects_missing_output_dir(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target, missing_output_dir=True)

    with pytest.raises(ValueError, match="no output_dir"):
        bayesbind_paper.finalize_bayesbind_plan_target(campaign, Path(plan["plan_path"]), target)


def test_finalize_bayesbind_plan_target_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target, duplicate=True)

    with pytest.raises(ValueError, match="Duplicate source_compound_id"):
        bayesbind_paper.finalize_bayesbind_plan_target(campaign, Path(plan["plan_path"]), target)


def test_audit_bayesbind_campaign_results_accepts_complete_registry(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target)
    finalized = bayesbind_paper.finalize_bayesbind_plan_target(
        campaign,
        Path(plan["plan_path"]),
        target,
    )

    result = bayesbind_paper.audit_bayesbind_campaign_results(
        campaign,
        Path(plan["plan_path"]),
        expected_target_count=1,
    )

    assert result["status"] == "ready"
    assert result["ready_target_count"] == 1
    assert result["finalized_target_count"] == 1
    assert result["total_source_compounds"] == 3
    assert result["total_scored"] == 2
    assert result["total_failed"] == 1
    assert result["issues"] == []
    assert Path(result["audit_path"]).exists()
    assert Path(result["target_audit_csv"]).exists()
    assert result["targets"][0]["metrics_path"] == finalized["metrics_path"]


def test_audit_bayesbind_campaign_results_reports_missing_registry(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target)

    result = bayesbind_paper.audit_bayesbind_campaign_results(
        campaign,
        Path(plan["plan_path"]),
        expected_target_count=1,
    )

    assert result["status"] == "not_ready"
    assert any("missing finalized target registry entry" in issue for issue in result["issues"])
    assert any("missing registry path" in issue for issue in result["issues"])


def test_audit_bayesbind_campaign_results_requires_official_comparison(tmp_path: Path) -> None:
    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target)
    bayesbind_paper.finalize_bayesbind_plan_target(campaign, Path(plan["plan_path"]), target)

    result = bayesbind_paper.audit_bayesbind_campaign_results(
        campaign,
        Path(plan["plan_path"]),
        require_official_comparison=True,
        expected_target_count=1,
    )

    assert result["status"] == "not_ready"
    assert any("official comparison missing" in issue for issue in result["issues"])


def test_cli_audit_bayesbind_campaign_strict_passes_ready_registry(tmp_path: Path) -> None:
    from ezscreen.cli import app

    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target)
    bayesbind_paper.finalize_bayesbind_plan_target(campaign, Path(plan["plan_path"]), target)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "audit-bayesbind-campaign",
            str(campaign),
            str(plan["plan_path"]),
            "--expected-target-count",
            "1",
            "--strict",
        ],
    )

    assert result.exit_code == 0
    assert "status=ready" in result.output


def test_cli_audit_bayesbind_campaign_strict_fails_not_ready(tmp_path: Path) -> None:
    from ezscreen.cli import app

    campaign, target, plan = _freeze_tiny_ready_plan(tmp_path)
    _write_completed_plan_shards(campaign, plan, target)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "audit-bayesbind-campaign",
            str(campaign),
            str(plan["plan_path"]),
            "--expected-target-count",
            "1",
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "status=not_ready" in result.output


def _write_preflight_ready_to_submit_campaign(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    campaign, target = _write_exact_single_target_campaign(tmp_path)
    inputs = campaign / "inputs"
    bigbind_archive = inputs / "BigBindV1.5.tar.gz"
    baselines_archive = inputs / "BayesBindBaselinesV1.5.tar.gz"
    bigbind_archive.write_bytes(b"bigbind")
    baselines_archive.write_bytes(b"baselines")
    monkeypatch.setattr(bayesbind_paper, "BIGBIND_ARCHIVE_BYTES", bigbind_archive.stat().st_size)
    monkeypatch.setattr(bayesbind_paper, "BASELINES_ARCHIVE_BYTES", baselines_archive.stat().st_size)
    (inputs / "archive_manifests").mkdir(exist_ok=True)
    bayesbind_paper._write_json(
        inputs / "archive_manifests" / "official_archives_manifest.json",
        {"archives": [{"path": str(bigbind_archive)}, {"path": str(baselines_archive)}]},
    )
    official_dir = inputs / "official_baselines" / "vina"
    official_dir.mkdir(parents=True)
    bayesbind_paper._write_json(
        official_dir / "official_baseline_manifest.json",
        {"method": "vina", "status": "ready"},
    )
    bayesbind_paper._write_json(
        campaign / "quota_budget.json",
        {
            "maximum_planned_gpu_hours": 100.0,
            "reserve_gpu_hours": 20.0,
            "planned_gpu_hours": 0.0,
            "observed_gpu_hours": 0.0,
        },
    )
    bayesbind_paper._write_json(
        campaign / "campaign_state.json",
        {"schema_version": 1, "status": "initialized", "jobs": []},
    )
    stage_dir = campaign / "work" / "bayesbind-vina-exact-sdf-calib-200-test" / "input"
    stage_dir.mkdir(parents=True)
    bayesbind_paper._write_json(
        stage_dir / "stage_manifest.json",
        {
            "run_id": "bayesbind-vina-exact-sdf-calib-200-test",
            "target_id": target,
            "sample_size": 100,
            "sample_seed": 20260724,
            "exact_sdf_archive_member_count": 100,
        },
    )
    return campaign, target


def test_preflight_bayesbind_execution_reports_ready_to_submit_calibration(tmp_path: Path, monkeypatch) -> None:
    campaign, _target = _write_preflight_ready_to_submit_campaign(tmp_path, monkeypatch)

    result = bayesbind_paper.preflight_bayesbind_execution(
        campaign,
        method="vina",
        minimum_free_gb_warning=0.0,
    )

    assert result["status"] == "ready_to_submit_calibration"
    assert result["next_action"] == "submit and supervise vina calibration"
    assert result["issues"] == []
    assert result["full_run_plan_exists"] is False
    assert Path(result["preflight_path"]).exists()
    gates = {row["name"]: row["status"] for row in result["gates"]}
    assert gates["exact_ligands"] == "pass"
    assert gates["calibration_staged"] == "pass"
    assert gates["calibration_complete"] == "wait"



def test_preflight_bayesbind_execution_uses_nondefault_ready_plan(tmp_path: Path, monkeypatch) -> None:
    campaign, target = _write_preflight_ready_to_submit_campaign(tmp_path, monkeypatch)
    plan_dir = campaign / "full_run_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "bayesbind_vina_full_run_plan_r3.json"
    run_id = "bayesbind-vina-exact-sdf-full-target-001-of-001"
    bayesbind_paper._write_json(
        plan_path,
        {
            "schema_version": 1,
            "method": "vina",
            "status": "ready",
            "targets": [
                {
                    "target_id": target,
                    "source_compounds": 1,
                    "shards": [
                        {"run_id": run_id, "shard_index": 1, "shard_count": 1}
                    ],
                }
            ],
        },
    )
    output_dir = campaign / "work" / run_id / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "target_metrics.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "vina_baseline_summary.json").write_text("{}\n", encoding="utf-8")
    bayesbind_paper._write_json(
        campaign / "campaign_state.json",
        {
            "schema_version": 1,
            "status": "running",
            "jobs": [
                {
                    "job_id": run_id,
                    "run_id": run_id,
                    "status": "running",
                    "kernel_ref": "owner/kernel",
                }
            ],
        },
    )

    result = bayesbind_paper.preflight_bayesbind_execution(
        campaign,
        method="vina",
        minimum_free_gb_warning=0.0,
    )

    assert result["full_run_plan_exists"] is True
    assert result["full_run_plan_path"] == str(plan_path)
    assert result["status"] == "full_shards_running_or_incomplete"
    assert result["next_action"] == "continue supervisor polling/recovery"

def test_cli_preflight_bayesbind_execution_writes_next_action(tmp_path: Path, monkeypatch) -> None:
    from ezscreen.cli import app

    campaign, _target = _write_preflight_ready_to_submit_campaign(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "preflight-bayesbind-execution",
            str(campaign),
            "--method",
            "vina",
            "--minimum-free-gb-warning",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "status=ready_to_submit_calibration" in result.output
    assert "submit and supervise vina calibration" in result.output
