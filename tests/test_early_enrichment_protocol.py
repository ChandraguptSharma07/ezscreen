from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ezscreen.backends.kaggle import runner
from ezscreen.benchmark import early_enrichment_prep_runtime as prep_runtime
from ezscreen.benchmark.early_enrichment import (
    ARM_IDS,
    ProtocolContractError,
    aggregate_raw_vina_scores,
    assemble_constituent_rows,
    compute_early_enrichment_metrics,
    load_protocol_freeze,
    render_matrix,
    stratified_bootstrap_intervals,
)
from ezscreen.benchmark.early_enrichment_bundle import (
    PREPARATION_ARM,
    pack_t4_preparation_dataset,
    refresh_t4_preparation_transport,
    stage_t4_preparation_bundle,
)
from ezscreen.benchmark.early_enrichment_prep_runtime import (
    L1_COORDINATE_TOLERANCE_ANGSTROM,
    heavy_atom_coordinate_drift,
    require_mmff94_convergence,
)

FREEZE = Path(
    "benchmarks/vina_family_ef_literature_review/optimization_protocol_freeze_v1.json"
)
requires_protocol_freeze = pytest.mark.skipif(
    not FREEZE.exists(),
    reason="protocol freeze record is kept outside the repository",
)


def _joined_source(notebook_text: str) -> str:
    notebook = json.loads(notebook_text)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _embedded_molecule(smiles: str):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 42
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    return molecule


def test_l1_coordinate_qc_is_symmetry_aware_and_absolute():
    source = _embedded_molecule("c1ccccc1")
    reversed_atoms = list(reversed(range(source.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(source, reversed_atoms)
    assert heavy_atom_coordinate_drift(source, reordered) < 1e-9

    shifted = Chem.Mol(source)
    conformer = shifted.GetConformer()
    for atom_index in range(shifted.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_index)
        position.x += 0.02
        conformer.SetAtomPosition(atom_index, position)
    drift = heavy_atom_coordinate_drift(source, shifted)
    assert drift == pytest.approx(0.02, abs=1e-6)
    assert drift > L1_COORDINATE_TOLERANCE_ANGSTROM


def test_l2_requires_mmff94_convergence():
    assert require_mmff94_convergence(0) == 0
    with pytest.raises(RuntimeError, match="return code 1"):
        require_mmff94_convergence(1)
    with pytest.raises(RuntimeError, match="return code -1"):
        require_mmff94_convergence(-1)


def test_preparation_command_ledger_records_exact_argv_and_bounded_tail(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prep_runtime, "WORKING", tmp_path)
    prep_runtime.COMMAND_LOG.clear()
    result = prep_runtime.run(
        [sys.executable, "-c", "print('x' * 2500)"],
        timeout=30,
    )
    assert result.returncode == 0
    ledger = json.loads((tmp_path / "command_ledger.json").read_text())
    command = ledger["commands"][0]
    assert command["argv"] == [sys.executable, "-c", "print('x' * 2500)"]
    assert command["return_code"] == 0
    assert len(command["output_tail"]) == prep_runtime.LOG_TAIL_LIMIT


def test_environment_manifest_whitelists_container_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_runtime, "WORKING", tmp_path)
    monkeypatch.setenv("SECRET_TEST_CREDENTIAL", "must-not-appear")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(
        prep_runtime,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="example==1.0\n"
        ),
    )
    prep_runtime.EXECUTABLES.clear()
    prep_runtime.write_environment_provenance(
        {
            "study_id": "test-study",
            "target_id": "SYNTHETIC",
            "preparation_arm": "B0_frozen_baseline",
        }
    )
    manifest_text = (tmp_path / "environment_manifest.json").read_text()
    assert "must-not-appear" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["container_identifiers"] == {
        "KAGGLE_KERNEL_RUN_TYPE": "Interactive"
    }
    assert manifest["pip_freeze"]["sha256"] == hashlib.sha256(
        (tmp_path / "pip_freeze.txt").read_bytes()
    ).hexdigest()
    executables = json.loads((tmp_path / "executable_manifest.json").read_text())
    assert executables["executables"]["python"]["sha256"]


def test_b0_failure_reason_preserves_return_code_and_output_tail(
    tmp_path, monkeypatch
):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    monkeypatch.setattr(prep_runtime, "PREPARED", prepared)
    monkeypatch.setattr(
        prep_runtime,
        "install_mgltools",
        lambda bundle_root, contract: (Path("pythonsh"), Path("prepare_ligand4.py")),
    )
    monkeypatch.setattr(
        prep_runtime,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, stdout="specific ADT failure"
        ),
    )
    rows = [
        {
            "source_key": "synthetic@0",
            "source_id": "synthetic",
            "sample_index": 0,
            "label": 0,
            "source_sha256": "0" * 64,
            "_path": "source.mol2",
        }
    ]
    result = prep_runtime.prepare_b0(
        rows,
        tmp_path,
        {"preparation_arm": "B0_frozen_baseline"},
    )
    assert result[0]["preparation_status"] == "failed"
    assert "return_code=7" in result[0]["failure_reason"]
    assert "specific ADT failure" in result[0]["failure_reason"]


@requires_protocol_freeze
def test_repository_freeze_validates_and_renders_nine_static_specs():
    document = load_protocol_freeze(FREEZE)
    specs = render_matrix(document)

    assert [arm["id"] for arm in document["arms"]] == list(ARM_IDS)
    l1_prep = document["arms"][1]["ligand_preparation"]
    l2_prep = document["arms"][2]["ligand_preparation"]
    assert l1_prep["coordinate_tolerance_angstrom"] == 0.01
    assert l2_prep["accepted_mmff_return_codes"] == [0]
    assert document["provenance_contract"]["log_tail_limit"] == 2000
    assert "docking_command_ledger.json" in document["provenance_contract"][
        "required_docking_outputs"
    ]
    assert "every finite raw Vina affinity" in document["engine"]["score_validity_policy"]
    assert document["metric_contract"]["within_target_uncertainty"] == {
        "method": "stratified nonparametric percentile bootstrap over actives and decoys",
        "replicates": 10000,
        "seed": 20260801,
        "confidence": 0.95,
    }
    assert len(specs) == 9
    assert [s["render_kwargs"]["seed"] for s in specs[-3:]] == [104729, 130363, 155921]
    assert [s["render_kwargs"]["exhaustiveness"] for s in specs[-3:]] == [11, 11, 11]


@pytest.mark.parametrize(
    ("arm_index", "field", "value", "message"),
    [
        (1, "coordinate_tolerance_angstrom", 0.02, "L1 coordinate"),
        (2, "accepted_mmff_return_codes", [0, 1], "L2 MMFF94"),
    ],
)
@requires_protocol_freeze
def test_freeze_rejects_changed_preparation_amendments(
    arm_index, field, value, message
):
    document = json.loads(FREEZE.read_text(encoding="utf-8"))
    document["arms"][arm_index]["ligand_preparation"][field] = value
    with pytest.raises(ProtocolContractError, match=message):
        render_matrix(document)


@requires_protocol_freeze
def test_all_static_specs_render_json_and_compile_without_submission():
    document = load_protocol_freeze(FREEZE)
    for number, spec in enumerate(render_matrix(document)):
        rendered = runner.render_vina_notebook(
            run_id=f"static-{number}",
            shard_filenames=["prepared_000.pdbqt"],
            box_center=[1.0, 2.0, 3.0],
            box_size=[20.0, 21.0, 22.0],
            accelerator="nvidiaTeslaT4",
            **spec["render_kwargs"],
        )
        notebook = json.loads(rendered)
        joined = _joined_source(rendered)
        kwargs = spec["render_kwargs"]
        assert f"SEED = {kwargs['seed']}" in joined
        assert f"EXHAUSTIVENESS = {kwargs['exhaustiveness']}" in joined
        assert "'--seed', str(SEED)" in joined
        assert "'--num_modes', str(NUM_MODES)" in joined
        assert "'--max_step', str(MAX_STEP)" in joined
        assert "SCORING = 'vina'" in joined
        assert "RETAIN_ALL_FINITE_SCORES = True" in joined
        assert "not RETAIN_ALL_FINITE_SCORES and score >= _SCORE_CEIL" in joined
        assert "docking_command_ledger.json" in joined
        assert "docking_executable_manifest.json" in joined
        assert "docking_inputs_manifest.json" in joined
        assert "docking_artifact_manifest.json" in joined
        assert "pip_freeze.txt" in joined
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if source.strip():
                compile(source, "static_early_enrichment_cell.py", "exec")


def test_fail_closed_enumeration_notebook_compiles():
    rendered = runner.render_vina_notebook(
        run_id="fail-closed-static",
        shard_filenames=["raw_000.smi"],
        box_center=[1.0, 2.0, 3.0],
        box_size=[20.0, 21.0, 22.0],
        prep_on_kaggle=True,
        enumerate_opts={
            "enabled": True,
            "protonation": True,
            "tautomers": True,
            "stereo": False,
            "ring": False,
            "max_variants": 4,
            "fail_closed": True,
        },
    )
    notebook = json.loads(rendered)
    joined = _joined_source(rendered)
    assert "Fail-closed enumeration failed" in joined
    assert "enumeration_no_valid_state" in joined
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.strip():
            compile(source, "fail_closed_enumeration_cell.py", "exec")


def test_aggregation_preserves_duplicate_source_ids_and_selects_best_raw_vina():
    intended = ["ZINC1@0", "ZINC1@1", "ZINC2@2"]
    rows = [
        {"source_key": "ZINC1@0", "state_index": 0, "search_index": 0, "score": -7.0, "engine": "unidock", "scoring": "vina"},
        {"source_key": "ZINC1@0", "state_index": 1, "search_index": 0, "score": -8.2, "engine": "unidock", "scoring": "vina"},
        {"source_key": "ZINC1@1", "state_index": 0, "search_index": 0, "score": -6.5, "engine": "unidock", "scoring": "vina"},
        {"source_key": "ZINC2@2", "state_index": 0, "search_index": 0, "score": "nan", "reason": "non_finite", "engine": "unidock", "scoring": "vina"},
    ]

    aggregated = aggregate_raw_vina_scores(rows, intended)

    assert [row["source_key"] for row in aggregated] == intended
    assert aggregated[0]["score"] == -8.2
    assert aggregated[0]["selected_state_index"] == 1
    assert aggregated[1]["score"] == -6.5
    assert aggregated[2]["status"] == "failed"
    assert aggregated[2]["failure_reasons"] == ["non_finite"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("engine", "vina", "mixes docking engines"), ("scoring", "vinardo", "mixes scoring")],
)
def test_aggregation_rejects_mixed_engine_or_scoring(field, value, message):
    row = {
        "source_key": "ZINC1@0",
        "state_index": 0,
        "search_index": 0,
        "score": -7.0,
        "engine": "unidock",
        "scoring": "vina",
    }
    row[field] = value
    with pytest.raises(ProtocolContractError, match=message):
        aggregate_raw_vina_scores([row], ["ZINC1@0"])


def test_aggregation_rejects_duplicate_constituent_rows():
    row = {
        "source_key": "ZINC1@0",
        "state_index": 0,
        "search_index": 0,
        "score": -7.0,
        "engine": "unidock",
        "scoring": "vina",
    }
    with pytest.raises(ProtocolContractError, match="duplicate constituent"):
        aggregate_raw_vina_scores([row, row], ["ZINC1@0"])


def test_constituent_assembler_preserves_state_search_and_failures():
    prepared = [
        {
            "source_key": "A@0",
            "state_index": 0,
            "ligand": "lig_a0",
            "preparation_status": "prepared",
            "prepared_sha256": "a" * 64,
        },
        {
            "source_key": "A@0",
            "state_index": 1,
            "ligand": "",
            "preparation_status": "failed",
            "failure_reason": "state_invalid",
        },
        {
            "source_key": "D@1",
            "state_index": 0,
            "ligand": "lig_d0",
            "preparation_status": "prepared",
        },
    ]
    scores = [{"ligand": "lig_a0", "score": -8.0}]

    rows = assemble_constituent_rows(prepared, scores, search_index=2)

    assert rows[0]["score"] == -8.0
    assert rows[0]["search_index"] == 2
    assert rows[1]["reason"] == "state_invalid"
    assert rows[2]["reason"] == "docking_missing_score"
    assert {row["engine"] for row in rows} == {"unidock"}
    assert {row["scoring"] for row in rows} == {"vina"}


def test_constituent_assembler_rejects_unknown_score_ligand():
    prepared = [
        {
            "source_key": "A@0",
            "state_index": 0,
            "ligand": "lig_a0",
            "preparation_status": "prepared",
        }
    ]
    with pytest.raises(ProtocolContractError, match="absent"):
        assemble_constituent_rows(
            prepared,
            [{"ligand": "wrong", "score": -7.0}],
            search_index=0,
        )


def test_complete_metric_contract_places_failures_last_and_reports_coverage():
    aggregated = [
        {"source_key": "A@0", "score": -9.0, "status": "scored"},
        {"source_key": "D1@1", "score": -8.0, "status": "scored"},
        {"source_key": "D2@2", "score": None, "status": "failed"},
        {"source_key": "D3@3", "score": None, "status": "failed"},
    ]
    metrics = compute_early_enrichment_metrics(
        aggregated,
        {"A@0": 1, "D1@1": 0, "D2@2": 0, "D3@3": 0},
    )

    assert metrics["ef1"] == 4.0
    assert metrics["ef5"] == 4.0
    assert metrics["ef10"] == 4.0
    assert metrics["coverage"] == 0.5
    assert metrics["failure_count"] == 2
    assert metrics["first_active_rank"] == 1
    assert set(("roc_auc", "pr_auc", "average_precision", "bedroc20")) <= set(metrics)


def test_stratified_bootstrap_is_reproducible_on_synthetic_rows():
    aggregated = [
        {"source_key": f"A{i}@{i}", "score": -9.0 + i, "status": "scored"}
        for i in range(3)
    ] + [
        {
            "source_key": f"D{i}@{i + 3}",
            "score": -6.0 + i,
            "status": "scored",
        }
        for i in range(5)
    ]
    labels = {
        **{f"A{i}@{i}": 1 for i in range(3)},
        **{f"D{i}@{i + 3}": 0 for i in range(5)},
    }

    first = stratified_bootstrap_intervals(
        aggregated, labels, replicates=20, seed=17
    )
    second = stratified_bootstrap_intervals(
        aggregated, labels, replicates=20, seed=17
    )

    assert first == second
    assert first["replicates"] == 20
    assert set(first["intervals"]) == {
        "roc_auc",
        "bedroc20",
        "pr_auc",
        "average_precision",
        "ef1",
        "ef5",
        "ef10",
        "ef20",
        "coverage",
    }


@pytest.mark.parametrize("arm_id", list(ARM_IDS))
@requires_protocol_freeze
def test_guarded_t4_preparation_bundle_stages_without_execution(
    tmp_path, arm_id
):
    source = tmp_path / "source.mol2"
    source.write_text(
        """@<TRIPOS>MOLECULE
synthetic
 1 0 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3       1 LIG 0.0000
@<TRIPOS>BOND
"""
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    document = load_protocol_freeze(FREEZE)
    output = tmp_path / f"bundle-{arm_id}"

    manifest = stage_t4_preparation_bundle(
        document,
        arm_id=arm_id,
        target_id="SYNTHETIC",
        pdb_id="0XXX",
        records=[
            {
                "source_id": "duplicate-safe-id",
                "sample_index": 0,
                "label": 1,
                "source_sha256": digest,
                "mol2_path": source,
            }
        ],
        output_dir=output,
    )

    assert manifest["launch_authorized"] is False
    assert manifest["preparation_arm"] == PREPARATION_ARM[arm_id]
    assert manifest["contains_raw_mol2"] is True
    notebook = json.loads((output / "prepare_t4.ipynb").read_text())
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    joined = _joined_source((output / "prepare_t4.ipynb").read_text())
    assert "Preparation execution is not authorized" in joined
    assert "Tesla T4 only" in joined
    assert not (output / "prepared_bundle.zip").exists()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]),
                "synthetic_preparation_bundle_cell.py",
                "exec",
            )


@requires_protocol_freeze
def test_preparation_bundle_rejects_hash_drift(tmp_path):
    source = tmp_path / "source.mol2"
    source.write_text("drift")
    document = load_protocol_freeze(FREEZE)
    with pytest.raises(ProtocolContractError, match="hash mismatch"):
        stage_t4_preparation_bundle(
            document,
            arm_id="B0_frozen_baseline",
            target_id="SYNTHETIC",
            pdb_id="0XXX",
            records=[
                {
                    "source_id": "id",
                    "sample_index": 0,
                    "label": 0,
                    "source_sha256": "0" * 64,
                    "mol2_path": source,
                }
            ],
            output_dir=tmp_path / "bundle",
        )


@requires_protocol_freeze
def test_synthetic_runtime_authorization_is_narrow(tmp_path):
    source = tmp_path / "source.mol2"
    source.write_text("synthetic")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    document = load_protocol_freeze(FREEZE)
    record = {
        "source_id": "synthetic-ethanol",
        "sample_index": 0,
        "label": 0,
        "source_sha256": digest,
        "mol2_path": source,
    }
    manifest = stage_t4_preparation_bundle(
        document,
        arm_id="L1_meeko_single_state",
        target_id="SYNTHETIC",
        pdb_id="NONE",
        records=[record],
        output_dir=tmp_path / "synthetic-bundle",
        execution_authorization="synthetic_runtime_validation",
    )
    assert manifest["launch_authorized"] is True
    notebook_path = tmp_path / "synthetic-bundle/prepare_t4.ipynb"
    notebook_text = notebook_path.read_text()
    notebook = json.loads(notebook_text)
    assert notebook["metadata"]["swiftscreen"]["launch_authorized"] is True
    install_source = "".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "gemmi=={versions['gemmi']}" in install_source
    assert "bootstrap_commands.json" in install_source
    assert "command_ledger.json" in notebook_text
    assert "artifact_manifest.json" in notebook_text
    assert "archive_manifest.json" in notebook_text

    with pytest.raises(ProtocolContractError, match="limited"):
        stage_t4_preparation_bundle(
            document,
            arm_id="L1_meeko_single_state",
            target_id="ACE",
            pdb_id="1O86",
            records=[record],
            output_dir=tmp_path / "forbidden-real-target",
            execution_authorization="synthetic_runtime_validation",
        )


@requires_protocol_freeze
def test_development_runtime_authorization_is_target_and_count_locked(tmp_path):
    source = tmp_path / "source.mol2"
    source.write_text("development")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    document = load_protocol_freeze(FREEZE)
    document = json.loads(json.dumps(document))
    document["execution"]["development_target_manifest"] = [
        {
            "target_id": "ACE",
            "pdb_id": "1O86",
            "library_stem": "ace",
            "actives": 1,
            "decoys": 0,
        }
    ]
    record = {
        "source_id": "development-compound",
        "sample_index": 0,
        "label": 1,
        "source_sha256": digest,
        "mol2_path": source,
    }
    manifest = stage_t4_preparation_bundle(
        document,
        arm_id="L1_meeko_single_state",
        target_id="ACE",
        pdb_id="1O86",
        records=[record],
        output_dir=tmp_path / "authorized-development",
        execution_authorization="development_execution",
    )
    assert manifest["launch_authorized"] is True
    assert manifest["authorization_scope"] == "development_execution"

    with pytest.raises(ProtocolContractError, match="excludes holdout"):
        stage_t4_preparation_bundle(
            document,
            arm_id="L1_meeko_single_state",
            target_id="ALR2",
            pdb_id="1AH3",
            records=[record],
            output_dir=tmp_path / "forbidden-holdout",
            execution_authorization="development_execution",
        )

    bad_record = {**record, "label": 0}
    with pytest.raises(ProtocolContractError, match="counts, labels"):
        stage_t4_preparation_bundle(
            document,
            arm_id="L1_meeko_single_state",
            target_id="ACE",
            pdb_id="1O86",
            records=[bad_record],
            output_dir=tmp_path / "forbidden-count-drift",
            execution_authorization="development_execution",
        )


@requires_protocol_freeze
def test_authorized_preparation_bundle_packs_hash_locked_transport(tmp_path):
    source = tmp_path / "source.mol2"
    source.write_text("synthetic")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    document = load_protocol_freeze(FREEZE)
    output = tmp_path / "packed"
    stage_t4_preparation_bundle(
        document,
        arm_id="L1_meeko_single_state",
        target_id="SYNTHETIC",
        pdb_id="NONE",
        records=[
            {
                "source_id": "synthetic-packed",
                "sample_index": 0,
                "label": 0,
                "source_sha256": digest,
                "mol2_path": source,
            }
        ],
        output_dir=output,
        execution_authorization="synthetic_runtime_validation",
    )
    packed = pack_t4_preparation_dataset(output)
    archive = Path(packed["archive"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == packed["transport"][
        "sha256"
    ]
    with zipfile.ZipFile(archive) as bundle:
        assert sorted(bundle.namelist()) == [
            "early_enrichment_prep_runtime.py",
            "intended_sources.csv",
            "preparation_contract.json",
            "source_mol2/000000.mol2",
        ]
    notebook = json.loads(Path(packed["notebook"]).read_text(encoding="utf-8"))
    joined = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert packed["transport"]["sha256"] in joined
    assert "Unsafe preparation archive member" in joined
    assert "Expanded preparation member hash mismatch" in joined
    refreshed = refresh_t4_preparation_transport(output)
    assert refreshed["transport"]["members_sha256"] == packed["transport"][
        "members_sha256"
    ]


@requires_protocol_freeze
def test_authorized_b0_bundle_requires_hash_checked_mgltools_source(tmp_path):
    source = tmp_path / "source.mol2"
    source.write_text("synthetic")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    document = load_protocol_freeze(FREEZE)
    record = {
        "source_id": "synthetic-ethanol",
        "sample_index": 0,
        "label": 0,
        "source_sha256": digest,
        "mol2_path": source,
    }
    missing_document = json.loads(json.dumps(document))
    missing_document["toolchain"].pop("mgltools_runtime_url")
    with pytest.raises(ProtocolContractError, match="requires"):
        stage_t4_preparation_bundle(
            missing_document,
            arm_id="B0_frozen_baseline",
            target_id="SYNTHETIC",
            pdb_id="NONE",
            records=[record],
            output_dir=tmp_path / "missing-runtime",
            execution_authorization="synthetic_runtime_validation",
        )

    url_manifest = stage_t4_preparation_bundle(
        document,
        arm_id="B0_frozen_baseline",
        target_id="SYNTHETIC",
        pdb_id="NONE",
        records=[record],
        output_dir=tmp_path / "with-frozen-url",
        execution_authorization="synthetic_runtime_validation",
    )
    assert url_manifest["launch_authorized"] is True

    runtime = tmp_path / "mgltools-test.tar.gz"
    runtime.write_bytes(b"frozen-runtime")
    test_document = json.loads(json.dumps(document))
    test_document["toolchain"]["mgltools_runtime_filename"] = runtime.name
    test_document["toolchain"]["mgltools_runtime_sha256"] = hashlib.sha256(
        runtime.read_bytes()
    ).hexdigest()
    manifest = stage_t4_preparation_bundle(
        test_document,
        arm_id="B0_frozen_baseline",
        target_id="SYNTHETIC",
        pdb_id="NONE",
        records=[record],
        output_dir=tmp_path / "with-runtime",
        execution_authorization="synthetic_runtime_validation",
        mgltools_runtime_path=runtime,
    )
    assert manifest["files_sha256"][runtime.name] == test_document["toolchain"][
        "mgltools_runtime_sha256"
    ]
