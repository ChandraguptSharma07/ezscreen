from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "benchmarks/bench-20260801-ada-1ndw-vina-t4-validation"


def load_renderer():
    path = CAMPAIGN / "render_ada_t4_campaign.py"
    spec = importlib.util.spec_from_file_location("render_ada_t4_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ada_evidence_is_preregistered_and_primary_rows_are_frozen() -> None:
    evidence = json.loads((CAMPAIGN / "target_evidence.json").read_text())
    assert evidence["selection"]["selected_before_pipeline_outcome"] is True
    assert evidence["selection"]["target_id"] == "ada"
    assert evidence["selection"]["pdb_id"] == "1NDW"
    assert evidence["supplement_rows_verified_directly"]["s2"]["row"] == [
        "ada",
        "1ndw",
        39,
        927,
    ]
    assert evidence["supplement_rows_verified_directly"]["s3"]["row"][2] == 0.451
    assert evidence["supplement_rows_verified_directly"]["s4"]["row"][2] == 0.005


def test_ada_box_and_protocol_are_target_specific_and_frozen() -> None:
    evidence = json.loads((CAMPAIGN / "target_evidence.json").read_text())
    protocol = json.loads((CAMPAIGN / "campaign_protocol.json").read_text())
    assert evidence["box"]["center"] == [42.2931, 17.3816, 49.5368]
    assert evidence["box"]["size"] == [36.9618, 39.689, 38.3826]
    assert protocol["protocol"]["box_center"] == evidence["box"]["center"]
    assert protocol["protocol"]["box_size"] == evidence["box"]["size"]
    assert protocol["protocol"]["exhaustiveness"] == 32
    assert protocol["protocol"]["grid_spacing_angstrom"] == 0.375
    assert protocol["protocol"]["scoring"] == "vina"
    assert protocol["protocol"]["record_identity"] == "source_id@sample_index"
    repair = evidence["receptor_input_normalization"]
    assert repair["atoms_added"] == 0
    assert repair["atoms_removed"] == 0
    assert repair["coordinates_changed"] == 0
    assert "element columns" in protocol["protocol"]["receptor_input_normalization"]
    assert protocol["hardware"] == {
        "required_gpu_count": 2,
        "required_model": "Tesla T4",
        "fail_fast_before_preparation_or_docking": True,
    }


def test_renderer_places_hardware_gate_before_preparation() -> None:
    renderer = load_renderer()
    path = renderer.render_preparation()
    notebook = json.loads(path.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    hardware_index = next(i for i, source in enumerate(sources) if "nvidia-smi" in source)
    prep_index = next(i for i, source in enumerate(sources) if "prepare_receptor4.py" in source)
    assert hardware_index < prep_index
    assert 'len(_gpu_rows) != 2' in sources[hardware_index]
    assert 'any("T4" not in model' in sources[hardware_index]
    assert "prepare_ligand4.py" in "\n".join(sources)
    assert "meeko_fallback" in "\n".join(sources)
    assert 'line.ljust(76) + "ZN"' in "\n".join(sources)
    assert "expected one blank-element ADA ZN atom" in "\n".join(sources)


def test_renderer_freezes_expected_unidock_source_and_scoring() -> None:
    renderer = load_renderer()
    assert renderer.UNIDOCK_COMMIT == "ebde3b4a22c8b2c7b26e523cc15109aab815be21"
    assert renderer.GRID_POINT_CAP == 2_097_152
    assert '"--scoring", "vina"' in renderer.DOCK
    assert '"--exhaustiveness", "32"' in renderer.DOCK
    assert '"--spacing", "0.375"' in renderer.DOCK
    assert "source_id@sample_index" in renderer.PREPARE


def test_shard_bounds_account_for_all_records_without_overlap() -> None:
    bounds = [(i * 966 // 12, (i + 1) * 966 // 12) for i in range(12)]
    indices = [index for start, end in bounds for index in range(start, end)]
    assert indices == list(range(966))
    assert min(end - start for start, end in bounds) == 80
    assert max(end - start for start, end in bounds) == 81


def test_frozen_prepared_manifest_has_complete_t4_accounting() -> None:
    frozen = json.loads((CAMPAIGN / "prepared_input_manifest.json").read_text())
    assert frozen["prepared_on_required_hardware"] is True
    assert frozen["hardware"]["gpu_models"] == ["Tesla T4", "Tesla T4"]
    assert frozen["records"] == {
        "intended": 966,
        "actives": 39,
        "decoys": 927,
        "prepared": 966,
        "preparation_failures": 0,
        "fallbacks": 0,
    }
    assert frozen["prepared_bundle_sha256"] == "4d307b979c1ab7560d4b12703d411c37795fbe5d536995705da8deba0d27dcfb"
    assert frozen["receptor_pdbqt_sha256"] == "e35617b87b8ecd15dc5b47fcc0427258e7ef7072e042b6993f471bcba3334078"


def test_all_docking_notebooks_compile_and_gate_hardware_first() -> None:
    renderer = load_renderer()
    paths = renderer.render_docking()
    assert len(paths) == 12
    for shard, path in enumerate(paths):
        notebook = json.loads(path.read_text())
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        hardware_index = next(i for i, source in enumerate(sources) if "nvidia-smi" in source)
        build_index = next(i for i, source in enumerate(sources) if "git clone" in source)
        dock_index = next(i for i, source in enumerate(sources) if '"--scoring", "vina"' in source)
        assert hardware_index < build_index < dock_index
        joined = "\n".join(sources)
        assert "4d307b979c1ab7560d4b12703d411c37795fbe5d536995705da8deba0d27dcfb" in joined
        assert f"SHARD_INDEX = {shard}" in joined
        assert "--exhaustiveness" in joined
        assert "prepared_bundle.kaggle" in joined


def test_launcher_preserves_frozen_zip_bytes_under_opaque_transport_name() -> None:
    source = (CAMPAIGN / "launch_ada_t4.py").read_text()
    assert 'PREPARED_BUNDLE_TRANSPORT = HERE / "inputs/prepared_bundle.kaggle"' in source
    assert "sha256(PREPARED_BUNDLE_TRANSPORT) != expected" in source
    assert "archive_failed_output(job_work, retry_count)" in source


def test_finalizer_freezes_published_values_and_similarity_margins() -> None:
    source = (CAMPAIGN / "finalize_ada_t4.py").read_text()
    assert 'PUBLISHED = {"roc_auc": 0.451, "bedroc20": 0.005}' in source
    assert 'MARGINS = {"roc_auc": -0.02, "bedroc20": -0.02}' in source
    assert "BOOTSTRAP_REPLICATES = 10_000" in source
    assert "BOOTSTRAP_SEED = 20260801" in source
    assert "overall_validation_pass" in source
    assert "execution_failure_audit" in source
    assert "total_observed_t4_device_hours" in source
    assert 'FINAL / "execution_attempt_failures.csv"' in source
