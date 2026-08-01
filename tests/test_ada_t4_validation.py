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
