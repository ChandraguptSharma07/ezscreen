from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "benchmarks/bench-20260727-ace-ad4zn-probe/full_vina_t4_validation"


def load(name: str):
    path = CAMPAIGN / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_campaign_protocol_is_preregistered_and_vina_only():
    protocol = json.loads(
        (CAMPAIGN / "campaign_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["records"] == {"total": 1846, "actives": 49, "decoys": 1797}
    assert protocol["protocol"]["scoring"] == "vina"
    assert protocol["protocol"]["ranking"] == "raw UniDock Vina score ascending"
    assert protocol["published"] == {"roc_auc": 0.431, "bedroc20": 0.159}
    assert protocol["predefined_similarity_margins_pipeline_minus_published"] == {
        "roc_auc": -0.02,
        "bedroc20": -0.02,
    }
    assert protocol["bootstrap"] == {
        "method": "stratified nonparametric",
        "replicates": 10000,
        "seed": 20260801,
        "confidence": 0.95,
    }


def test_rendered_notebooks_have_preparation_hardware_and_provenance_gates():
    notebooks = sorted((CAMPAIGN / "notebooks").glob("ace_vina_t4_shard_*.ipynb"))
    assert len(notebooks) == 12
    bounds = json.loads((CAMPAIGN / "shard_plan.json").read_text(encoding="utf-8"))[
        "bounds"
    ]
    assert sum(bound["count"] for bound in bounds) == 1846
    assert {bound["count"] for bound in bounds} == {153, 154}
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        joined = "\n".join(sources)
        normalized = joined.replace(chr(34), "")
        hardware_index = next(
            index
            for index, source in enumerate(sources)
            if "HARDWARE_PROVENANCE" in source
        )
        install_index = next(
            index for index, source in enumerate(sources) if "git clone" in source
        )
        assert hardware_index < install_index
        assert "len(_gpu_rows) != 2" in joined
        assert "any(T4 not in model" in normalized
        assert "for scoring in (vina,):" in normalized
        assert "for scoring in (vina, vinardo):" not in normalized
        assert "source_build_manifest.json" in joined
        assert "hardware_manifest.json" in joined
        assert "input_manifest.json" in joined
        assert "prepared_identity_check.json" in joined
        assert "ebde3b4a22c8b2c7b26e523cc15109aab815be21" in joined
        assert "--exhaustiveness, str(EXHAUSTIVENESS)" in normalized
        for cell_index, source in enumerate(sources):
            if source.strip():
                compile(source, f"{path.name}:{cell_index}", "exec")


def test_launcher_requires_t4_and_two_shards_per_account():
    launcher = load("launch_full_vina_t4")
    assert launcher.ACCELERATOR == "nvidiaTeslaT4"
    source = (CAMPAIGN / "launch_full_vina_t4.py").read_text(encoding="utf-8")
    normalized = source.replace(chr(34), "")
    assert "required_gpu_model: Tesla T4" in normalized
    assert "required_gpu_count: 2" in normalized
    assert "for shard in (account_index * 2, account_index * 2 + 1)" in source


def test_finalizer_has_frozen_gates_and_rejects_protocol_drift():
    finalizer = load("finalize_full_vina_t4")
    assert finalizer.BOOTSTRAP_REPLICATES == 10_000
    assert finalizer.BOOTSTRAP_SEED == 20260801
    assert finalizer.PUBLISHED == {"roc_auc": 0.431, "bedroc20": 0.159}
    assert finalizer.MARGINS == {"roc_auc": -0.02, "bedroc20": -0.02}
    valid = [
        "unidock",
        "--receptor",
        "r.pdbqt",
        "--center_x",
        "40.7317",
        "--center_y",
        "32.8822",
        "--center_z",
        "47.3635",
        "--size_x",
        "36.8194",
        "--size_y",
        "41.835",
        "--size_z",
        "43.7048",
        "--spacing",
        "0.375",
        "--scoring",
        "vina",
        "--exhaustiveness",
        "32",
    ]
    finalizer.validate_command(valid)
    invalid = list(valid)
    invalid[invalid.index("32")] = "8"
    with pytest.raises(RuntimeError, match="--exhaustiveness"):
        finalizer.validate_command(invalid)


def test_quota_allows_exactly_two_gpu_kernels_per_account():
    quota = json.loads((CAMPAIGN / "quota_budget.json").read_text(encoding="utf-8"))
    assert quota["account_count"] == 6
    assert quota["maximum_concurrent_gpu_notebooks_per_account"] == 2
    assert quota["allocations_gpu_hours"] == {"t4_validation": 24.0}
