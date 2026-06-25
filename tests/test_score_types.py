from __future__ import annotations

import csv
import json

from ezscreen.results import score_types
from ezscreen.results.merger import merge_shard_results


def test_default_score_type_helpers():
    assert score_types.unit("vina_kcal_mol") == "kcal/mol"
    assert "kcal/mol" in score_types.label("vina_kcal_mol")
    assert "negative" in score_types.describe("vina_kcal_mol")


def test_unknown_score_type_falls_back_to_default():
    assert score_types.label("does_not_exist") == score_types.label("vina_kcal_mol")


def test_read_score_type_defaults_when_sidecar_absent(tmp_path):
    assert score_types.read_score_type(tmp_path) == "vina_kcal_mol"


def test_read_score_type_reads_sidecar(tmp_path):
    (tmp_path / "results_meta.json").write_text(json.dumps({"score_type": "cnn_affinity"}))
    assert score_types.read_score_type(tmp_path) == "cnn_affinity"


def test_merger_writes_score_type_sidecar(tmp_path):
    shard = tmp_path / "shard0"
    shard.mkdir()
    with (shard / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ligand", "score"])
        w.writeheader()
        w.writerow({"ligand": "lig1", "score": "-9.5"})
        w.writerow({"ligand": "lig2", "score": "-7.0"})

    out = tmp_path / "output"
    result = merge_shard_results([shard], out)

    assert result["score_type"] == "vina_kcal_mol"
    meta = json.loads((out / "results_meta.json").read_text())
    assert meta["score_type"] == "vina_kcal_mol"
    assert score_types.read_score_type(out) == "vina_kcal_mol"
