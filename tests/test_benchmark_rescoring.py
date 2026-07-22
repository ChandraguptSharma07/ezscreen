from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from ezscreen.benchmark.rescoring import compare_rescore_columns, join_rescore_scores
from ezscreen.benchmark.runner import run_benchmark, score_direction_for_col

runner = CliRunner()


def _write_scores(path: Path) -> None:
    rows = [
        {"ligand": "shard_000_lig_000", "name": "ACTIVE001", "smiles": "CCO", "score": "-7.0", "prolif_score": "-8.5", "CNNscore": "0.91", "CNNaffinity": "8.1"},
        {"ligand": "shard_000_lig_001", "name": "DECOY001", "smiles": "CCC", "score": "-9.0", "prolif_score": "-6.0", "CNNscore": "0.20", "CNNaffinity": "4.2"},
        {"ligand": "shard_001_lig_002", "name": "ACTIVE002", "smiles": "CCN", "score": "-6.5", "prolif_score": "-7.8", "CNNscore": "0.86", "CNNaffinity": "7.7"},
        {"ligand": "shard_001_lig_003", "name": "DECOY002", "smiles": "CCCC", "score": "-8.0", "prolif_score": "-5.0", "CNNscore": "0.30", "CNNaffinity": "5.1"},
        {"ligand": "shard_002_lig_004", "name": "DECOY003", "smiles": "CCCl", "score": "-5.0", "prolif_score": "-4.0", "CNNscore": "0.10", "CNNaffinity": "3.1"},
        {"ligand": "shard_002_lig_005", "name": "DECOY004", "smiles": "CCBr", "score": "-4.5", "prolif_score": "-4.2", "CNNscore": "0.15", "CNNaffinity": "3.4"},
        {"ligand": "shard_003_lig_006", "name": "DECOY005", "smiles": "CCF", "score": "-4.0", "prolif_score": "-3.8", "CNNscore": "0.12", "CNNaffinity": "3.2"},
        {"ligand": "shard_003_lig_007", "name": "DECOY006", "smiles": "COC", "score": "-3.5", "prolif_score": "-3.7", "CNNscore": "0.11", "CNNaffinity": "3.0"},
        {"ligand": "shard_004_lig_008", "name": "DECOY007", "smiles": "CCS", "score": "-3.0", "prolif_score": "-3.6", "CNNscore": "0.13", "CNNaffinity": "3.3"},
        {"ligand": "shard_004_lig_009", "name": "DECOY008", "smiles": "CC=O", "score": "-2.5", "prolif_score": "-3.5", "CNNscore": "0.14", "CNNaffinity": "3.5"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_run_benchmark_ranks_cnn_affinity_higher_is_better(tmp_path: Path) -> None:
    actives = tmp_path / "actives.ism"
    actives.write_text("CCO ACTIVE001\nCCN ACTIVE002\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    _write_scores(scores)

    raw = run_benchmark(actives, scores, score_col="score")
    cnn = run_benchmark(actives, scores, score_col="CNNaffinity")

    assert raw.label_diagnostics["first_active_rank"] == 3
    assert cnn.label_diagnostics["score_direction"] == "higher"
    assert cnn.label_diagnostics["first_active_rank"] == 1
    assert cnn.auc_roc == 1.0



def test_rfscore_family_columns_rank_higher_is_better() -> None:
    assert score_direction_for_col("oddt_rfscore_v1") == "higher"
    assert score_direction_for_col("oddt_rfscore_v2") == "higher"
    assert score_direction_for_col("oddt_rfscore_v3") == "higher"
    assert score_direction_for_col("RFScoreVS_v2") == "higher"

def test_compare_rescore_columns_writes_metrics_top20_and_controls(tmp_path: Path) -> None:
    actives = tmp_path / "actives.ism"
    actives.write_text("CCO ACTIVE001\nCCN ACTIVE002\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    _write_scores(scores)

    result = compare_rescore_columns(
        actives,
        scores,
        tmp_path / "comparison",
        score_cols=["score", "prolif_score", "CNNscore", "CNNaffinity"],
    )

    metrics_text = result["metrics_table"].read_text(encoding="utf-8")
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    top20 = list(csv.DictReader((tmp_path / "comparison" / "top20_CNNaffinity.csv").open(newline="", encoding="utf-8")))

    assert "score_col,score_direction,ef1,ef5,auc_roc" in metrics_text
    assert "CNNaffinity,higher" in metrics_text
    assert summary["property_only_controls"]["property_only_balanced_component"]["auc_roc"] is not None
    assert summary["columns"]["CNNaffinity"]["metrics"]["active_numeric_missing_entries"] == 0
    assert summary["columns"]["CNNaffinity"]["top_windows"]["top_20"]["shard_counts"]
    assert top20[0]["label"] == "active"
    assert top20[0]["name"] == "ACTIVE001"



def test_join_rescore_scores_adds_backend_columns(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    backend = tmp_path / "rfscore_scores.csv"
    backend.write_text(
        "ligand,rfscore_vscore\n"
        "shard_000_lig_000,7.2\n"
        "shard_000_lig_001,3.1\n",
        encoding="utf-8",
    )

    result = join_rescore_scores(scores, backend, score_cols=["rfscore_vscore"])
    rows = list(csv.DictReader(scores.open(newline="", encoding="utf-8")))

    assert result["matched_rows"] == 2
    assert rows[0]["rfscore_vscore"] == "7.2"
    assert rows[2]["rfscore_vscore"] == ""
def test_benchmark_cli_compare_rescores(tmp_path: Path) -> None:
    from ezscreen.cli import app

    actives = tmp_path / "actives.ism"
    actives.write_text("CCO ACTIVE001\nCCN ACTIVE002\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    _write_scores(scores)

    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "compare-rescores",
            str(scores),
            "--actives",
            str(actives),
            "--output-dir",
            str(output),
            "--score-cols",
            "score,prolif_score,CNNscore,CNNaffinity",
        ],
    )

    assert result.exit_code == 0
    assert (output / "rescore_metrics_table.csv").exists()
    assert (output / "rescore_comparison_summary.json").exists()
