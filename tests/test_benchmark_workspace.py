from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ezscreen.benchmark.runner import pose_retention_diagnostics, run_benchmark
from ezscreen.benchmark.workspace import (
    benchmark_root,
    create_campaign_scaffold,
    create_cell_scaffold,
    templates_dir,
)
from ezscreen.results.report_html import write_benchmark_report

runner = CliRunner()


def test_benchmark_templates_exist():
    root = benchmark_root()
    assert root.name == "benchmarks"
    tdir = templates_dir()
    assert (tdir / "campaign_manifest.template.json").exists()
    assert (tdir / "targets.template.json").exists()
    assert (tdir / "cell_manifest.template.json").exists()
    assert (tdir / "provenance.template.json").exists()
    assert (tdir / "failure_summary.template.json").exists()
    assert (tdir / "notes.template.md").exists()


def test_create_campaign_scaffold_copies_manifest_and_targets(tmp_path: Path):
    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)

    assert campaign == tmp_path / "bench-20260714-core-panel"
    assert (campaign / "campaign_manifest.json").exists()
    assert (campaign / "inputs" / "targets.json").exists()
    assert (campaign / "inputs" / "actives").is_dir()
    assert (campaign / "inputs" / "decoys").is_dir()

    manifest = json.loads(
        (campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    targets = json.loads(
        (campaign / "inputs" / "targets.json").read_text(encoding="utf-8")
    )

    assert manifest["benchmark_id"] == "bench-YYYYMMDD-name"
    assert isinstance(targets, list)
    assert targets[0]["target_id"] == "hiv1_protease"


def test_create_cell_scaffold_builds_required_contract_dirs_and_files(tmp_path: Path):
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )

    assert (
        cell
        == tmp_path
        / "bench-20260714-core-panel"
        / "hiv1_protease"
        / "crystal_holo"
        / "run-001"
    )
    for name in ("inputs", "raw_outputs", "merged_outputs", "logs", "reports"):
        assert (cell / name).is_dir(), name
    for name in (
        "cell_manifest.json",
        "provenance.json",
        "failure_summary.json",
        "metrics_summary.json",
        "notes.md",
    ):
        assert (cell / name).exists(), name


def test_create_cell_scaffold_is_idempotent_for_existing_notes(tmp_path: Path):
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    notes = cell / "notes.md"
    notes.write_text("custom note\n", encoding="utf-8")

    create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )

    assert notes.read_text(encoding="utf-8") == "custom note\n"


def test_archived_benchmark_cell_can_regenerate_report(tmp_path: Path):
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    actives = cell / "inputs" / "actives.smi"
    actives.write_text("CCO ethanol\n", encoding="utf-8")

    scores = cell / "merged_outputs" / "scores.csv"
    scores.write_text(
        "ligand,smiles,docking_score\n" "lig_0001,CCO,-8.0\n" "lig_0002,CCC,-6.0\n",
        encoding="utf-8",
    )

    result = run_benchmark(actives, scores)
    report = write_benchmark_report(result, cell / "reports" / "benchmark_report.html")

    html = report.read_text(encoding="utf-8")
    assert report.exists()
    assert "AUC-ROC" in html
    assert "EF 1%" in html
    assert "EF 5%" in html


def test_benchmark_cli_init_creates_campaign_scaffold(tmp_path: Path):
    from ezscreen.cli import app

    result = runner.invoke(
        app,
        [
            "benchmark",
            "init",
            "bench-20260714-core-panel",
            "--root",
            str(tmp_path),
        ],
    )

    campaign = tmp_path / "bench-20260714-core-panel"
    assert result.exit_code == 0
    assert campaign.exists()
    assert (campaign / "campaign_manifest.json").exists()
    assert (campaign / "inputs" / "targets.json").exists()


def test_benchmark_cli_init_creates_cell_scaffold(tmp_path: Path):
    from ezscreen.cli import app

    result = runner.invoke(
        app,
        [
            "benchmark",
            "init",
            "bench-20260714-core-panel",
            "--target",
            "hiv1_protease",
            "--regime",
            "crystal_holo",
            "--run-label",
            "run-001",
            "--root",
            str(tmp_path),
        ],
    )

    cell = (
        tmp_path
        / "bench-20260714-core-panel"
        / "hiv1_protease"
        / "crystal_holo"
        / "run-001"
    )
    assert result.exit_code == 0
    assert cell.exists()
    assert (cell / "cell_manifest.json").exists()
    assert (cell / "provenance.json").exists()


def test_benchmark_cli_init_rejects_partial_cell_args(tmp_path: Path):
    from ezscreen.cli import app

    result = runner.invoke(
        app,
        [
            "benchmark",
            "init",
            "bench-20260714-core-panel",
            "--target",
            "hiv1_protease",
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "must be provided together" in result.output


def test_benchmark_cli_run_cell_writes_metrics_and_report(tmp_path: Path):
    from ezscreen.cli import app

    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "inputs" / "actives.smi").write_text("CCO ethanol\n", encoding="utf-8")
    (cell / "merged_outputs" / "scores.csv").write_text(
        "ligand,name,smiles,docking_score\n"
        "lig_0001,ethanol,CCO,-8.0\n"
        "lig_0002,decoy,CCC,-6.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "run-cell", str(cell)])

    metrics = json.loads((cell / "metrics_summary.json").read_text(encoding="utf-8"))
    failure = json.loads((cell / "failure_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((cell / "provenance.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metrics["status"] == "complete"
    assert metrics["ef1"] is not None
    assert metrics["label_diagnostics"]["matched_by_identifier"] == 1
    assert metrics["label_diagnostics"]["first_active_rank"] == 1
    assert metrics["label_diagnostics"]["unmatched_active_entries"] == 0
    assert "ranking_diagnostics" in metrics
    assert "alternative_rankings" in metrics["ranking_diagnostics"]
    assert metrics["report_path"].endswith("benchmark_report.html")
    assert metrics["provenance_path"].endswith("provenance.json")
    assert failure["status"] == "complete"
    assert provenance["local_python_version"]
    assert provenance["local_platform"]
    assert provenance["timestamps"]["benchmark_run_completed_at"]
    report_html = (cell / "reports" / "benchmark_report.html").read_text(encoding="utf-8")
    assert "Ranking Diagnostics" in report_html
    assert (cell / "reports" / "benchmark_report.html").exists()



def test_benchmark_cli_correct_scores_and_run_cell_with_score_col(tmp_path: Path):
    from ezscreen.cli import app

    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "cah2",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "inputs" / "actives.smi").write_text("CCO ACTIVE001\n", encoding="utf-8")
    scores = cell / "merged_outputs" / "scores.csv"
    scores.write_text(
        "ligand,name,smiles,docking_score\n"
        "lig_0001,DECOY_CHARGED,O=P([O-])([O-])O,-12.0\n"
        "lig_0002,ACTIVE001,CCO,-8.0\n"
        "lig_0003,DECOY_NEUTRAL,CCCC,-6.0\n",
        encoding="utf-8",
    )

    correct = runner.invoke(app, ["benchmark", "correct-scores", str(scores)])
    assert correct.exit_code == 0
    assert "bias_corrected_score" in scores.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "benchmark",
            "run-cell",
            str(cell),
            "--score-col",
            "bias_corrected_score",
        ],
    )

    metrics = json.loads((cell / "metrics_summary.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metrics["score_col"] == "bias_corrected_score"
    assert metrics["label_diagnostics"]["first_active_rank"] == 1
    assert metrics["ef1"] > 0

    balanced = runner.invoke(
        app,
        [
            "benchmark",
            "correct-scores",
            str(scores),
            "--output-col",
            "balanced_corrected_score",
        ],
    )
    assert balanced.exit_code == 0
    assert "balanced_corrected_score" in scores.read_text(encoding="utf-8")
    assert "25.0 * abs_formal_charge" in balanced.output


def test_benchmark_cli_run_cell_records_failure_summary(tmp_path: Path):
    from ezscreen.cli import app

    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "inputs" / "actives.smi").write_text("CCO ethanol\n", encoding="utf-8")

    result = runner.invoke(app, ["benchmark", "run-cell", str(cell)])

    failure = json.loads((cell / "failure_summary.json").read_text(encoding="utf-8"))
    assert result.exit_code == 1
    assert failure["status"] == "failed"
    assert failure["failure_stage"] == "benchmark_run"
    assert "scores" in failure["failure_reason"].lower()


def _write_campaign_inputs(campaign: Path, target_id: str = "hiv1_protease") -> None:
    (campaign / "inputs" / "actives" / f"{target_id}.smi").write_text(
        "CCO ethanol\n", encoding="utf-8"
    )
    (campaign / "inputs" / "decoys" / f"{target_id}_decoys.smi").write_text(
        "CCC decoy\n", encoding="utf-8"
    )


def test_benchmark_cli_validate_campaign_checks_target_registry(tmp_path: Path):
    from ezscreen.cli import app

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)

    result = runner.invoke(app, ["benchmark", "validate-campaign", str(campaign)])

    assert result.exit_code == 0
    assert "Benchmark campaign validated" in result.output
    assert "targets -> 1" in result.output


def test_benchmark_cli_validate_campaign_warns_on_identifierless_actives(tmp_path: Path):
    from ezscreen.cli import app

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    (campaign / "inputs" / "actives" / "hiv1_protease.smi").write_text(
        "CCO\n", encoding="utf-8"
    )
    (campaign / "inputs" / "decoys" / "hiv1_protease_decoys.smi").write_text(
        "CCC decoy\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["benchmark", "validate-campaign", str(campaign)])

    assert result.exit_code == 0
    assert "active file has no identifiers" in result.output


def test_benchmark_cli_validate_campaign_fails_on_missing_target_inputs(tmp_path: Path):
    from ezscreen.cli import app

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)

    result = runner.invoke(app, ["benchmark", "validate-campaign", str(campaign)])

    assert result.exit_code == 1
    assert "missing referenced file" in result.output.lower()


def test_benchmark_cli_summarize_writes_campaign_tables(tmp_path: Path):
    from ezscreen.cli import app

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)

    first = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    second = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-002",
        root=tmp_path,
    )
    for cell, better_score in ((first, -8.0), (second, -7.5)):
        (cell / "inputs" / "actives.smi").write_text("CCO ethanol\n", encoding="utf-8")
        (cell / "merged_outputs" / "scores.csv").write_text(
            "ligand,smiles,docking_score\n"
            f"lig_0001,CCO,{better_score}\n"
            "lig_0002,CCC,-6.0\n",
            encoding="utf-8",
        )
        run_result = runner.invoke(app, ["benchmark", "run-cell", str(cell)])
        assert run_result.exit_code == 0
        metrics = json.loads((cell / "metrics_summary.json").read_text(encoding="utf-8"))
        report_html = (cell / "reports" / "benchmark_report.html").read_text(encoding="utf-8")
        assert metrics["credibility_gate"]["status"] == "fail"
        assert "Credibility Gate: fail" in report_html

    result = runner.invoke(app, ["benchmark", "summarize", str(campaign)])

    summary = campaign / "summary"
    metrics_csv = (summary / "metrics_table.csv").read_text(encoding="utf-8")
    variance_csv = (summary / "variance_table.csv").read_text(encoding="utf-8")
    failure_csv = (summary / "failure_table.csv").read_text(encoding="utf-8")
    assert result.exit_code == 0
    assert (summary / "metrics_table.csv").exists()
    assert (summary / "failure_table.csv").exists()
    assert (summary / "throughput_table.csv").exists()
    assert (summary / "variance_table.csv").exists()
    assert "first_active_rank" in metrics_csv
    assert "matched_by_identifier" in metrics_csv
    assert "ranking_flags" in metrics_csv
    assert "charge_size_adjusted_ef5" in metrics_csv
    assert "score_col" in metrics_csv
    assert "credibility_gate_status" in metrics_csv
    assert "credibility_gate_failed" in metrics_csv
    assert "pose_retention" in metrics_csv
    assert "external_baseline_status" in metrics_csv
    assert "external_baseline_ef1_floor" in metrics_csv
    assert "bias_corrected_ef5" in metrics_csv
    assert "balanced_corrected_ef5" in metrics_csv
    assert "run-001" in metrics_csv
    assert "run-002" in metrics_csv
    assert ",2," in variance_csv or variance_csv.count("\n") >= 2
    assert "complete" in failure_csv


def test_benchmark_cli_audit_campaign_reports_publishability_gaps(tmp_path: Path):
    from ezscreen.cli import app

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "inputs" / "actives.smi").write_text("CCO ethanol\n", encoding="utf-8")
    (cell / "merged_outputs" / "scores.csv").write_text(
        "ligand,smiles,docking_score\n" "lig_0001,CCO,-8.0\n" "lig_0002,CCC,-6.0\n",
        encoding="utf-8",
    )
    assert runner.invoke(app, ["benchmark", "run-cell", str(cell)]).exit_code == 0

    result = runner.invoke(app, ["benchmark", "audit-campaign", str(campaign)])
    strict_result = runner.invoke(
        app, ["benchmark", "audit-campaign", str(campaign), "--strict"]
    )

    assert result.exit_code == 0
    assert "not_ready" in result.output
    assert "target panel too small" in result.output
    assert strict_result.exit_code == 1




def test_benchmark_quality_audit_warns_when_prep_protocol_audit_missing(tmp_path: Path):
    from ezscreen.benchmark.summary import audit_campaign_quality

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "metrics_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "ef1": 1.0,
                "ef5": 1.0,
                "auc_roc": 0.75,
                "n_actives": 1,
                "n_decoys": 1,
                "total_screened": 2,
                "report_path": str(cell / "reports" / "benchmark_report.html"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "provenance.json").write_text(
        json.dumps(
            {
                "ezscreen_commit": "abc123",
                "local_python_version": "3.11.0",
                "local_platform": "test-platform",
                "timestamps": {"benchmark_run_completed_at": "2026-07-14T00:00:00+00:00"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "reports" / "benchmark_report.html").write_text("<html></html>\n", encoding="utf-8")

    audit = audit_campaign_quality(
        campaign,
        min_targets=1,
        min_target_classes=1,
        min_complete_cells=1,
        min_repeats=1,
        min_comparators=1,
    )

    assert any("receptor-prep protocol audit missing" in warning for warning in audit["warnings"])


def test_benchmark_summary_table_includes_prep_protocol_risk(tmp_path: Path):
    from ezscreen.benchmark.summary import summarize_campaign

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "metrics_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "ef1": 1.0,
                "ef5": 1.0,
                "auc_roc": 0.75,
                "n_actives": 1,
                "n_decoys": 1,
                "total_screened": 2,
                "ranking_diagnostics": {
                    "flags": ["top_1_percent_enriched_for_formally_charged_ligands", "best_scoring_ligand_is_decoy"],
                    "top_windows": {
                        "top_1_percent": {"charged_fraction": 1.0, "decoy_count": 10},
                        "top_5_percent": {"charged_fraction": 0.8, "decoy_count": 50},
                    },
                    "alternative_rankings": {
                        "charge_size_penalty_score_plus_4_abs_charge_plus_0_3_heavy_atoms": {
                            "ef1": 3.25,
                            "ef5": 1.95,
                            "auc_roc": 0.5846,
                            "first_active_rank": 15,
                        },
                        "bias_corrected_score_plus_8_abs_charge_plus_0_1_heavy_atoms_minus_0_03_tpsa": {
                            "ef1": 9.77,
                            "ef5": 3.70,
                            "auc_roc": 0.6874,
                            "first_active_rank": 3,
                        },
                        "balanced_corrected_score_plus_25_abs_charge_minus_0_8_heavy_atoms_minus_0_03_tpsa_minus_0_4_rotatable_bonds": {
                            "ef1": 5.69,
                            "ef5": 2.60,
                            "auc_roc": 0.6755,
                            "first_active_rank": 1,
                        },
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "provenance.json").write_text(
        json.dumps(
            {
                "receptor_prep": {
                    "prep_protocol_audit": {
                        "risk_level": "high",
                        "detected": {
                            "retained_metal_elements": [],
                            "removed_metal_elements": ["ZN"],
                            "retained_cofactors": [],
                            "removed_cofactors": ["HEM"],
                        },
                        "benchmark_ablation_suggestions": [
                            "ad4zn_scoring",
                            "metal_keep_remove",
                        ],
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summarize_campaign(campaign)

    metrics_csv = (campaign / "summary" / "metrics_table.csv").read_text(encoding="utf-8")
    assert "prep_protocol_risk" in metrics_csv
    assert "prep_removed_metals" in metrics_csv
    assert "prep_removed_cofactors" in metrics_csv
    assert "high" in metrics_csv
    assert "ZN" in metrics_csv
    assert "HEM" in metrics_csv
    assert "ad4zn_scoring,metal_keep_remove" in metrics_csv
    assert "ranking_flags" in metrics_csv
    assert "top_1_percent_enriched_for_formally_charged_ligands,best_scoring_ligand_is_decoy" in metrics_csv
    assert "charge_size_adjusted_ef5" in metrics_csv
    assert "external_baseline_status" in metrics_csv
    assert "external_baseline_failed_metrics" in metrics_csv
    assert "bias_corrected_ef5" in metrics_csv
    assert "balanced_corrected_ef5" in metrics_csv
    assert "9.77" in metrics_csv
    assert "5.69" in metrics_csv
    assert "fail" in metrics_csv
    assert "EF1 1 < 4" in metrics_csv
    assert "1.95" in metrics_csv


def test_benchmark_quality_audit_surfaces_external_baseline_failure(tmp_path: Path):
    from ezscreen.benchmark.summary import audit_campaign_quality

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "metrics_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "ef1": 1.0,
                "ef5": 1.0,
                "auc_roc": 0.55,
                "n_actives": 1,
                "n_decoys": 1,
                "total_screened": 2,
                "report_path": str(cell / "reports" / "benchmark_report.html"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "provenance.json").write_text(
        json.dumps(
            {
                "ezscreen_commit": "abc123",
                "local_python_version": "3.11.0",
                "local_platform": "test-platform",
                "timestamps": {"benchmark_run_completed_at": "2026-07-14T00:00:00+00:00"},
                "receptor_prep": {"prep_protocol_audit": {"risk_level": "low"}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "reports" / "benchmark_report.html").write_text("<html></html>\n", encoding="utf-8")

    audit = audit_campaign_quality(
        campaign,
        min_targets=1,
        min_target_classes=1,
        min_complete_cells=1,
        min_repeats=1,
        min_comparators=1,
    )

    assert any("below external benchmark floor" in warning for warning in audit["warnings"])
    assert any("EF1 1 < 4" in warning for warning in audit["warnings"])


def test_benchmark_quality_audit_surfaces_prep_protocol_risk(tmp_path: Path):
    from ezscreen.benchmark.summary import audit_campaign_quality

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    _write_campaign_inputs(campaign)
    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "metrics_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "ef1": 1.0,
                "ef5": 1.0,
                "auc_roc": 0.75,
                "n_actives": 1,
                "n_decoys": 1,
                "total_screened": 2,
                "report_path": str(cell / "reports" / "benchmark_report.html"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "provenance.json").write_text(
        json.dumps(
            {
                "ezscreen_commit": "abc123",
                "local_python_version": "3.11.0",
                "local_platform": "test-platform",
                "timestamps": {"benchmark_run_completed_at": "2026-07-14T00:00:00+00:00"},
                "receptor_prep": {
                    "prep_protocol_audit": {
                        "risk_level": "high",
                        "benchmark_ablation_suggestions": ["ad4zn_scoring", "zn_coordination_qc"],
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (cell / "reports" / "benchmark_report.html").write_text("<html></html>\n", encoding="utf-8")

    audit = audit_campaign_quality(
        campaign,
        min_targets=1,
        min_target_classes=1,
        min_complete_cells=1,
        min_repeats=1,        min_comparators=1,
    )

    assert any("high receptor-prep protocol risk" in warning for warning in audit["warnings"])
    assert any("ad4zn_scoring" in warning for warning in audit["warnings"])


def test_benchmark_quality_audit_passes_publishable_pipeline_shape(tmp_path: Path):
    from ezscreen.benchmark.summary import audit_campaign_quality

    campaign = create_campaign_scaffold("bench-20260714-core-panel", root=tmp_path)
    manifest = json.loads(
        (campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    manifest["baselines"] = ["ezscreen_kaggle_unidock", "local_vina"]
    manifest["required_ablations"] = ["pocket_mode"]
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    targets = []
    for target_id, target_class in (
        ("hiv1_protease", "protease"),
        ("abl1", "kinase"),
        ("aa2ar", "gpcr"),
    ):
        actives = campaign / "inputs" / "actives" / f"{target_id}.smi"
        decoys = campaign / "inputs" / "decoys" / f"{target_id}_decoys.smi"
        actives.write_text("CCO active\n", encoding="utf-8")
        decoys.write_text("CCC decoy\n", encoding="utf-8")
        targets.append(
            {
                "target_id": target_id,
                "target_class": target_class,
                "primary_structure": {"kind": "pdb", "id": "1ABC", "path": ""},
                "actives_file": f"inputs/actives/{target_id}.smi",
                "decoys_file": f"inputs/decoys/{target_id}_decoys.smi",
            }
        )
    (campaign / "inputs" / "targets.json").write_text(
        json.dumps(targets, indent=2) + "\n", encoding="utf-8"
    )

    cells = [
        (
            "hiv1_protease",
            "run-001",
            "ezscreen_kaggle_unidock",
            "ezscreen_kaggle_unidock",
        ),
        (
            "hiv1_protease",
            "run-002",
            "ezscreen_kaggle_unidock",
            "ezscreen_kaggle_unidock",
        ),
        (
            "hiv1_protease",
            "run-003",
            "ezscreen_kaggle_unidock",
            "ezscreen_kaggle_unidock",
        ),
        ("hiv1_protease", "run-004", "local_vina", "local_vina"),
        ("abl1", "run-001", "ezscreen_kaggle_unidock", "ezscreen_kaggle_unidock"),
        ("aa2ar", "run-001", "ezscreen_kaggle_unidock", "ezscreen_kaggle_unidock"),
    ]
    for target_id, run_label, engine_path, baseline_name in cells:
        cell = create_cell_scaffold(
            "bench-20260714-core-panel",
            target_id,
            "crystal_holo",
            run_label,
            root=tmp_path,
        )
        cell_manifest = json.loads(
            (cell / "cell_manifest.json").read_text(encoding="utf-8")
        )
        cell_manifest.update(
            {
                "benchmark_id": "bench-20260714-core-panel",
                "target_id": target_id,
                "target_class": next(
                    t["target_class"] for t in targets if t["target_id"] == target_id
                ),
                "receptor_regime": "crystal_holo",
                "pocket_mode": "cocrystal",
                "engine_path": engine_path,
                "baseline_name": baseline_name,
                "ablation_flags": {"pocket_mode": True},
            }
        )
        (cell / "cell_manifest.json").write_text(
            json.dumps(cell_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (cell / "metrics_summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "ef1": 1.0,
                    "ef5": 1.0,
                    "auc_roc": 0.75,
                    "n_actives": 1,
                    "n_decoys": 1,
                    "total_screened": 2,
                    "report_path": str(cell / "reports" / "benchmark_report.html"),
                    "redock_rmsd": 1.2,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (cell / "provenance.json").write_text(
            json.dumps(
                {
                    "ezscreen_commit": "abc123",
                    "local_python_version": "3.11.0",
                    "local_platform": "test-platform",
                    "timestamps": {
                        "benchmark_run_completed_at": "2026-07-14T00:00:00+00:00"
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (cell / "reports" / "benchmark_report.html").write_text(
            "<html></html>\n", encoding="utf-8"
        )
        (cell / "logs" / "methods.txt").write_text("methods\n", encoding="utf-8")

    audit = audit_campaign_quality(campaign)

    assert audit["readiness"] == "publishable_pipeline_ready"
    assert audit["target_class_count"] == 3
    assert audit["complete_cell_count"] == 6
    assert audit["max_repeat_count"] == 3
    assert audit["max_comparator_count"] == 2
    assert audit["issues"] == []



def test_submit_kaggle_cell_archives_remote_outputs_without_local_docking(tmp_path: Path, monkeypatch):
    from ezscreen.commands.benchmark import submit_kaggle_cell

    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    receptor = cell / "inputs" / "receptor_fixed.pdbqt"
    receptor.write_text("RECEPTOR\n", encoding="utf-8")
    ligands = tmp_path / "ligands.ism"
    ligands.write_text("CCO active\nCCC decoy\n", encoding="utf-8")
    (cell / "inputs" / "actives.smi").write_text("CCO active\n", encoding="utf-8")

    def fake_shard_raw(input_path, output_dir, n_shards=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        shard = output_dir / "shard_000.smi"
        shard.write_text("CCO active\nCCC decoy\n", encoding="utf-8")
        return {"shard_paths": [shard], "report": {"total_input": 2}}

    captured = {}

    def fake_render_vina_notebook(**kwargs):
        captured.update(kwargs)
        return "{}"

    def fake_run_screening_job(**kwargs):
        out = tmp_path / "remote_output"
        out.mkdir()
        (out / "scores.csv").write_text(
            "ligand,smiles,docking_score\n"
            "lig_0001,CCO,-8.0\n"
            "lig_0002,CCC,-6.0\n",
            encoding="utf-8",
        )
        (out / "results_meta.json").write_text('{"engine":"unidock"}\n', encoding="utf-8")
        return {"status": "complete", "output_dir": out, "error_type": None}

    monkeypatch.setattr("ezscreen.prep.ligands.shard_raw", fake_shard_raw)
    monkeypatch.setattr("ezscreen.backends.kaggle.runner.render_vina_notebook", fake_render_vina_notebook)
    monkeypatch.setattr("ezscreen.backends.kaggle.runner.run_screening_job", fake_run_screening_job)

    result = submit_kaggle_cell(
        cell,
        receptor_pdbqt=receptor,
        ligands=ligands,
        box_center="1,2,3",
        box_size="20,20,20",
        run_id="ezs-bench-test",
        n_shards=1,
        kaggle_username="tester",
    )

    metrics = json.loads((cell / "metrics_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((cell / "provenance.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert captured["prep_on_kaggle"] is True
    assert captured["prep_cfg"]["enumerate_enabled"] is True
    assert captured["prep_cfg"]["enumerate_protonation"] is True
    assert captured["prep_cfg"]["enumerate_tautomers"] is True
    assert captured["prep_cfg"]["max_variants_per_ligand"] >= 2
    assert captured["box_center"] == [1.0, 2.0, 3.0]
    assert captured["receptor_filename"] == "receptor.pdbqt"
    assert (cell / "raw_outputs" / "scores.csv").exists()
    assert (cell / "merged_outputs" / "scores.csv").exists()
    assert (cell / "logs" / "benchmark_kaggle.ipynb").exists()
    assert metrics["status"] == "complete"
    assert provenance["kaggle_runtime_identifiers"]["run_id"] == "ezs-bench-test"
    assert provenance["kaggle_runtime_identifiers"]["prep_on_kaggle"] is True
    assert provenance["settings_snapshot"]["benchmark_prep_cfg"]["enumerate_enabled"] is True



def test_benchmark_cli_run_cell_fails_when_active_unscored(tmp_path: Path):
    from ezscreen.cli import app

    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "inputs" / "actives.smi").write_text(
        "CCO ACTIVE001\nCCC ACTIVE002\n", encoding="utf-8"
    )
    (cell / "merged_outputs" / "scores.csv").write_text(
        "ligand,name,smiles,docking_score\n"
        "lig_0001,ACTIVE001,CCO,-8.0\n"
        "lig_0002,DECOY001,CCCC,-6.0\n",
        encoding="utf-8",
    )
    (cell / "merged_outputs" / "unscored_reasons.csv").write_text(
        "ligand,name,smiles,reason,raw_score\n"
        "lig_0003,ACTIVE002,CCC,meeko_failed,\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "run-cell", str(cell)])

    failure = json.loads((cell / "failure_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((cell / "metrics_summary.json").read_text(encoding="utf-8"))
    assert result.exit_code == 1
    assert failure["status"] == "failed"
    assert failure["failure_stage"] == "benchmark_run"
    assert "active survival gate failed" in failure["failure_reason"].lower()
    assert "meeko_failed" in failure["failure_reason"]
    assert metrics["status"] == "failed"
    assert metrics["ef1"] is None
    assert metrics["unscored_diagnostics"]["active_unscored_entries"] == 1
    assert metrics["unscored_diagnostics"]["active_unscored_reason_counts"] == {"meeko_failed": 1}


def test_benchmark_cli_run_cell_fails_when_active_missing(tmp_path: Path):
    from ezscreen.cli import app

    cell = create_cell_scaffold(
        "bench-20260714-core-panel",
        "hiv1_protease",
        "crystal_holo",
        "run-001",
        root=tmp_path,
    )
    (cell / "inputs" / "actives.smi").write_text(
        "CCO ACTIVE001\nCCC ACTIVE002\n", encoding="utf-8"
    )
    (cell / "merged_outputs" / "scores.csv").write_text(
        "ligand,name,smiles,docking_score\n"
        "lig_0001,ACTIVE001,CCO,-8.0\n"
        "lig_0002,DECOY001,CCCC,-6.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["benchmark", "run-cell", str(cell)])

    failure = json.loads((cell / "failure_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((cell / "metrics_summary.json").read_text(encoding="utf-8"))
    assert result.exit_code == 1
    assert "active survival gate failed" in failure["failure_reason"].lower()
    assert "1 active entry missing" in failure["failure_reason"]
    assert metrics["status"] == "failed"
    assert metrics["unscored_diagnostics"]["active_missing_entries"] == 1



def test_run_benchmark_records_charge_size_ranking_diagnostics(tmp_path: Path):
    actives = tmp_path / "actives.ism"
    actives.write_text("CCO ACTIVE001\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "ligand,name,smiles,score\n"
        "lig_0001,DECOY_CHARGED,O=P([O-])([O-])O,-12.0\n"
        "lig_0002,ACTIVE001,CCO,-8.0\n"
        "lig_0003,DECOY_NEUTRAL,CCCC,-6.0\n",
        encoding="utf-8",
    )

    result = run_benchmark(actives, scores, score_col="score")
    diag = result.ranking_diagnostics

    assert diag["top_windows"]["top_1_percent"]["decoy_count"] == 1
    assert diag["top_decoy_examples"][0]["name"] == "DECOY_CHARGED"
    assert diag["top_decoy_examples"][0]["abs_formal_charge"] == 2
    assert "charge_size_penalty_score_plus_4_abs_charge_plus_0_3_heavy_atoms" in diag["alternative_rankings"]
    assert "bias_corrected_score_plus_8_abs_charge_plus_0_1_heavy_atoms_minus_0_03_tpsa" in diag["alternative_rankings"]
    assert "balanced_corrected_score_plus_25_abs_charge_minus_0_8_heavy_atoms_minus_0_03_tpsa_minus_0_4_rotatable_bonds" in diag["alternative_rankings"]


def test_run_benchmark_matches_active_identifiers_after_prep_changes_smiles(tmp_path: Path):
    actives = tmp_path / "actives.ism"
    actives.write_text("CCO ACTIVE001 CHEMBL1\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "ligand,name,smiles,score\n"
        "lig_0001,DECOY001,CCC,-8.0\n"
        "lig_0002,ACTIVE001,CC[O-],-9.0\n",
        encoding="utf-8",
    )

    result = run_benchmark(actives, scores, score_col="score")

    assert result.n_actives == 1
    assert result.total_screened == 2
    assert result.auc_roc == 1.0
    assert result.label_diagnostics["matched_by_identifier"] == 1
    assert result.label_diagnostics["matched_by_smiles_only"] == 0
    assert result.label_diagnostics["unmatched_active_identifiers"] == 1
    assert result.label_diagnostics["unmatched_active_entries"] == 0
    assert result.label_diagnostics["first_active_rank"] == 1

def test_pose_retention_diagnostics_flags_missing_top_score_floor_pose(tmp_path: Path):
    from rdkit import Chem

    actives = tmp_path / "actives.smi"
    actives.write_text("CCO A1\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "ligand,name,smiles,score,score_qc\n"
        "lig_d1,D1,CCN,-20,score_floor\n"
        "lig_a1,A1,CCO,-8,\n"
        "lig_d2,D2,CCC,-7,\n",
        encoding="utf-8",
    )
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("lig_id", "lig_a1")
    mol.SetProp("_Name", "lig_a1")
    writer = Chem.SDWriter(str(tmp_path / "poses.sdf"))
    writer.write(mol)
    writer.close()

    diag = pose_retention_diagnostics(actives, scores, score_col="score", top_n=2)

    assert diag["status"] == "complete"
    assert diag["top_window_missing_pose_count"] == 1
    assert diag["score_floor_rows"] == 1
    assert diag["score_floor_pose_count"] == 0
    assert diag["retained_active_pose_count"] == 1
    assert "top_window_missing_retained_poses" in diag["flags"]
    assert "score_floor_rows_missing_retained_poses" in diag["flags"]
    assert "no_retained_active_poses" not in diag["flags"]



def test_benchmark_matches_enumerated_variant_active_identifier(tmp_path: Path):
    from ezscreen.commands.benchmark import _active_unscored_diagnostics

    actives = tmp_path / "actives.smi"
    actives.write_text("CCO ACTIVE001\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "ligand,name,smiles,score\n"
        "lig_0001,ACTIVE001_v2,C[NH3+],-8.0\n"
        "lig_0002,DECOY001,CCC,-6.0\n",
        encoding="utf-8",
    )

    result = run_benchmark(actives, scores, score_col="score")
    diagnostics = _active_unscored_diagnostics(actives, scores)

    assert result.label_diagnostics["matched_by_identifier"] == 1
    assert result.label_diagnostics["matched_active_entries"] == 1
    assert result.label_diagnostics["unmatched_active_entries"] == 0
    assert diagnostics["active_missing_entries"] == 0
