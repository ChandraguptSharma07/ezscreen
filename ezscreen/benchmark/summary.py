from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

_REQUIRED_TARGET_FIELDS = (
    "target_id",
    "target_class",
    "primary_structure",
    "actives_file",
    "decoys_file",
)

_REQUIRED_MANIFEST_FIELDS = (
    "benchmark_id",
    "targets_file",
)

_PUBLISHABLE_MIN_TARGETS = 3
_PUBLISHABLE_MIN_TARGET_CLASSES = 3
_PUBLISHABLE_MIN_COMPLETE_CELLS = 3
_PUBLISHABLE_MIN_REPEATS = 3
_PUBLISHABLE_MIN_COMPARATORS = 2

_DEFAULT_EXTERNAL_BASELINE = {
    "auc_roc": 0.60,
    "ef1": 3.0,
    "ef5": 2.0,
    "source": "conservative raw docking floor from Vina/AutoDock-style DUD-E benchmarks",
}

_TARGET_EXTERNAL_BASELINES = {
    "abl1": {
        "auc_roc": 0.70,
        "ef1": 7.0,
        "ef5": 2.0,
        "source": "ABL1 target-specific floor anchored to classical DUD-E docking medians; stronger shape methods report higher EF1",
    },
    "cah2": {
        "auc_roc": 0.60,
        "ef1": 4.5,
        "ef5": 2.0,
        "source": "CAH2 target-specific floor from DUD-E virtual-screening baselines plus conservative EF5 floor",
    },
    "carbonic_anhydrase_2": {
        "auc_roc": 0.60,
        "ef1": 4.5,
        "ef5": 2.0,
        "source": "CAH2 target-specific floor from DUD-E virtual-screening baselines plus conservative EF5 floor",
    },
    "hiv1_protease": {
        "auc_roc": 0.70,
        "ef1": 4.0,
        "ef5": 2.0,
        "source": "HIV protease Vina-family benchmark floor",
    },
    "hivpr": {
        "auc_roc": 0.70,
        "ef1": 4.0,
        "ef5": 2.0,
        "source": "HIV protease Vina-family benchmark floor",
    },
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def _join_csv_list(value) -> str:
    if not isinstance(value, list):
        return ""
    return ",".join(str(item) for item in value if str(item))


def _iter_cell_dirs(campaign_dir: Path) -> list[Path]:
    return sorted(path.parent for path in campaign_dir.rglob("cell_manifest.json"))


def _identifier_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            count += max(0, len(line.split()) - 1)
    return count


def validate_campaign(campaign_dir: Path) -> dict:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest_path = campaign_dir / "campaign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Benchmark campaign manifest not found: {manifest_path}"
        )

    manifest = _load_json(manifest_path)
    missing_manifest = [
        field for field in _REQUIRED_MANIFEST_FIELDS if not manifest.get(field)
    ]
    if missing_manifest:
        raise ValueError(
            f"Campaign manifest missing required fields: {', '.join(missing_manifest)}"
        )

    targets_path = campaign_dir / str(manifest["targets_file"])
    if not targets_path.exists():
        raise FileNotFoundError(f"Benchmark targets file not found: {targets_path}")

    targets = _load_json(targets_path)
    if not isinstance(targets, list) or not targets:
        raise ValueError(
            f"Benchmark targets file must contain a non-empty list: {targets_path}"
        )

    target_ids: list[str] = []
    checked_files: list[str] = []
    warnings: list[str] = []
    for entry in targets:
        missing = [field for field in _REQUIRED_TARGET_FIELDS if not entry.get(field)]
        if missing:
            target_name = entry.get("target_id") or "<unknown>"
            raise ValueError(
                f"Target {target_name} missing required fields: {', '.join(missing)}"
            )
        primary = entry.get("primary_structure")
        if not isinstance(primary, dict) or not primary.get("kind"):
            raise ValueError(
                f"Target {entry['target_id']} has invalid primary_structure"
            )
        for key in ("actives_file", "decoys_file"):
            rel = Path(str(entry[key]))
            full = campaign_dir / rel
            if not full.exists():
                raise FileNotFoundError(
                    f"Target {entry['target_id']} missing referenced file: {full}"
                )
            checked_files.append(str(full))
            if key == "actives_file" and _identifier_count(full) == 0:
                warnings.append(
                    f"Target {entry['target_id']} active file has no identifiers after SMILES; "
                    "exact-SMILES labeling is fragile after ligand prep changes protonation or tautomer form."
                )
        target_ids.append(str(entry["target_id"]))

    return {
        "campaign_dir": str(campaign_dir),
        "manifest_path": str(manifest_path),
        "targets_path": str(targets_path),
        "target_count": len(targets),
        "target_ids": target_ids,
        "checked_file_count": len(checked_files),
        "warnings": warnings,
    }


def _cell_status(cell: Path) -> str:
    metrics = (
        _load_json(cell / "metrics_summary.json")
        if (cell / "metrics_summary.json").exists()
        else {}
    )
    failure = (
        _load_json(cell / "failure_summary.json")
        if (cell / "failure_summary.json").exists()
        else {}
    )
    return str(metrics.get("status") or failure.get("status") or "pending")


def _has_value(payload: dict, dotted_key: str) -> bool:
    value = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return value not in ("", None, [], {})


def _metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _external_baseline_for(target_id: str) -> dict:
    key = str(target_id or "").strip().lower().replace("-", "_")
    baseline = _TARGET_EXTERNAL_BASELINES.get(key, _DEFAULT_EXTERNAL_BASELINE)
    return dict(baseline)


def _cell_target_id(cell: Path, manifest: dict) -> str:
    target_id = str(manifest.get("target_id") or "").strip()
    if target_id:
        return target_id
    try:
        return cell.parent.parent.name
    except IndexError:
        return ""


def _evaluate_external_baseline(target_id: str, metrics: dict, status: str) -> dict:
    baseline = _external_baseline_for(target_id)
    result = {
        "external_baseline_auc_floor": baseline["auc_roc"],
        "external_baseline_ef1_floor": baseline["ef1"],
        "external_baseline_ef5_floor": baseline["ef5"],
        "external_baseline_source": baseline["source"],
        "external_baseline_status": "not_applicable",
        "external_baseline_failed_metrics": "",
    }
    if status != "complete":
        return result

    failed: list[str] = []
    missing: list[str] = []
    for key, label in (("auc_roc", "AUC-ROC"), ("ef1", "EF1"), ("ef5", "EF5")):
        value = _metric_value(metrics, key)
        if value is None:
            missing.append(label)
            continue
        floor = float(baseline[key])
        if value < floor:
            failed.append(f"{label} {value:g} < {floor:g}")

    if missing:
        result["external_baseline_status"] = "incomplete_metrics"
        result["external_baseline_failed_metrics"] = ",".join(missing)
    elif failed:
        result["external_baseline_status"] = "fail"
        result["external_baseline_failed_metrics"] = "; ".join(failed)
    else:
        result["external_baseline_status"] = "pass"
    return result


def audit_campaign_quality(
    campaign_dir: Path,
    *,
    min_targets: int = _PUBLISHABLE_MIN_TARGETS,
    min_target_classes: int = _PUBLISHABLE_MIN_TARGET_CLASSES,
    min_complete_cells: int = _PUBLISHABLE_MIN_COMPLETE_CELLS,
    min_repeats: int = _PUBLISHABLE_MIN_REPEATS,
    min_comparators: int = _PUBLISHABLE_MIN_COMPARATORS,
) -> dict:
    """Audit whether a campaign is shaped like publishable pipeline evidence.

    This deliberately does not require strong enrichment values. A pipeline paper can
    publish weak cells if it reports them honestly; what it cannot publish credibly is
    a narrow, non-comparable, non-reproducible evidence bundle.
    """
    campaign_dir = campaign_dir.expanduser().resolve()
    validation = validate_campaign(campaign_dir)
    manifest = _load_json(campaign_dir / "campaign_manifest.json")
    targets = _load_json(campaign_dir / str(manifest["targets_file"]))
    cells = _iter_cell_dirs(campaign_dir)

    issues: list[str] = []
    warnings: list[str] = []
    strengths: list[str] = []
    warnings.extend(str(item) for item in validation.get("warnings", []))

    target_classes = sorted(
        {
            str(target.get("target_class", ""))
            for target in targets
            if target.get("target_class")
        }
    )
    if len(targets) < min_targets:
        issues.append(
            f"target panel too small: {len(targets)} target(s), need at least {min_targets}"
        )
    else:
        strengths.append(f"target count meets gate: {len(targets)}")
    if len(target_classes) < min_target_classes:
        issues.append(
            f"target-class diversity too narrow: {len(target_classes)} class(es), need at least {min_target_classes}"
        )
    else:
        strengths.append(f"target-class diversity meets gate: {len(target_classes)}")

    baselines = [
        str(item) for item in manifest.get("baselines", []) if str(item).strip()
    ]
    real_baselines = [item for item in baselines if "placeholder" not in item.lower()]
    if len(set(real_baselines)) < min_comparators:
        issues.append(
            f"baseline definitions insufficient: {len(set(real_baselines))} real baseline(s), need at least {min_comparators}"
        )
    else:
        strengths.append(f"baseline definitions meet gate: {len(set(real_baselines))}")

    complete_cells: list[Path] = []
    complete_rows: list[dict] = []
    for cell in cells:
        manifest_row = _load_json(cell / "cell_manifest.json")
        metrics = (
            _load_json(cell / "metrics_summary.json")
            if (cell / "metrics_summary.json").exists()
            else {}
        )
        provenance = (
            _load_json(cell / "provenance.json")
            if (cell / "provenance.json").exists()
            else {}
        )
        status = _cell_status(cell)
        if status != "complete":
            continue
        complete_cells.append(cell)
        complete_rows.append(
            {
                "cell": cell,
                "manifest": manifest_row,
                "metrics": metrics,
                "provenance": provenance,
            }
        )

        required_metrics = (
            "ef1",
            "ef5",
            "auc_roc",
            "n_actives",
            "n_decoys",
            "total_screened",
        )
        if int(metrics.get("metric_contract_version") or 1) >= 2:
            required_metrics += (
                "ef20",
                "bedroc",
                "coverage",
                "first_active_rank",
            )
        for key in required_metrics:
            if metrics.get(key) in ("", None):
                issues.append(f"{cell}: complete cell missing metric `{key}`")
        for key in (
            "ezscreen_commit",
            "local_python_version",
            "local_platform",
            "timestamps.benchmark_run_completed_at",
        ):
            if not _has_value(provenance, key):
                issues.append(f"{cell}: complete cell missing provenance `{key}`")
        if not (
            cell / "reports" / "benchmark_report.html"
        ).exists() and not metrics.get("report_path"):
            issues.append(f"{cell}: complete cell missing benchmark HTML report")
        if not (cell / "logs" / "methods.txt").exists():
            warnings.append(f"{cell}: missing archived methods.txt")

        auc = _metric_value(metrics, "auc_roc")
        ef1 = _metric_value(metrics, "ef1")
        if auc is not None and auc < 0.6:
            warnings.append(
                f"{cell}: weak AUC-ROC ({auc}); publish as diagnostic, not validation"
            )
        if ef1 == 0:
            warnings.append(
                f"{cell}: EF1% is zero; audit pocket/prep/scoring before strengthening claims"
            )
        baseline_eval = _evaluate_external_baseline(
            _cell_target_id(cell, manifest_row), metrics, status
        )
        if baseline_eval["external_baseline_status"] in {"fail", "incomplete_metrics"}:
            warnings.append(
                f"{cell}: below external benchmark floor ({baseline_eval['external_baseline_failed_metrics']}); "
                f"source: {baseline_eval['external_baseline_source']}"
            )

        prep_audit = (
            provenance.get("receptor_prep", {}).get("prep_protocol_audit")
            if isinstance(provenance.get("receptor_prep"), dict)
            else None
        )
        if isinstance(prep_audit, dict):
            risk = str(prep_audit.get("risk_level") or "").strip().lower()
            if risk in {"medium", "high"}:
                suggestions = prep_audit.get("benchmark_ablation_suggestions") or []
                suggestion_text = ", ".join(str(item) for item in suggestions) if suggestions else "review target-aware prep"
                warnings.append(
                    f"{cell}: {risk} receptor-prep protocol risk; suggested ablation(s): {suggestion_text}"
                )
        else:
            warnings.append(
                f"{cell}: receptor-prep protocol audit missing; rerun or backfill prep provenance before trusting EF/AUC"
            )

    if len(complete_cells) < min_complete_cells:
        issues.append(
            f"completed evidence too small: {len(complete_cells)} complete cell(s), need at least {min_complete_cells}"
        )
    else:
        strengths.append(f"completed evidence meets gate: {len(complete_cells)}")

    repeat_groups: dict[tuple[str, str, str, str, str], int] = {}
    comparator_groups: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    covered_ablations: set[str] = set()
    cocrystal_cells = 0
    cocrystal_rmsd_cells = 0
    for row in complete_rows:
        manifest_row = row["manifest"]
        metrics = row["metrics"]
        key = (
            str(manifest_row.get("target_id", "")),
            str(manifest_row.get("receptor_regime", "")),
            str(manifest_row.get("pocket_mode", "")),
            str(manifest_row.get("engine_path", "")),
            str(manifest_row.get("baseline_name", "")),
        )
        repeat_groups[key] = repeat_groups.get(key, 0) + 1
        comparator_key = key[:3]
        comparator_groups.setdefault(comparator_key, set()).add((key[3], key[4]))
        flags = manifest_row.get("ablation_flags")
        if isinstance(flags, dict):
            covered_ablations.update(
                str(name) for name, enabled in flags.items() if enabled
            )
        if str(manifest_row.get("pocket_mode", "")).lower() == "cocrystal":
            cocrystal_cells += 1
            if metrics.get("redock_rmsd") not in ("", None):
                cocrystal_rmsd_cells += 1

    max_repeats = max(repeat_groups.values(), default=0)
    if max_repeats < min_repeats:
        issues.append(
            f"repeat evidence missing: max repeat count is {max_repeats}, need at least {min_repeats}"
        )
    else:
        strengths.append(f"repeat evidence meets gate: max repeat count {max_repeats}")

    max_comparators = max(
        (len(values) for values in comparator_groups.values()), default=0
    )
    if max_comparators < min_comparators:
        issues.append(
            f"baseline/comparator cells missing: max comparable methods per target/regime is {max_comparators}, need {min_comparators}"
        )
    else:
        strengths.append(
            f"comparator evidence meets gate: max comparable methods {max_comparators}"
        )

    required_ablations = [
        str(item)
        for item in manifest.get("required_ablations", [])
        if str(item).strip()
    ]
    missing_ablations = sorted(set(required_ablations) - covered_ablations)
    if missing_ablations:
        issues.append("ablation evidence missing for: " + ", ".join(missing_ablations))
    elif required_ablations:
        strengths.append("ablation coverage meets manifest gate")

    if cocrystal_cells and cocrystal_rmsd_cells < cocrystal_cells:
        warnings.append(
            f"redocking RMSD missing for {cocrystal_cells - cocrystal_rmsd_cells}/{cocrystal_cells} cocrystal cell(s)"
        )

    readiness = "publishable_pipeline_ready" if not issues else "not_ready"
    return {
        "campaign_dir": str(campaign_dir),
        "readiness": readiness,
        "target_count": len(targets),
        "target_class_count": len(target_classes),
        "target_classes": target_classes,
        "cell_count": len(cells),
        "complete_cell_count": len(complete_cells),
        "max_repeat_count": max_repeats,
        "max_comparator_count": max_comparators,
        "covered_ablations": sorted(covered_ablations),
        "missing_ablations": missing_ablations,
        "issues": issues,
        "warnings": warnings,
        "strengths": strengths,
        "validation": validation,
    }


def summarize_campaign(campaign_dir: Path) -> dict[str, Path | int]:
    campaign_dir = campaign_dir.expanduser().resolve()
    validate_campaign(campaign_dir)
    cells = _iter_cell_dirs(campaign_dir)
    if not cells:
        raise ValueError(f"No benchmark cells found under {campaign_dir}")

    metrics_rows: list[dict] = []
    failure_rows: list[dict] = []
    throughput_rows: list[dict] = []
    complete_metric_rows: list[dict] = []

    for cell in cells:
        manifest = _load_json(cell / "cell_manifest.json")
        metrics = (
            _load_json(cell / "metrics_summary.json")
            if (cell / "metrics_summary.json").exists()
            else {}
        )
        failure = (
            _load_json(cell / "failure_summary.json")
            if (cell / "failure_summary.json").exists()
            else {}
        )
        provenance = (
            _load_json(cell / "provenance.json")
            if (cell / "provenance.json").exists()
            else {}
        )
        prep_audit = provenance.get("receptor_prep", {}).get("prep_protocol_audit")
        if not isinstance(prep_audit, dict):
            prep_audit = {}
        prep_detected = prep_audit.get("detected")
        if not isinstance(prep_detected, dict):
            prep_detected = {}
        run_label = cell.name

        label_diagnostics = metrics.get("label_diagnostics")
        if not isinstance(label_diagnostics, dict):
            label_diagnostics = {}
        unscored_diagnostics = metrics.get("unscored_diagnostics")
        if not isinstance(unscored_diagnostics, dict):
            unscored_diagnostics = {}
        ranking_diagnostics = metrics.get("ranking_diagnostics")
        if not isinstance(ranking_diagnostics, dict):
            ranking_diagnostics = {}
        pose_diagnostics = metrics.get("pose_retention_diagnostics")
        if not isinstance(pose_diagnostics, dict):
            pose_diagnostics = {}
        ranking_flags = ranking_diagnostics.get("flags")
        if not isinstance(ranking_flags, list):
            ranking_flags = []
        top_windows = ranking_diagnostics.get("top_windows")
        if not isinstance(top_windows, dict):
            top_windows = {}
        top1_window = top_windows.get("top_1_percent")
        if not isinstance(top1_window, dict):
            top1_window = {}
        top5_window = top_windows.get("top_5_percent")
        if not isinstance(top5_window, dict):
            top5_window = {}
        credibility_gate = metrics.get("credibility_gate")
        if not isinstance(credibility_gate, dict):
            credibility_gate = ranking_diagnostics.get("credibility_gate")
        if not isinstance(credibility_gate, dict):
            credibility_gate = {}
        credibility_gate_items = credibility_gate.get("gates")
        if not isinstance(credibility_gate_items, dict):
            credibility_gate_items = {}
        failed_gate_names = [name for name, payload in credibility_gate_items.items() if isinstance(payload, dict) and payload.get("status") == "fail"]
        warned_gate_names = [name for name, payload in credibility_gate_items.items() if isinstance(payload, dict) and payload.get("status") == "warn"]
        alternative_rankings = ranking_diagnostics.get("alternative_rankings")
        if not isinstance(alternative_rankings, dict):
            alternative_rankings = {}
        charge_size_rank = alternative_rankings.get("charge_size_penalty_score_plus_4_abs_charge_plus_0_3_heavy_atoms")
        if not isinstance(charge_size_rank, dict):
            charge_size_rank = {}
        bias_corrected_rank = alternative_rankings.get("bias_corrected_score_plus_8_abs_charge_plus_0_1_heavy_atoms_minus_0_03_tpsa")
        if not isinstance(bias_corrected_rank, dict):
            bias_corrected_rank = {}
        balanced_corrected_rank = alternative_rankings.get("balanced_corrected_score_plus_25_abs_charge_minus_0_8_heavy_atoms_minus_0_03_tpsa_minus_0_4_rotatable_bonds")
        if not isinstance(balanced_corrected_rank, dict):
            balanced_corrected_rank = {}
        row_status = metrics.get("status") or failure.get("status", "pending")
        target_id = _cell_target_id(cell, manifest)
        baseline_eval = _evaluate_external_baseline(target_id, metrics, str(row_status))

        metric_row = {
            "benchmark_id": manifest.get("benchmark_id", ""),
            "target_id": manifest.get("target_id", "") or target_id,
            "target_class": manifest.get("target_class", ""),
            "receptor_regime": manifest.get("receptor_regime", ""),
            "pocket_mode": manifest.get("pocket_mode", ""),
            "engine_path": manifest.get("engine_path", ""),
            "baseline_name": manifest.get("baseline_name", ""),
            "run_label": run_label,
            "status": row_status,
            "ef1": metrics.get("ef1", ""),
            "ef5": metrics.get("ef5", ""),
            "ef20": metrics.get("ef20", ""),
            "auc_roc": metrics.get("auc_roc", ""),
            "bedroc": metrics.get("bedroc", ""),
            "coverage": metrics.get("coverage", ""),
            "score_col": metrics.get("score_col", ""),
            "external_baseline_status": baseline_eval["external_baseline_status"],
            "external_baseline_failed_metrics": baseline_eval["external_baseline_failed_metrics"],
            "external_baseline_auc_floor": baseline_eval["external_baseline_auc_floor"],
            "external_baseline_ef1_floor": baseline_eval["external_baseline_ef1_floor"],
            "external_baseline_ef5_floor": baseline_eval["external_baseline_ef5_floor"],
            "external_baseline_source": baseline_eval["external_baseline_source"],
            "credibility_gate_status": credibility_gate.get("status", ""),
            "credibility_gate_failed": _join_csv_list(failed_gate_names),
            "credibility_gate_warned": _join_csv_list(warned_gate_names),
            "redock_rmsd": metrics.get("redock_rmsd", ""),
            "n_actives": metrics.get("n_actives", ""),
            "n_decoys": metrics.get("n_decoys", ""),
            "total_screened": metrics.get("total_screened", ""),
            "first_active_rank": metrics.get(
                "first_active_rank", label_diagnostics.get("first_active_rank", "")
            ),
            "pose_retention_fraction": pose_diagnostics.get(
                "retained_pose_fraction", ""
            ),
            "counts_before_prep": label_diagnostics.get("counts_before_prep", ""),
            "counts_after_prep": label_diagnostics.get("counts_after_prep", ""),
            "counts_after_aggregation": label_diagnostics.get(
                "counts_after_aggregation", ""
            ),
            "counts_scored": label_diagnostics.get("counts_scored", ""),
            "counts_failed": label_diagnostics.get("counts_failed", ""),
            "counts_filtered": label_diagnostics.get("counts_filtered", ""),
            "counts_unmatched": label_diagnostics.get("counts_unmatched", ""),
            "matched_by_identifier": label_diagnostics.get("matched_by_identifier", ""),
            "matched_by_smiles_only": label_diagnostics.get("matched_by_smiles_only", ""),
            "unmatched_active_identifiers": label_diagnostics.get("unmatched_active_identifiers", ""),
            "matched_active_entries": label_diagnostics.get("matched_active_entries", ""),
            "unmatched_active_entries": label_diagnostics.get("unmatched_active_entries", ""),
            "top_1_percent_actives": label_diagnostics.get("top_1_percent_actives", ""),
            "top_5_percent_actives": label_diagnostics.get("top_5_percent_actives", ""),
            "active_unscored_entries": unscored_diagnostics.get("active_unscored_entries", ""),
            "active_missing_entries": unscored_diagnostics.get("active_missing_entries", ""),
            "active_unscored_reason_counts": json.dumps(unscored_diagnostics.get("active_unscored_reason_counts", {}), sort_keys=True) if unscored_diagnostics else "",
            "ranking_flags": _join_csv_list(ranking_flags),
            "top_1_percent_charged_fraction": top1_window.get("charged_fraction", ""),
            "top_5_percent_charged_fraction": top5_window.get("charged_fraction", ""),
            "top_1_percent_decoy_count": top1_window.get("decoy_count", ""),
            "top_5_percent_decoy_count": top5_window.get("decoy_count", ""),
            "charge_size_adjusted_ef1": charge_size_rank.get("ef1", ""),
            "charge_size_adjusted_ef5": charge_size_rank.get("ef5", ""),
            "charge_size_adjusted_auc_roc": charge_size_rank.get("auc_roc", ""),
            "charge_size_adjusted_first_active_rank": charge_size_rank.get("first_active_rank", ""),
            "bias_corrected_ef1": bias_corrected_rank.get("ef1", ""),
            "bias_corrected_ef5": bias_corrected_rank.get("ef5", ""),
            "bias_corrected_auc_roc": bias_corrected_rank.get("auc_roc", ""),
            "bias_corrected_first_active_rank": bias_corrected_rank.get("first_active_rank", ""),
            "balanced_corrected_ef1": balanced_corrected_rank.get("ef1", ""),
            "balanced_corrected_ef5": balanced_corrected_rank.get("ef5", ""),
            "balanced_corrected_auc_roc": balanced_corrected_rank.get("auc_roc", ""),
            "balanced_corrected_first_active_rank": balanced_corrected_rank.get("first_active_rank", ""),
            "prep_protocol_risk": prep_audit.get("risk_level", ""),
            "prep_retained_metals": _join_csv_list(prep_detected.get("retained_metal_elements")),
            "prep_removed_metals": _join_csv_list(prep_detected.get("removed_metal_elements")),
            "prep_retained_cofactors": _join_csv_list(prep_detected.get("retained_cofactors")),
            "prep_removed_cofactors": _join_csv_list(prep_detected.get("removed_cofactors")),
            "prep_ablation_suggestions": _join_csv_list(prep_audit.get("benchmark_ablation_suggestions")),
            "generated_at": metrics.get("generated_at", ""),
            "report_path": metrics.get("report_path", ""),
            "provenance_path": metrics.get(
                "provenance_path",
                (
                    str(cell / "provenance.json")
                    if (cell / "provenance.json").exists()
                    else ""
                ),
            ),
        }
        metrics_rows.append(metric_row)

        failure_rows.append(
            {
                "benchmark_id": manifest.get("benchmark_id", ""),
                "target_id": manifest.get("target_id", ""),
                "run_label": run_label,
                "status": failure.get("status", metrics.get("status", "pending")),
                "failure_stage": failure.get("failure_stage", ""),
                "failure_reason": failure.get("failure_reason", ""),
                "retry_count": failure.get("retry_count", 0),
                "recovered": failure.get("recovered", False),
                "notes": failure.get("notes", ""),
            }
        )

        throughput_rows.append(
            {
                "benchmark_id": manifest.get("benchmark_id", ""),
                "target_id": manifest.get("target_id", ""),
                "run_label": run_label,
                "total_screened": metrics.get("total_screened", ""),
                "ligands_per_hour": metrics.get("ligands_per_hour", ""),
                "runtime_seconds": metrics.get("runtime_seconds", ""),
                "generated_at": metrics.get("generated_at", ""),
                "local_python_version": provenance.get("local_python_version", ""),
                "local_platform": provenance.get("local_platform", ""),
            }
        )

        if (
            metric_row["status"] == "complete"
            and metric_row["ef1"] != ""
            and metric_row["ef5"] != ""
            and metric_row["auc_roc"] != ""
        ):
            complete_metric_rows.append(metric_row)

    variance_groups: dict[tuple, list[dict]] = {}
    for row in complete_metric_rows:
        key = (
            row["benchmark_id"],
            row["target_id"],
            row["receptor_regime"],
            row["pocket_mode"],
            row["engine_path"],
            row["baseline_name"],
        )
        variance_groups.setdefault(key, []).append(row)

    variance_rows: list[dict] = []
    for key, rows in sorted(variance_groups.items()):
        ef1_vals = [float(r["ef1"]) for r in rows]
        ef5_vals = [float(r["ef5"]) for r in rows]
        auc_vals = [float(r["auc_roc"]) for r in rows]
        variance_rows.append(
            {
                "benchmark_id": key[0],
                "target_id": key[1],
                "receptor_regime": key[2],
                "pocket_mode": key[3],
                "engine_path": key[4],
                "baseline_name": key[5],
                "repeat_count": len(rows),
                "ef1_mean": f"{mean(ef1_vals):.6f}",
                "ef1_std": f"{pstdev(ef1_vals):.6f}",
                "ef1_min": f"{min(ef1_vals):.6f}",
                "ef1_max": f"{max(ef1_vals):.6f}",
                "ef5_mean": f"{mean(ef5_vals):.6f}",
                "ef5_std": f"{pstdev(ef5_vals):.6f}",
                "ef5_min": f"{min(ef5_vals):.6f}",
                "ef5_max": f"{max(ef5_vals):.6f}",
                "auc_roc_mean": f"{mean(auc_vals):.6f}",
                "auc_roc_std": f"{pstdev(auc_vals):.6f}",
                "auc_roc_min": f"{min(auc_vals):.6f}",
                "auc_roc_max": f"{max(auc_vals):.6f}",
            }
        )

    summary_dir = campaign_dir / "summary"
    metrics_path = _write_csv(
        summary_dir / "metrics_table.csv",
        [
            "benchmark_id",
            "target_id",
            "target_class",
            "receptor_regime",
            "pocket_mode",
            "engine_path",
            "baseline_name",
            "run_label",
            "status",
            "ef1",
            "ef5",
            "ef20",
            "auc_roc",
            "bedroc",
            "coverage",
            "score_col",
            "external_baseline_status",
            "external_baseline_failed_metrics",
            "external_baseline_auc_floor",
            "external_baseline_ef1_floor",
            "external_baseline_ef5_floor",
            "external_baseline_source",
            "credibility_gate_status",
            "credibility_gate_failed",
            "credibility_gate_warned",
            "redock_rmsd",
            "n_actives",
            "n_decoys",
            "total_screened",
            "first_active_rank",
            "pose_retention_fraction",
            "counts_before_prep",
            "counts_after_prep",
            "counts_after_aggregation",
            "counts_scored",
            "counts_failed",
            "counts_filtered",
            "counts_unmatched",
            "matched_by_identifier",
            "matched_by_smiles_only",
            "unmatched_active_identifiers",
            "matched_active_entries",
            "unmatched_active_entries",
            "top_1_percent_actives",
            "top_5_percent_actives",
            "active_unscored_entries",
            "active_missing_entries",
            "active_unscored_reason_counts",
            "ranking_flags",
            "top_1_percent_charged_fraction",
            "top_5_percent_charged_fraction",
            "top_1_percent_decoy_count",
            "top_5_percent_decoy_count",
            "charge_size_adjusted_ef1",
            "charge_size_adjusted_ef5",
            "charge_size_adjusted_auc_roc",
            "charge_size_adjusted_first_active_rank",
            "bias_corrected_ef1",
            "bias_corrected_ef5",
            "bias_corrected_auc_roc",
            "bias_corrected_first_active_rank",
            "balanced_corrected_ef1",
            "balanced_corrected_ef5",
            "balanced_corrected_auc_roc",
            "balanced_corrected_first_active_rank",
            "prep_protocol_risk",
            "prep_retained_metals",
            "prep_removed_metals",
            "prep_retained_cofactors",
            "prep_removed_cofactors",
            "prep_ablation_suggestions",
            "generated_at",
            "report_path",
            "provenance_path",
        ],
        metrics_rows,
    )
    failure_path = _write_csv(
        summary_dir / "failure_table.csv",
        [
            "benchmark_id",
            "target_id",
            "run_label",
            "status",
            "failure_stage",
            "failure_reason",
            "retry_count",
            "recovered",
            "notes",
        ],
        failure_rows,
    )
    throughput_path = _write_csv(
        summary_dir / "throughput_table.csv",
        [
            "benchmark_id",
            "target_id",
            "run_label",
            "total_screened",
            "ligands_per_hour",
            "runtime_seconds",
            "generated_at",
            "local_python_version",
            "local_platform",
        ],
        throughput_rows,
    )
    variance_path = _write_csv(
        summary_dir / "variance_table.csv",
        [
            "benchmark_id",
            "target_id",
            "receptor_regime",
            "pocket_mode",
            "engine_path",
            "baseline_name",
            "repeat_count",
            "ef1_mean",
            "ef1_std",
            "ef1_min",
            "ef1_max",
            "ef5_mean",
            "ef5_std",
            "ef5_min",
            "ef5_max",
            "auc_roc_mean",
            "auc_roc_std",
            "auc_roc_min",
            "auc_roc_max",
        ],
        variance_rows,
    )

    return {
        "campaign_dir": campaign_dir,
        "cell_count": len(cells),
        "metrics_path": metrics_path,
        "failure_path": failure_path,
        "throughput_path": throughput_path,
        "variance_path": variance_path,
    }
