from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import secrets
import shutil
from pathlib import Path
import subprocess
import sys
import time

from ezscreen.benchmark.rescoring import compare_rescore_columns
from ezscreen.benchmark.runner import (
    BIAS_CORRECTED_SCORE_COL,
    BIAS_CORRECTION_FORMULA,
    pose_retention_diagnostics,
    run_benchmark,
    write_bias_corrected_scores,
)
from ezscreen.benchmark.workspace import (
    create_campaign_scaffold,
    create_cell_scaffold,
)
from ezscreen.results.report_html import write_benchmark_report
from ezscreen.results.variants import source_name


def init_scaffold(
    benchmark_id: str,
    *,
    target: str | None = None,
    regime: str | None = None,
    run_label: str | None = None,
    root: Path | None = None,
) -> Path:
    target = target.strip() if target else None
    regime = regime.strip() if regime else None
    run_label = run_label.strip() if run_label else None
    root = root.expanduser() if root else None

    cell_args = (target, regime, run_label)
    if any(cell_args) and not all(cell_args):
        raise ValueError(
            "target, regime, and run_label must be provided together to create a cell scaffold"
        )

    if all(cell_args):
        return create_cell_scaffold(
            benchmark_id,
            target,
            regime,
            run_label,
            root=root,
        )

    return create_campaign_scaffold(benchmark_id, root=root)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _git_commit(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_provenance(
    cell_dir: Path,
    *,
    actives_path: Path,
    scores_csv: Path,
    score_col: str | None = None,
    intended_compounds_path: Path | None = None,
) -> Path:
    provenance_path = cell_dir / "provenance.json"
    provenance = _load_json(provenance_path)
    timestamps = provenance.get("timestamps")
    if not isinstance(timestamps, dict):
        timestamps = {}
    timestamps["benchmark_run_completed_at"] = datetime.now(timezone.utc).isoformat()

    package_versions = provenance.get("package_versions")
    if not isinstance(package_versions, dict):
        package_versions = {}
    package_versions.setdefault("rdkit", "")
    package_versions.setdefault("matplotlib", "")

    provenance.update(
        {
            "ezscreen_commit": _git_commit(cell_dir),
            "local_python_version": sys.version.split()[0],
            "local_platform": platform.platform(),
            "package_versions": package_versions,
            "settings_snapshot": provenance.get("settings_snapshot") if isinstance(provenance.get("settings_snapshot"), dict) else {},
            "receptor_sources": provenance.get("receptor_sources") if isinstance(provenance.get("receptor_sources"), dict) else {},
            "pocket_sources": provenance.get("pocket_sources") if isinstance(provenance.get("pocket_sources"), dict) else {},
            "seeds": provenance.get("seeds") if isinstance(provenance.get("seeds"), dict) else {},
            "timestamps": timestamps,
        }
    )
    settings_snapshot = provenance["settings_snapshot"]
    settings_snapshot["benchmark_actives_path"] = str(actives_path)
    settings_snapshot["benchmark_scores_csv"] = str(scores_csv)
    input_hashes = provenance.get("input_hashes_sha256")
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    input_hashes.update(
        {
            "actives": _sha256(actives_path),
            "scores": _sha256(scores_csv),
        }
    )
    provenance["input_hashes_sha256"] = input_hashes
    if intended_compounds_path:
        settings_snapshot["benchmark_intended_compounds_path"] = str(
            intended_compounds_path
        )
        provenance["input_hashes_sha256"]["intended_compounds"] = _sha256(
            intended_compounds_path
        )
    if score_col:
        settings_snapshot["benchmark_score_col"] = score_col
    return _write_json(provenance_path, provenance)




def _canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def _active_unscored_diagnostics(actives_path: Path, scores_csv: Path) -> dict[str, object]:
    unscored_path = scores_csv.parent / "unscored_reasons.csv"
    canonical_cache: dict[str, str] = {}

    def _cached_canonical_smiles(smiles: str) -> str:
        smiles = smiles.strip()
        if not smiles:
            return ""
        if smiles not in canonical_cache:
            canonical_cache[smiles] = _canonical_smiles(smiles)
        return canonical_cache[smiles]

    active_entries: list[dict[str, object]] = []
    with actives_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            ids = {part.strip() for part in parts[1:] if part.strip()}
            active_entries.append(
                {
                    "smiles": smiles,
                    "canonical_smiles": _cached_canonical_smiles(smiles),
                    "ids": ids,
                }
            )

    def _row_ids(row: dict[str, str]) -> set[str]:
        ids = {
            str(row.get(key, "")).strip()
            for key in ("name", "id", "ligand", "chembl_id", "compound_id")
            if str(row.get(key, "")).strip()
        }
        return ids | {source_name(row_id) for row_id in ids if row_id}

    def _row_canonical_smiles(row: dict[str, str]) -> str:
        return _cached_canonical_smiles(str(row.get("smiles", "")).strip())

    with scores_csv.open(newline="", encoding="utf-8") as handle:
        scored_rows = list(csv.DictReader(handle))

    unscored_rows: list[dict[str, str]] = []
    if unscored_path.exists():
        with unscored_path.open(newline="", encoding="utf-8") as handle:
            unscored_rows = list(csv.DictReader(handle))

    scored_ids: set[str] = set()
    scored_smiles: set[str] = set()
    for row in scored_rows:
        scored_ids.update(_row_ids(row))
        canonical = _row_canonical_smiles(row)
        if canonical:
            scored_smiles.add(canonical)

    unscored_by_id: dict[str, list[dict[str, str]]] = {}
    unscored_by_smiles: dict[str, list[dict[str, str]]] = {}
    for row in unscored_rows:
        for row_id in _row_ids(row):
            unscored_by_id.setdefault(row_id, []).append(row)
        canonical = _row_canonical_smiles(row)
        if canonical:
            unscored_by_smiles.setdefault(canonical, []).append(row)

    active_unscored_entries = 0
    active_missing_entries = 0
    reason_counts: dict[str, int] = {}
    missing_entries: list[str] = []
    unscored_entries: list[str] = []
    for entry in active_entries:
        ids = entry.get("ids")
        entry_ids = ids if isinstance(ids, set) else set()
        canonical = str(entry.get("canonical_smiles", ""))
        if (entry_ids and scored_ids.intersection(entry_ids)) or (canonical and canonical in scored_smiles):
            continue
        matching_by_identity: list[dict[str, str]] = []
        for entry_id in entry_ids:
            matching_by_identity.extend(unscored_by_id.get(entry_id, []))
        matching_unscored = matching_by_identity or unscored_by_smiles.get(canonical, [])
        label = next(iter(entry_ids), "") or str(entry.get("smiles", ""))
        if matching_unscored:
            active_unscored_entries += 1
            unscored_entries.append(label)
            reasons = {
                str(row.get("reason") or row.get("stage") or row.get("status") or "unknown").strip()
                or "unknown"
                for row in matching_unscored
            }
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            active_missing_entries += 1
            missing_entries.append(label)

    return {
        "unscored_file": str(unscored_path) if unscored_path.exists() else "",
        "active_file_entries": len(active_entries),
        "active_unscored_entries": active_unscored_entries,
        "active_missing_entries": active_missing_entries,
        "active_unscored_reason_counts": dict(sorted(reason_counts.items())),
        "active_unscored_labels": unscored_entries,
        "active_missing_labels": missing_entries,
    }
def _enforce_benchmark_active_survival(diagnostics: dict[str, object]) -> None:
    """Retain active failures in QC while metric ranking assigns worst rank."""
    return None



def _gate_item(status: str, message: str, details: dict | None = None) -> dict[str, object]:
    return {"status": status, "message": message, "details": details or {}}


def _benchmark_credibility_gate(result, unscored_diagnostics: dict, pose_diagnostics: dict) -> dict[str, object]:
    """Build pass/warn/fail benchmark trust gates without changing EF/AUC."""
    gates: dict[str, dict[str, object]] = {}
    label_diag = result.label_diagnostics if isinstance(result.label_diagnostics, dict) else {}
    ranking_diag = result.ranking_diagnostics if isinstance(result.ranking_diagnostics, dict) else {}

    active_unscored = int(unscored_diagnostics.get("active_unscored_entries") or 0)
    active_missing = int(unscored_diagnostics.get("active_missing_entries") or 0)
    if active_unscored or active_missing:
        gates["active_survival"] = _gate_item(
            "fail",
            "Known actives were lost before scoring; EF/AUC are not trustworthy.",
            {"active_unscored_entries": active_unscored, "active_missing_entries": active_missing},
        )
    else:
        gates["active_survival"] = _gate_item("pass", "All known active entries survived into scored results.")

    ef1 = float(result.ef1 or 0.0)
    ef5 = float(result.ef5 or 0.0)
    auc = float(result.auc_roc or 0.0)
    if ef1 >= 3.0 and ef5 >= 2.0 and auc >= 0.6:
        gates["early_enrichment"] = _gate_item("pass", "Raw EF1/EF5/AUC clear conservative docking floors.", {"ef1": ef1, "ef5": ef5, "auc_roc": auc})
    elif auc >= 0.6 or ef1 >= 1.5 or ef5 >= 1.5:
        gates["early_enrichment"] = _gate_item("warn", "Raw enrichment is partial/modest; do not use as broad reliability proof.", {"ef1": ef1, "ef5": ef5, "auc_roc": auc})
    else:
        gates["early_enrichment"] = _gate_item("fail", "Raw early enrichment is below conservative docking floors.", {"ef1": ef1, "ef5": ef5, "auc_roc": auc})

    first_active_rank = label_diag.get("first_active_rank")
    try:
        first_active_rank_int = int(first_active_rank)
    except (TypeError, ValueError):
        first_active_rank_int = None
    if first_active_rank_int is None:
        gates["top20_mix"] = _gate_item("fail", "No active appears in the ranked list.")
    elif first_active_rank_int <= 20:
        gates["top20_mix"] = _gate_item("pass", "At least one active appears in the top 20.", {"first_active_rank": first_active_rank_int})
    else:
        gates["top20_mix"] = _gate_item("fail", "Top 20 contains no known actives.", {"first_active_rank": first_active_rank_int})

    pose_flags = set(str(flag) for flag in pose_diagnostics.get("flags", []) if flag)
    fail_pose_flags = {
        "poses_sdf_missing_or_empty",
        "top_window_missing_retained_poses",
        "score_floor_rows_missing_retained_poses",
        "no_retained_active_poses",
    }
    warn_pose_flags = {"score_ceiling_rows_present", "low_pose_retention_fraction"}
    if pose_flags & fail_pose_flags:
        gates["pose_retention"] = _gate_item(
            "fail",
            "Top-window or active pose evidence is missing; top-hit docking evidence is unauditable.",
            {"flags": sorted(pose_flags & fail_pose_flags), "top_window_missing_pose_count": pose_diagnostics.get("top_window_missing_pose_count"), "retained_active_pose_count": pose_diagnostics.get("retained_active_pose_count")},
        )
    elif pose_flags & warn_pose_flags:
        gates["pose_retention"] = _gate_item(
            "warn",
            "Pose evidence exists, but score QC or low retention needs review.",
            {"flags": sorted(pose_flags & warn_pose_flags)},
        )
    else:
        gates["pose_retention"] = _gate_item("pass", "Top-window and active pose retention checks passed.")

    ranking_flags = set(str(flag) for flag in ranking_diag.get("flags", []) if flag)
    if "best_scoring_ligand_is_decoy" in ranking_flags:
        gates["top_rank_identity"] = _gate_item("warn", "Best-scoring ligand is a decoy; inspect top hits before claims.")
    else:
        gates["top_rank_identity"] = _gate_item("pass", "Best-scoring ligand is not flagged as a decoy.")

    if "top_1_percent_enriched_for_formally_charged_ligands" in ranking_flags:
        gates["charge_bias"] = _gate_item("warn", "Top 1% is enriched for formally charged ligands.")
    else:
        gates["charge_bias"] = _gate_item("pass", "No top-window formal-charge enrichment flag fired.")

    severity = {"pass": 0, "warn": 1, "fail": 2}
    overall = "pass"
    for gate in gates.values():
        if severity[str(gate["status"])] > severity[overall]:
            overall = str(gate["status"])
    return {
        "status": overall,
        "summary": "Benchmark credibility gate " + overall,
        "gates": gates,
    }

def generate_benchmark_artifacts(
    *,
    actives_path: Path,
    scores_csv: Path,
    report_path: Path,
    metrics_path: Path | None = None,
    failure_path: Path | None = None,
    provenance_path: Path | None = None,
    context_dir: Path | None = None,
    score_col: str | None = None,
    intended_compounds_path: Path | None = None,
) -> dict[str, object]:
    actives_path = actives_path.expanduser().resolve()
    scores_csv = scores_csv.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve() if metrics_path else None
    failure_path = failure_path.expanduser().resolve() if failure_path else None
    provenance_path = provenance_path.expanduser().resolve() if provenance_path else None
    context_dir = context_dir.expanduser().resolve() if context_dir else report_path.parent

    retry_count = 0
    if failure_path:
        failure_summary = _load_json(failure_path)
        retry_count = int(failure_summary.get("retry_count") or 0)

    try:
        if not actives_path.exists():
            raise FileNotFoundError(f"Benchmark actives file not found: {actives_path}")
        if not scores_csv.exists():
            raise FileNotFoundError(f"Benchmark scores file not found: {scores_csv}")

        unscored_diagnostics = _active_unscored_diagnostics(actives_path, scores_csv)
        try:
            _enforce_benchmark_active_survival(unscored_diagnostics)
        except ValueError:
            if metrics_path:
                _write_json(
                    metrics_path,
                    {
                        "status": "failed",
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "actives_path": str(actives_path),
                        "scores_csv": str(scores_csv),
                        "report_path": "",
                        "ef1": None,
                        "ef5": None,
                        "auc_roc": None,
                        "redock_rmsd": None,
                        "n_actives": 0,
                        "n_decoys": 0,
                        "total_screened": 0,
                        "label_diagnostics": {},
                        "ranking_diagnostics": {},
                        "unscored_diagnostics": unscored_diagnostics,
                        "score_col": score_col or "",
                    },
                )
            raise

        result = run_benchmark(
            actives_path,
            scores_csv,
            score_col=score_col,
            intended_compounds_path=intended_compounds_path,
        )
        pose_diagnostics = pose_retention_diagnostics(actives_path, scores_csv, score_col=score_col)
        ranking_diagnostics = result.ranking_diagnostics if isinstance(result.ranking_diagnostics, dict) else {}
        flags = ranking_diagnostics.get("flags") if isinstance(ranking_diagnostics.get("flags"), list) else []
        for flag in pose_diagnostics.get("flags", []):
            if flag not in flags:
                flags.append(str(flag))
        ranking_diagnostics["flags"] = flags
        credibility_gate = _benchmark_credibility_gate(result, unscored_diagnostics, pose_diagnostics)
        ranking_diagnostics["credibility_gate"] = credibility_gate
        gate_flag = f"credibility_gate_{credibility_gate['status']}"
        if gate_flag not in flags:
            flags.append(gate_flag)
        result.ranking_diagnostics = ranking_diagnostics
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_benchmark_report(result, report_path)
        if provenance_path:
            _update_provenance(
                provenance_path.parent,
                actives_path=actives_path,
                scores_csv=scores_csv,
                score_col=score_col,
                intended_compounds_path=intended_compounds_path,
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        metrics = {
            "metric_contract_version": 2,
            "metric_contract": {
                "unit": "source_compound",
                "source_aggregation": "best_score_across_variants_conformers_receptors",
                "auc_ties": "half_credit",
                "ef_boundary_ties": "expected_active_fraction",
                "failed_intended_compounds": "shared_worst_rank",
                "bedroc_alpha": 20.0,
            },
            'ef20': result.ef20,
            'bedroc': result.bedroc,
            'bedroc_alpha': 20.0,
            'coverage': result.coverage,
            'first_active_rank': result.first_active_rank,
            "status": "complete",
            "generated_at": timestamp,
            "actives_path": str(actives_path),
            "scores_csv": str(scores_csv),
            "report_path": str(report_path),
            "ef1": result.ef1,
            "ef5": result.ef5,
            "auc_roc": result.auc_roc,
            "n_actives": result.n_actives,
            "n_decoys": result.n_decoys,
            "total_screened": result.total_screened,
            "label_diagnostics": result.label_diagnostics,
            "ranking_diagnostics": result.ranking_diagnostics,
            "pose_retention_diagnostics": pose_diagnostics,
            "credibility_gate": credibility_gate,
            "score_col": score_col or result.label_diagnostics.get("score_column", ""),
        }
        metrics["unscored_diagnostics"] = unscored_diagnostics
        if provenance_path:
            metrics["provenance_path"] = str(provenance_path)
        if metrics_path:
            _write_json(metrics_path, metrics)
        if failure_path:
            _write_json(
                failure_path,
                {
                    "status": "complete",
                    "failure_stage": "",
                    "failure_reason": "",
                    "retry_count": retry_count,
                    "recovered": False,
                    "notes": "benchmark metrics, provenance, and report generated" if provenance_path else "benchmark metrics and report generated",
                },
            )
        return {
            "context_dir": context_dir,
            "cell_dir": context_dir,
            "actives_path": actives_path,
            "scores_csv": scores_csv,
            "metrics_path": metrics_path,
            "provenance_path": provenance_path,
            "report_path": report_path,
            "result": metrics,
        }
    except Exception as exc:
        if failure_path:
            _write_json(
                failure_path,
                {
                    "status": "failed",
                    "failure_stage": "benchmark_run",
                    "failure_reason": str(exc),
                    "retry_count": retry_count,
                    "recovered": False,
                    "notes": "benchmark metrics failed; diagnostics written to metrics_summary.json when available",
                },
            )
        raise


def run_cell_benchmark(
    cell_dir: Path,
    *,
    actives_path: Path | None = None,
    scores_csv: Path | None = None,
    report_name: str = "benchmark_report.html",
    score_col: str | None = None,
) -> dict[str, Path | dict]:
    cell_dir = cell_dir.expanduser().resolve()
    if not cell_dir.exists():
        raise FileNotFoundError(f"Benchmark cell not found: {cell_dir}")

    actives_path = actives_path if actives_path else cell_dir / "inputs" / "actives.smi"
    scores_csv = scores_csv if scores_csv else cell_dir / "merged_outputs" / "scores.csv"
    report_path = cell_dir / "reports" / report_name
    metrics_path = cell_dir / "metrics_summary.json"
    failure_path = cell_dir / "failure_summary.json"
    provenance_path = cell_dir / "provenance.json"

    intended_compounds_path = next(
        (
            path
            for path in (
                cell_dir / 'inputs' / 'intended_compounds.csv',
                cell_dir / 'inputs' / 'intended_compounds.smi',
                cell_dir / 'inputs' / 'library.smi',
            )
            if path.exists()
        ),
        None,
    )
    return generate_benchmark_artifacts(
        actives_path=actives_path,
        scores_csv=scores_csv,
        report_path=report_path,
        metrics_path=metrics_path,
        failure_path=failure_path,
        provenance_path=provenance_path,
        context_dir=cell_dir,
        score_col=score_col,
        intended_compounds_path=intended_compounds_path,
    )


def correct_scores_file(
    scores_csv: Path,
    *,
    output_csv: Path | None = None,
    score_col: str | None = None,
    output_col: str = BIAS_CORRECTED_SCORE_COL,
) -> dict[str, object]:
    return write_bias_corrected_scores(
        scores_csv,
        output_csv,
        score_col=score_col,
        output_col=output_col,
    )



def compare_rescore_file(
    *,
    actives_path: Path,
    scores_csv: Path,
    output_dir: Path | None = None,
    score_cols: list[str] | None = None,
) -> dict[str, object]:
    return compare_rescore_columns(
        actives_path=actives_path,
        scores_csv=scores_csv,
        output_dir=output_dir,
        score_cols=score_cols,
    )

def _parse_vec3(value: str, *, label: str) -> list[float]:
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{label} must contain three comma-separated numbers")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{label} must contain only numbers") from exc


def _copy_tree_contents(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _archive_kaggle_output(output_dir: Path, cell_dir: Path) -> Path:
    raw_dir = cell_dir / "raw_outputs"
    merged_dir = cell_dir / "merged_outputs"
    _copy_tree_contents(output_dir, raw_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    scores = output_dir / "scores.csv"
    if not scores.exists():
        raise FileNotFoundError(f"Kaggle output did not contain scores.csv: {output_dir}")
    shutil.copy2(scores, merged_dir / "scores.csv")
    for name in ("results_meta.json", "unscored_reasons.csv", "poses.sdf", "failed_prep.csv", "failed_docking.csv"):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, merged_dir / name)
    return merged_dir / "scores.csv"


def submit_kaggle_cell(
    cell_dir: Path,
    *,
    receptor_pdbqt: Path,
    ligands: Path,
    box_center: str,
    box_size: str,
    actives_path: Path | None = None,
    run_id: str | None = None,
    n_shards: int | None = None,
    accelerator: str = "nvidiaTeslaT4",
    engine: str = "unidock",
    search_mode: str = "balance",
    enumerate_enabled: bool = False,
    intended_compounds_path: Path | None = None,
    retry_limit: int = 3,
    kaggle_username: str | None = None,
    kaggle_json_path: Path | None = None,
    campaign_dir: Path | None = None,
    campaign_account: str | None = None,
    campaign_stage: str = "baseline",
    planned_gpu_hours: float | None = None,
) -> dict[str, object]:
    """Submit a benchmark cell to Kaggle and archive results back into the cell.

    This path intentionally avoids local docking and local ligand preparation. It only
    shards raw input locally, renders the Kaggle notebook with prep_on_kaggle=True,
    then lets Kaggle do prep + docking.
    """
    from ezscreen import auth, config
    from ezscreen.backends import engines
    from ezscreen.backends.kaggle import runner as kaggle_runner
    from ezscreen.prep import ligands as ligand_prep

    cell_dir = cell_dir.expanduser().resolve()
    if not cell_dir.exists():
        raise FileNotFoundError(f"Benchmark cell not found: {cell_dir}")
    receptor_pdbqt = receptor_pdbqt.expanduser().resolve()
    ligands = ligands.expanduser().resolve()
    if not receptor_pdbqt.exists():
        raise FileNotFoundError(f"Prepared receptor PDBQT not found: {receptor_pdbqt}")
    if not ligands.exists():
        raise FileNotFoundError(f"Benchmark ligand library not found: {ligands}")

    center = _parse_vec3(box_center, label="box_center")
    size = _parse_vec3(box_size, label="box_size")
    run_id = run_id or "ezs-bench-" + secrets.token_hex(3)
    work_dir = Path.home() / ".ezscreen" / "runs" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    if not kaggle_username or kaggle_json_path is None:
        try:
            creds = auth.load_credentials()
            loaded_path = Path(str(creds.get("kaggle_json_path", ""))).expanduser()
            if kaggle_json_path is None and loaded_path.exists():
                kaggle_json_path = loaded_path
            if not kaggle_username:
                with open(loaded_path, encoding="utf-8") as f:
                    kaggle_username = json.load(f).get("username") or "user"
        except Exception:
            kaggle_username = kaggle_username or "user"

    campaign_path = campaign_dir.expanduser().resolve() if campaign_dir else None
    if campaign_path is not None and float(planned_gpu_hours or 0.0) <= 0.0:
        raise ValueError(
            "Supervised Kaggle submissions require positive --planned-gpu-hours"
        )

    def observe_campaign_job(event: str, details: dict[str, object]) -> None:
        if campaign_path is None:
            return
        from ezscreen.benchmark import supervisor

        if event == "preparing":
            supervisor.register_job(
                campaign_path,
                {
                    "job_id": run_id,
                    "status": "preparing",
                    "account": campaign_account or str(kaggle_username),
                    "stage": campaign_stage,
                    "planned_gpu_hours": float(planned_gpu_hours or 0.0),
                    **details,
                },
            )
            return
        updates = dict(details)
        updates["status"] = "failed" if event in {"timeout", "error"} else event
        supervisor.update_registered_job(campaign_path, run_id, **updates)

    cfg = config.load()
    cfg = cfg if isinstance(cfg, dict) else {}
    prep_section = cfg.get("prep", {})
    prep_cfg = prep_section if isinstance(prep_section, dict) else {}
    prep_cfg = dict(prep_cfg)
    prep_cfg["enumerate_enabled"] = enumerate_enabled
    prep_cfg["enumerate_protonation"] = bool(enumerate_enabled)
    prep_cfg["enumerate_tautomers"] = bool(enumerate_enabled)
    prep_cfg["enumerate_stereo"] = False
    prep_cfg["enumerate_ring"] = False
    try:
        max_variants = int(prep_cfg.get("max_variants_per_ligand", 4))
    except (TypeError, ValueError):
        max_variants = 4
    prep_cfg["max_variants_per_ligand"] = (
        max(2, max_variants) if enumerate_enabled else 1
    )

    shard_result = ligand_prep.shard_raw(
        input_path=ligands,
        output_dir=work_dir / "shards",
        n_shards=n_shards,
    )
    shard_paths = [Path(path) for path in shard_result["shard_paths"]]
    if not shard_paths:
        raise RuntimeError("Benchmark ligand library produced no raw shards")

    inputs_dir = cell_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    receptor_archive = inputs_dir / "submitted_receptor.pdbqt"
    library_archive = inputs_dir / f"submitted_library{ligands.suffix.lower()}"
    shutil.copy2(receptor_pdbqt, receptor_archive)
    shutil.copy2(ligands, library_archive)

    intended_archive = inputs_dir / "intended_compounds.csv"
    if intended_compounds_path is not None:
        intended_compounds_path = intended_compounds_path.expanduser().resolve()
        if not intended_compounds_path.exists():
            raise FileNotFoundError(
                f"Intended-compound manifest not found: {intended_compounds_path}"
            )
        if intended_compounds_path != intended_archive.resolve():
            shutil.copy2(intended_compounds_path, intended_archive)
    else:
        intended_rows: list[dict[str, str]] = []
        for shard_path in shard_paths:
            if shard_path.suffix.lower() not in {".smi", ".ism"}:
                raise ValueError(
                    "Non-SMILES benchmark inputs require --intended-compounds so "
                    "failed source compounds remain in the metric denominator"
                )
            for line in shard_path.read_text(encoding="utf-8").splitlines():
                parts = line.split(maxsplit=1)
                if not parts:
                    continue
                smiles = parts[0]
                name = parts[1].strip() if len(parts) > 1 else ""
                source_id = name or f"source_{len(intended_rows) + 1:08d}"
                intended_rows.append(
                    {
                        "source_compound_id": source_id,
                        "name": name,
                        "smiles": smiles,
                    }
                )
        with intended_archive.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_compound_id", "name", "smiles"],
            )
            writer.writeheader()
            writer.writerows(intended_rows)

    notebook_src = kaggle_runner.render_vina_notebook(
        run_id=run_id,
        shard_filenames=[path.name for path in shard_paths],
        # upload_run_dataset normalizes the primary receptor to this stable name.
        receptor_filename="receptor.pdbqt",
        box_center=center,
        box_size=size,
        search_mode=search_mode,
        ph=float(cfg.get("run", {}).get("default_ph", 7.4)),
        prep_on_kaggle=True,
        accelerator=accelerator,
        prep_cfg=prep_cfg,
        results_cfg=cfg.get("results", {}),
        engine=engine,
        scoring=engines.default_scoring(engine) or "vina",
    )
    notebook_path = work_dir / "benchmark_kaggle.ipynb"
    notebook_path.write_text(notebook_src, encoding="utf-8")

    result = kaggle_runner.run_screening_job(
        run_id=run_id,
        receptor_pdbqt=receptor_pdbqt,
        shard_paths=shard_paths,
        notebook_path=notebook_path,
        username=str(kaggle_username),
        work_dir=work_dir,
        retry_limit=retry_limit,
        accelerator=accelerator,
        engine=engine,
        kaggle_json_path=kaggle_json_path,
        job_observer=observe_campaign_job if campaign_path is not None else None,
    )
    if result.get("status") != "complete":
        _write_json(
            cell_dir / "failure_summary.json",
            {
                "status": result.get("status", "failed"),
                "failure_stage": "kaggle_screening",
                "failure_reason": result.get("error_type") or result.get("status", "unknown"),
                "retry_count": retry_limit,
                "recovered": False,
                "notes": f"remote benchmark run_id={run_id}",
            },
        )
        return {"status": result.get("status", "failed"), "run_id": run_id, "work_dir": work_dir, "result": result}

    output_dir = Path(str(result["output_dir"]))
    scores_csv = _archive_kaggle_output(output_dir, cell_dir)
    logs_dir = cell_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, logs_dir / "benchmark_kaggle.ipynb")

    provenance = _load_json(cell_dir / "provenance.json")
    kaggle_runtime = provenance.get("kaggle_runtime_identifiers")
    if not isinstance(kaggle_runtime, dict):
        kaggle_runtime = {}
    kaggle_runtime.update(
        {
            "run_id": run_id,
            "work_dir": str(work_dir),
            "accelerator_requested": accelerator,
            "engine": engine,
            "prep_on_kaggle": True,
            "enumerate_enabled": enumerate_enabled,
            "shard_count": len(shard_paths),
        }
    )
    provenance["kaggle_runtime_identifiers"] = kaggle_runtime
    settings_snapshot = provenance.get("settings_snapshot")
    if not isinstance(settings_snapshot, dict):
        settings_snapshot = {}
    settings_snapshot["benchmark_prep_cfg"] = prep_cfg
    settings_snapshot["benchmark_box_center"] = center
    settings_snapshot["benchmark_box_size"] = size
    settings_snapshot["benchmark_search_mode"] = search_mode
    provenance["settings_snapshot"] = settings_snapshot
    input_hashes = provenance.get("input_hashes_sha256")
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    input_hashes.update(
        {
            "submitted_receptor": _sha256(receptor_archive),
            "submitted_library": _sha256(library_archive),
            "intended_compounds": _sha256(intended_archive),
            "rendered_notebook": _sha256(notebook_path),
        }
    )
    provenance["input_hashes_sha256"] = input_hashes
    _write_json(cell_dir / "provenance.json", provenance)

    actives_for_metrics = actives_path if actives_path else cell_dir / "inputs" / "actives.smi"
    artifacts = run_cell_benchmark(
        cell_dir,
        actives_path=actives_for_metrics,
        scores_csv=scores_csv,
    )
    return {
        "status": "complete",
        "run_id": run_id,
        "work_dir": work_dir,
        "output_dir": output_dir,
        "scores_csv": scores_csv,
        **artifacts,
    }



def _count_raw_ligand_records(input_path: Path) -> int:
    from ezscreen.prep import ligands as ligand_prep

    total = 0
    for fp in ligand_prep.scan_input(input_path):
        suffix = fp.suffix.lower()
        if suffix in {".smi", ".smiles", ".ism"}:
            with open(fp, encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
        elif suffix == ".sdf":
            from rdkit import Chem

            total += sum(1 for mol in Chem.SDMolSupplier(str(fp), sanitize=False) if mol is not None)
        elif suffix in {".mol2", ".mol", ".pdb"}:
            total += 1
    return total


def _checkpoint_visible_run_config(
    *,
    cell_dir: Path,
    run_id: str,
    receptor_pdbqt: Path,
    ligands: Path,
    box_center: list[float],
    box_size: list[float],
    accelerator: str,
    engine: str,
    scoring: str,
    search_mode: str,
    shard_plan: object,
) -> dict[str, object]:
    return {
        "source": "benchmark_multi_account_kaggle",
        "cell_dir": str(cell_dir),
        "run_id": run_id,
        "receptor_pdbqt": str(receptor_pdbqt),
        "ligands": str(ligands),
        "box": {"center": box_center, "size": box_size},
        "accelerator": accelerator,
        "engine": engine,
        "scoring": scoring,
        "search_mode": search_mode,
        "shard_plan": {
            "ligand_count": getattr(shard_plan, "ligand_count", None),
            "account_count": getattr(shard_plan, "account_count", None),
            "total_notebooks": getattr(shard_plan, "total_notebooks", None),
            "notebooks_per_account": list(getattr(shard_plan, "notebooks_per_account", ())),
            "rationale": getattr(shard_plan, "rationale", ""),
        },
    }


def submit_multi_account_kaggle_cell(
    cell_dir: Path,
    *,
    receptor_pdbqt: Path,
    ligands: Path,
    box_center: str,
    box_size: str,
    actives_path: Path | None = None,
    run_id: str | None = None,
    accelerator: str = "nvidiaTeslaT4",
    engine: str = "unidock",
    scoring: str | None = None,
    search_mode: str = "balance",
    retry_limit: int = 3,
    max_notebooks_per_account: int = 2,
    min_accounts: int | None = None,
    force_notebooks: int | None = None,
) -> dict[str, object]:
    """Submit a benchmark cell through multi-account Kaggle with checkpoint visibility.

    The planner intentionally uses fewer notebooks for small panels because Kaggle
    dataset upload and notebook startup are fixed costs. The created checkpoint row
    makes the run appear under TUI Recent Runs while the benchmark cell still owns
    archived metrics and provenance.
    """
    from ezscreen import auth, checkpoint, config
    from ezscreen.backends import engines as engine_registry
    from ezscreen.backends.kaggle import runner as kaggle_runner
    from ezscreen.benchmark.kaggle_planning import (
        build_account_assignments,
        distribute_notebooks_across_accounts,
        plan_kaggle_shards,
    )
    from ezscreen.prep import ligands as ligand_prep

    cell_dir = cell_dir.expanduser().resolve()
    receptor_pdbqt = receptor_pdbqt.expanduser().resolve()
    ligands = ligands.expanduser().resolve()
    if not cell_dir.exists():
        raise FileNotFoundError(f"Benchmark cell not found: {cell_dir}")
    if not receptor_pdbqt.exists():
        raise FileNotFoundError(f"Prepared receptor PDBQT not found: {receptor_pdbqt}")
    if not ligands.exists():
        raise FileNotFoundError(f"Benchmark ligand library not found: {ligands}")

    accounts = auth.get_all_kaggle_accounts()
    if min_accounts is not None and len(accounts) < min_accounts:
        raise RuntimeError(f"Need at least {min_accounts} Kaggle account(s); found {len(accounts)} configured")
    if not accounts:
        raise RuntimeError("No Kaggle accounts configured")

    center = _parse_vec3(box_center, label="box_center")
    size = _parse_vec3(box_size, label="box_size")
    run_id = run_id or "ezs-bench-" + secrets.token_hex(3)
    work_dir = Path.home() / ".ezscreen" / "runs" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    ligand_count = _count_raw_ligand_records(ligands)
    if ligand_count <= 0:
        raise RuntimeError("Benchmark ligand library produced no countable ligand records")

    if force_notebooks is None:
        shard_plan = plan_kaggle_shards(
            ligand_count,
            len(accounts),
            max_notebooks_per_account=max_notebooks_per_account,
        )
    else:
        notebook_counts = distribute_notebooks_across_accounts(
            force_notebooks,
            len(accounts),
            max_notebooks_per_account=max_notebooks_per_account,
        )
        from ezscreen.benchmark.kaggle_planning import KaggleShardPlan

        shard_plan = KaggleShardPlan(
            ligand_count=ligand_count,
            account_count=len(accounts),
            total_notebooks=force_notebooks,
            notebooks_per_account=notebook_counts,
            rationale=f"forced {force_notebooks} notebook(s) for {ligand_count} ligand(s)",
        )

    shard_result = ligand_prep.shard_raw(
        input_path=ligands,
        output_dir=work_dir / "shards",
        n_shards=shard_plan.total_notebooks,
    )
    shard_paths = [Path(path) for path in shard_result["shard_paths"]]
    if not shard_paths:
        raise RuntimeError("Benchmark ligand library produced no raw shards")

    cfg = config.load()
    cfg = cfg if isinstance(cfg, dict) else {}
    prep_section = cfg.get("prep", {})
    prep_cfg = prep_section if isinstance(prep_section, dict) else {}
    prep_cfg = dict(prep_cfg)
    prep_cfg["enumerate_enabled"] = True
    prep_cfg.setdefault("enumerate_protonation", True)
    prep_cfg.setdefault("enumerate_tautomers", True)
    prep_cfg.setdefault("enumerate_stereo", False)
    prep_cfg.setdefault("enumerate_ring", False)
    prep_cfg["max_variants_per_ligand"] = max(2, int(prep_cfg.get("max_variants_per_ligand", 4) or 4))

    scoring_name = scoring or engine_registry.default_scoring(engine) or "vina"
    account_assignments = build_account_assignments(accounts, shard_plan.notebooks_per_account)

    checkpoint.init_db()
    run_config = _checkpoint_visible_run_config(
        cell_dir=cell_dir,
        run_id=run_id,
        receptor_pdbqt=receptor_pdbqt,
        ligands=ligands,
        box_center=center,
        box_size=size,
        accelerator=accelerator,
        engine=engine,
        scoring=scoring_name,
        search_mode=search_mode,
        shard_plan=shard_plan,
    )
    if checkpoint.get_run(run_id) is None:
        checkpoint.create_run(run_id, run_config, ligand_count)
    for idx, shard_path in enumerate(shard_paths):
        checkpoint.add_shard(run_id, idx, _count_raw_ligand_records(shard_path))

    remote_start = time.perf_counter()
    result = kaggle_runner.run_multi_account_screening(
        run_id=run_id,
        receptor_pdbqt=receptor_pdbqt,
        shard_paths=shard_paths,
        account_assignments=account_assignments,
        work_dir=work_dir,
        box_center=center,
        box_size=size,
        search_mode=search_mode,
        ph=float(cfg.get("run", {}).get("default_ph", 7.4)),
        retry_limit=retry_limit,
        accelerator=accelerator,
        prep_on_kaggle=True,
        enumerate_opts=prep_cfg,
        results_cfg=cfg.get("results", {}),
        engine=engine,
        scoring=scoring_name,
    )

    runtime_seconds = round(time.perf_counter() - remote_start, 3)
    status = str(result.get("status", "failed"))
    if status == "complete":
        checkpoint.mark_run_complete(run_id)
    elif status == "partial":
        checkpoint.mark_run_partial(run_id)
    else:
        checkpoint.mark_run_failed(run_id)
        _write_json(
            cell_dir / "failure_summary.json",
            {
                "status": status,
                "failure_stage": "multi_account_kaggle_screening",
                "failure_reason": result.get("error_type") or result.get("error") or status,
                "retry_count": retry_limit,
                "recovered": False,
                "notes": f"remote benchmark run_id={run_id}; {shard_plan.rationale}; runtime_seconds={runtime_seconds}",
            },
        )
        return {"status": status, "run_id": run_id, "work_dir": work_dir, "shard_plan": shard_plan, "runtime_seconds": runtime_seconds, "result": result}

    output_dir = Path(str(result["output_dir"]))
    scores_csv = _archive_kaggle_output(output_dir, cell_dir)

    provenance = _load_json(cell_dir / "provenance.json")
    kaggle_runtime = provenance.get("kaggle_runtime_identifiers")
    if not isinstance(kaggle_runtime, dict):
        kaggle_runtime = {}
    kaggle_runtime.update(
        {
            "run_id": run_id,
            "work_dir": str(work_dir),
            "accelerator_requested": accelerator,
            "engine": engine,
            "scoring": scoring_name,
            "prep_on_kaggle": True,
            "shard_count": len(shard_paths),
            "multi_account": True,
            "account_count_configured": len(accounts),
            "total_notebooks": shard_plan.total_notebooks,
            "notebooks_per_account": list(shard_plan.notebooks_per_account),
            "shard_planner_rationale": shard_plan.rationale,
            "runtime_seconds": runtime_seconds,
        }
    )
    provenance["kaggle_runtime_identifiers"] = kaggle_runtime
    settings_snapshot = provenance.get("settings_snapshot")
    if not isinstance(settings_snapshot, dict):
        settings_snapshot = {}
    settings_snapshot["benchmark_prep_cfg"] = prep_cfg
    provenance["settings_snapshot"] = settings_snapshot
    _write_json(cell_dir / "provenance.json", provenance)

    actives_for_metrics = actives_path if actives_path else cell_dir / "inputs" / "actives.smi"
    artifacts = run_cell_benchmark(cell_dir, actives_path=actives_for_metrics, scores_csv=scores_csv)
    metrics_path = artifacts.get("metrics_path")
    if metrics_path:
        metrics = _load_json(Path(str(metrics_path)))
        metrics["runtime_seconds"] = runtime_seconds
        if metrics.get("total_screened"):
            metrics["ligands_per_hour"] = round(float(metrics["total_screened"]) / max(runtime_seconds / 3600.0, 1e-9), 3)
        _write_json(Path(str(metrics_path)), metrics)
    return {
        "status": status,
        "run_id": run_id,
        "work_dir": work_dir,
        "output_dir": output_dir,
        "scores_csv": scores_csv,
        "shard_plan": shard_plan,
        "runtime_seconds": runtime_seconds,
        **artifacts,
    }
