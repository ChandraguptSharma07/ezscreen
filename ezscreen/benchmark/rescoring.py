from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ezscreen.benchmark.metrics import compute_metrics
from ezscreen.benchmark.runner import (
    _canonical,
    _finite_number,
    _ligand_descriptors,
    _load_actives,
    ligand_balanced_corrected_score,
    ligand_bias_corrected_score,
    ranking_score,
    run_benchmark,
    score_direction_for_col,
)

_DEFAULT_SCORE_PRIORITY = (
    "score",
    "docking_score",
    "prolif_score",
    "CNNscore",
    "CNNaffinity",
)
_EXCLUDED_SCORE_COLUMNS = {"raw_score", "score_qc"}
_ID_COLUMNS = ("name", "id", "ligand", "chembl_id", "compound_id")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "score"


def _row_ids(row: dict[str, str]) -> set[str]:
    return {str(row.get(key, "")).strip() for key in _ID_COLUMNS if str(row.get(key, "")).strip()}


def _row_label(row: dict[str, str], active_smiles: set[str], active_ids: set[str]) -> int:
    canonical = _canonical(str(row.get("smiles", ""))) if row.get("smiles") else None
    if canonical and canonical in active_smiles:
        return 1
    return 1 if active_ids & _row_ids(row) else 0


def _entry_matches_row(entry: tuple[str | None, set[str]], row: dict[str, str]) -> bool:
    entry_smiles, entry_ids = entry
    canonical = _canonical(str(row.get("smiles", ""))) if row.get("smiles") else None
    return bool((entry_smiles and entry_smiles == canonical) or (entry_ids & _row_ids(row)))


def _candidate_score_columns(headers: list[str], rows: list[dict[str, str]]) -> list[str]:
    columns = [col for col in headers if col not in _EXCLUDED_SCORE_COLUMNS and ("score" in col.lower() or "affinity" in col.lower())]
    numeric = []
    for col in columns:
        if any(_finite_number(row.get(col)) is not None for row in rows):
            numeric.append(col)
    priority = [col for col in _DEFAULT_SCORE_PRIORITY if col in numeric]
    return priority + [col for col in numeric if col not in priority]


def _shard_id(row: dict[str, str]) -> str:
    for key in ("shard", "source_shard", "shard_id", "notebook", "source_notebook"):
        if str(row.get(key, "")).strip():
            return str(row[key]).strip()
    ligand = str(row.get("ligand", ""))
    for pattern in (r"shard[_-]?(\d+)", r"nb[_-]?(\d+)"):
        match = re.search(pattern, ligand, re.IGNORECASE)
        if match:
            return match.group(0)
    return "unknown"


def _top_window(records: list[dict[str, object]], count: int) -> dict[str, object]:
    top = records[:count]
    shard_counts: dict[str, int] = {}
    for record in top:
        shard = str(record.get("shard") or "unknown")
        shard_counts[shard] = shard_counts.get(shard, 0) + 1
    return {
        "count": len(top),
        "active_count": sum(1 for record in top if record.get("label") == 1),
        "decoy_count": sum(1 for record in top if record.get("label") == 0),
        "shard_counts": dict(sorted(shard_counts.items())),
    }


def _top20_rows(records: list[dict[str, object]], score_col: str) -> list[dict[str, object]]:
    rows = []
    for rank, record in enumerate(records[:20], 1):
        desc = record.get("descriptors") if isinstance(record.get("descriptors"), dict) else {}
        rows.append(
            {
                "rank": rank,
                "label": "active" if record.get("label") == 1 else "decoy",
                "ligand": record.get("ligand", ""),
                "name": record.get("name", ""),
                "smiles": record.get("smiles", ""),
                "raw_score": record.get("raw_score", ""),
                score_col: record.get("column_score", ""),
                "shard": record.get("shard", ""),
                "abs_formal_charge": desc.get("abs_formal_charge"),
                "heavy_atoms": desc.get("heavy_atoms"),
                "molecular_weight": desc.get("molecular_weight"),
                "tpsa": desc.get("tpsa"),
                "rotatable_bonds": desc.get("rotatable_bonds"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _property_only_controls(records: list[dict[str, object]]) -> dict[str, object]:
    labels: list[int] = []
    bias_scores: list[float] = []
    balanced_scores: list[float] = []
    for record in records:
        desc = record.get("descriptors") if isinstance(record.get("descriptors"), dict) else {}
        labels.append(int(record.get("label") or 0))
        bias_scores.append(ligand_bias_corrected_score(0.0, desc))
        balanced_scores.append(ligand_balanced_corrected_score(0.0, desc))
    controls = {}
    for name, scores in (
        ("property_only_bias_component", bias_scores),
        ("property_only_balanced_component", balanced_scores),
    ):
        result = compute_metrics(scores, labels)
        controls[name] = {
            "formula": "correction formula with raw score fixed to 0.0",
            "ef1": result.ef1,
            "ef5": result.ef5,
            "auc_roc": result.auc_roc,
        }
    return controls



def join_rescore_scores(
    scores_csv: Path,
    rescored_csv: Path,
    output_csv: Path | None = None,
    *,
    id_cols: tuple[str, ...] = ("ligand", "lig_id", "name"),
    score_cols: list[str] | None = None,
) -> dict[str, object]:
    scores_csv = scores_csv.expanduser().resolve()
    rescored_csv = rescored_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve() if output_csv else scores_csv
    with scores_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    with rescored_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rescore_rows = list(reader)
        rescore_headers = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows in {scores_csv}")
    if not rescore_rows:
        raise ValueError(f"No rows in {rescored_csv}")

    def first_key(row: dict[str, str]) -> str:
        for col in id_cols:
            value = str(row.get(col, "")).strip()
            if value:
                return value
        return ""

    detected_score_cols = [
        col for col in rescore_headers
        if col not in id_cols and col not in _EXCLUDED_SCORE_COLUMNS and any(_finite_number(r.get(col)) is not None for r in rescore_rows)
    ]
    score_cols = score_cols or detected_score_cols
    if not score_cols:
        raise ValueError(f"No numeric rescoring columns found in {rescored_csv}")
    for col in score_cols:
        if col not in headers:
            headers.append(col)

    by_key = {first_key(row): row for row in rescore_rows if first_key(row)}
    matched = 0
    for row in rows:
        entry = by_key.get(first_key(row))
        if not entry:
            continue
        matched += 1
        for col in score_cols:
            row[col] = entry.get(col, row.get(col, ""))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "scores_csv": str(scores_csv),
        "rescored_csv": str(rescored_csv),
        "output_csv": str(output_csv),
        "score_cols": score_cols,
        "matched_rows": matched,
        "unmatched_rows": max(0, len(rows) - matched),
    }
def compare_rescore_columns(
    actives_path: Path,
    scores_csv: Path,
    output_dir: Path | None = None,
    *,
    score_cols: list[str] | None = None,
) -> dict[str, object]:
    actives_path = actives_path.expanduser().resolve()
    scores_csv = scores_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve() if output_dir else scores_csv.parent / "rescore_comparison"

    active_smiles, active_ids, active_entries = _load_actives(actives_path)
    with scores_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows in {scores_csv}")
    score_cols = score_cols or _candidate_score_columns(headers, rows)
    if not score_cols:
        raise ValueError(f"No numeric score/affinity columns found in {scores_csv}")

    all_labeled_records: list[dict[str, object]] = []
    for row in rows:
        raw = _finite_number(row.get("score"))
        label = _row_label(row, active_smiles, active_ids)
        all_labeled_records.append(
            {
                "label": label,
                "descriptors": _ligand_descriptors(str(row.get("smiles", ""))),
                "raw_score": raw,
            }
        )

    metric_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for score_col in score_cols:
        if score_col not in headers:
            raise ValueError(f"Score column `{score_col}` not found in {scores_csv}")
        direction = score_direction_for_col(score_col)
        result = run_benchmark(
            actives_path,
            scores_csv,
            score_col=score_col,
            score_direction=direction,
        )
        records: list[dict[str, object]] = []
        matched_entry_indexes: set[int] = set()
        for row in rows:
            raw_value = _finite_number(row.get(score_col))
            if raw_value is None:
                continue
            label = _row_label(row, active_smiles, active_ids)
            for index, entry in enumerate(active_entries):
                if label and _entry_matches_row(entry, row):
                    matched_entry_indexes.add(index)
            records.append(
                {
                    "rank_score": ranking_score(raw_value, score_col, direction),
                    "column_score": raw_value,
                    "raw_score": row.get("score", ""),
                    "label": label,
                    "ligand": row.get("ligand", ""),
                    "name": row.get("name", ""),
                    "smiles": row.get("smiles", ""),
                    "shard": _shard_id(row),
                    "descriptors": _ligand_descriptors(str(row.get("smiles", ""))),
                }
            )
        records.sort(key=lambda record: float(record["rank_score"]))
        top1 = max(1, int(len(records) * 0.01))
        top5 = max(1, int(len(records) * 0.05))
        active_numeric_missing = max(0, len(active_entries) - len(matched_entry_indexes))
        metric_row = {
            "score_col": score_col,
            "score_direction": direction,
            "ef1": result.ef1,
            "ef5": result.ef5,
            "auc_roc": result.auc_roc,
            "n_actives": result.n_actives,
            "n_decoys": result.n_decoys,
            "total_screened": result.total_screened,
            "first_active_rank": result.label_diagnostics.get("first_active_rank"),
            "top_1_percent_actives": result.label_diagnostics.get("top_1_percent_actives"),
            "top_5_percent_actives": result.label_diagnostics.get("top_5_percent_actives"),
            "numeric_rows": len(records),
            "skipped_rows": result.label_diagnostics.get("skipped_rows"),
            "active_numeric_missing_entries": active_numeric_missing,
            "top_1_percent_shard_counts": json.dumps(_top_window(records, top1)["shard_counts"], sort_keys=True),
            "top_5_percent_shard_counts": json.dumps(_top_window(records, top5)["shard_counts"], sort_keys=True),
        }
        metric_rows.append(metric_row)
        top20 = _top20_rows(records, score_col)
        _write_csv(output_dir / f"top20_{_safe_name(score_col)}.csv", top20)
        summaries[score_col] = {
            "metrics": metric_row,
            "top_windows": {
                "top_1_percent": _top_window(records, top1),
                "top_5_percent": _top_window(records, top5),
                "top_20": _top_window(records, 20),
            },
            "top20_path": str(output_dir / f"top20_{_safe_name(score_col)}.csv"),
        }

    property_controls = _property_only_controls(all_labeled_records)
    _write_csv(output_dir / "rescore_metrics_table.csv", metric_rows)
    summary = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actives_path": str(actives_path),
        "scores_csv": str(scores_csv),
        "score_columns": score_cols,
        "property_only_controls": property_controls,
        "columns": summaries,
        "warnings": [
            "Do not use balanced_corrected_score as headline metric unless property-only controls fail to explain improvement.",
            "CNNscore and CNNaffinity are ranked higher-is-better; energy-like scores are ranked lower-is-better.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rescore_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "metrics_table": output_dir / "rescore_metrics_table.csv",
        "summary_path": output_dir / "rescore_comparison_summary.json",
        "summary": summary,
    }
