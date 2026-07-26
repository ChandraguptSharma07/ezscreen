from __future__ import annotations

import csv
import math
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Callable

from ezscreen.benchmark.metrics import BenchmarkResult, compute_metrics
from ezscreen.benchmark.aggregation import aggregate_source_compounds
from ezscreen.results.variants import source_name


BIAS_CORRECTED_SCORE_COL = "bias_corrected_score"
BIAS_CORRECTION_FORMULA = "score + 8.0 * abs_formal_charge + 0.1 * heavy_atoms - 0.03 * tpsa"
BALANCED_CORRECTED_SCORE_COL = "balanced_corrected_score"
BALANCED_CORRECTION_FORMULA = "score + 25.0 * abs_formal_charge - 0.8 * heavy_atoms - 0.03 * tpsa - 0.4 * rotatable_bonds"
HIGHER_IS_BETTER_SCORE_COLS = {
    "cnnscore",
    "cnnaffinity",
    "diffdock_confidence",
    "oddt_rfscore_v1",
    "oddt_rfscore_v2",
    "oddt_rfscore_v3",
    "rfscorevs_v2",
}


@lru_cache(maxsize=200000)
def _canonical(smiles: str) -> str | None:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)



def _identifier_matches(active_ids: set[str], row_ids: set[str]) -> set[str]:
    if not active_ids or not row_ids:
        return set()
    expanded: set[str] = set()
    for row_id in row_ids:
        if row_id:
            expanded.add(row_id)
            expanded.add(source_name(row_id))
    return active_ids & expanded

def _load_actives(path: Path) -> tuple[set[str], set[str], list[tuple[str | None, set[str]]]]:
    canonical: set[str] = set()
    identifiers: set[str] = set()
    entries: list[tuple[str | None, set[str]]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            entry_ids = {part.strip() for part in parts[1:] if part.strip()}
            c = _canonical(smi)
            if c:
                canonical.add(c)
            identifiers.update(entry_ids)
            entries.append((c, entry_ids))
    return canonical, identifiers, entries


@lru_cache(maxsize=200000)
def _ligand_descriptors(smiles: str) -> dict[str, float | int | None]:
    empty = {
        "formal_charge": None,
        "abs_formal_charge": None,
        "heavy_atoms": None,
        "molecular_weight": None,
        "tpsa": None,
        "hbd": None,
        "hba": None,
        "rotatable_bonds": None,
        "ring_count": None,
    }
    if not smiles:
        return empty
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return empty
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        abs_formal_charge = sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())
        return {
            "formal_charge": formal_charge,
            "abs_formal_charge": abs_formal_charge,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "molecular_weight": Descriptors.MolWt(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "ring_count": rdMolDescriptors.CalcNumRings(mol),
        }
    except Exception:
        return empty


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _autodetect_score_col(headers: list[str]) -> str:
    return next(
        (h for h in headers if "score" in h.lower() or "affinity" in h.lower()),
        headers[-1],
    )


def score_direction_for_col(score_col: str | None) -> str:
    normalized = str(score_col or "").strip().lower()
    if normalized in HIGHER_IS_BETTER_SCORE_COLS:
        return "higher"
    return "lower"


def ranking_score(raw_score: float, score_col: str | None, score_direction: str | None = None) -> float:
    direction = (score_direction or score_direction_for_col(score_col)).strip().lower()
    if direction not in {"lower", "higher"}:
        raise ValueError("score_direction must be `lower` or `higher`")
    return -raw_score if direction == "higher" else raw_score


def _descriptor_value(descriptors: dict, key: str, default: float = 0.0) -> float:
    value = _finite_number(descriptors.get(key))
    return default if value is None else value


def ligand_bias_corrected_score(raw_score: float, descriptors: dict) -> float:
    return (
        raw_score
        + 8.0 * _descriptor_value(descriptors, "abs_formal_charge")
        + 0.1 * _descriptor_value(descriptors, "heavy_atoms")
        - 0.03 * _descriptor_value(descriptors, "tpsa")
    )


def ligand_balanced_corrected_score(raw_score: float, descriptors: dict) -> float:
    return (
        raw_score
        + 25.0 * _descriptor_value(descriptors, "abs_formal_charge")
        - 0.8 * _descriptor_value(descriptors, "heavy_atoms")
        - 0.03 * _descriptor_value(descriptors, "tpsa")
        - 0.4 * _descriptor_value(descriptors, "rotatable_bonds")
    )


def _correction_for_output_col(output_col: str):
    if output_col == BALANCED_CORRECTED_SCORE_COL:
        return ligand_balanced_corrected_score, BALANCED_CORRECTION_FORMULA
    return ligand_bias_corrected_score, BIAS_CORRECTION_FORMULA


def write_bias_corrected_scores(
    scores_csv: Path,
    output_csv: Path | None = None,
    *,
    score_col: str | None = None,
    smiles_col: str = "smiles",
    output_col: str = BIAS_CORRECTED_SCORE_COL,
) -> dict[str, object]:
    scores_csv = scores_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve() if output_csv else scores_csv
    with scores_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows in {scores_csv}")
    if not headers:
        raise ValueError(f"No header in {scores_csv}")
    score_col = score_col or _autodetect_score_col(headers)
    if score_col not in headers:
        raise ValueError(f"Score column `{score_col}` not found in {scores_csv}")
    if output_col not in headers:
        headers.append(output_col)
    correction, formula = _correction_for_output_col(output_col)

    corrected = 0
    skipped = 0
    for row in rows:
        raw = _finite_number(row.get(score_col))
        if raw is None:
            row[output_col] = ""
            skipped += 1
            continue
        descriptors = _ligand_descriptors(str(row.get(smiles_col, "")))
        row[output_col] = f"{correction(raw, descriptors):.6f}"
        corrected += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "scores_csv": str(scores_csv),
        "output_csv": str(output_csv),
        "score_col": score_col,
        "output_col": output_col,
        "formula": formula,
        "corrected_rows": corrected,
        "skipped_rows": skipped,
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    vals = sorted(value for value in values if _finite_number(value) is not None)
    if not vals:
        return {key: None for key in ("min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean")}

    def q(frac: float) -> float:
        return vals[min(len(vals) - 1, int(frac * (len(vals) - 1)))]

    return {
        "min": round(vals[0], 4),
        "p01": round(q(0.01), 4),
        "p05": round(q(0.05), 4),
        "p10": round(q(0.10), 4),
        "p25": round(q(0.25), 4),
        "median": round(q(0.50), 4),
        "p75": round(q(0.75), 4),
        "p90": round(q(0.90), 4),
        "p95": round(q(0.95), 4),
        "p99": round(q(0.99), 4),
        "max": round(vals[-1], 4),
        "mean": round(statistics.mean(vals), 4),
    }


def _median_descriptor(records: list[dict], key: str) -> float | None:
    vals = [_finite_number((record.get("descriptors") or {}).get(key)) for record in records]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return round(statistics.median(vals), 4)


def _window_diagnostics(ranked_records: list[dict], count: int) -> dict[str, object]:
    top = ranked_records[:count]
    if not top:
        return {"count": 0, "active_count": 0, "decoy_count": 0, "descriptor_medians": {}}
    charged = [record for record in top if _finite_number((record.get("descriptors") or {}).get("abs_formal_charge")) and float((record.get("descriptors") or {}).get("abs_formal_charge") or 0) > 0]
    return {
        "count": len(top),
        "active_count": sum(1 for record in top if record.get("label") == 1),
        "decoy_count": sum(1 for record in top if record.get("label") == 0),
        "charged_fraction": round(len(charged) / len(top), 4),
        "descriptor_medians": {
            "abs_formal_charge": _median_descriptor(top, "abs_formal_charge"),
            "heavy_atoms": _median_descriptor(top, "heavy_atoms"),
            "molecular_weight": _median_descriptor(top, "molecular_weight"),
            "tpsa": _median_descriptor(top, "tpsa"),
        },
    }


def _alternative_ranking(
    records: list[dict],
    *,
    formula: str,
    transform: Callable[[dict], float],
) -> dict[str, object]:
    adjusted_scores: list[float] = []
    labels: list[int] = []
    for record in records:
        adjusted_scores.append(transform(record))
        labels.append(int(record.get("label") or 0))
    result = compute_metrics(adjusted_scores, labels)
    ranked = sorted(zip(adjusted_scores, labels), key=lambda item: item[0])
    return {
        "formula": formula,
        "ef1": result.ef1,
        "ef5": result.ef5,
        "auc_roc": result.auc_roc,
        "first_active_rank": next((idx + 1 for idx, (_, label) in enumerate(ranked) if label == 1), None),
        "top_1_percent_actives": sum(label for _, label in ranked[: max(1, int(len(ranked) * 0.01))]),
        "top_5_percent_actives": sum(label for _, label in ranked[: max(1, int(len(ranked) * 0.05))]),
    }


def _ranking_diagnostics(records: list[dict]) -> dict[str, object]:
    ranked_records = sorted(records, key=lambda record: float(record["score"]))
    active_records = [record for record in ranked_records if record.get("label") == 1]
    decoy_records = [record for record in ranked_records if record.get("label") == 0]
    total = len(ranked_records)
    top1 = max(1, int(total * 0.01))
    top5 = max(1, int(total * 0.05))

    top_decoys = []
    for rank, record in enumerate(ranked_records, 1):
        if record.get("label") != 0:
            continue
        desc = record.get("descriptors") or {}
        top_decoys.append(
            {
                "rank": rank,
                "name": record.get("name", ""),
                "ligand": record.get("ligand", ""),
                "score": round(float(record["score"]), 4),
                "score_qc": record.get("score_qc", ""),
                "formal_charge": desc.get("formal_charge"),
                "abs_formal_charge": desc.get("abs_formal_charge"),
                "heavy_atoms": desc.get("heavy_atoms"),
                "molecular_weight": round(float(desc["molecular_weight"]), 4) if _finite_number(desc.get("molecular_weight")) is not None else None,
                "tpsa": round(float(desc["tpsa"]), 4) if _finite_number(desc.get("tpsa")) is not None else None,
                "smiles": record.get("smiles", ""),
            }
        )
        if len(top_decoys) >= 10:
            break

    def descriptor_split(key: str) -> dict[str, object]:
        return {
            "actives": _quantiles([float(value) for value in (_finite_number((record.get("descriptors") or {}).get(key)) for record in active_records) if value is not None]),
            "decoys": _quantiles([float(value) for value in (_finite_number((record.get("descriptors") or {}).get(key)) for record in decoy_records) if value is not None]),
            "top_1_percent": _quantiles([float(value) for value in (_finite_number((record.get("descriptors") or {}).get(key)) for record in ranked_records[:top1]) if value is not None]),
        }

    def desc(record: dict, key: str, default: float = 0.0) -> float:
        value = _finite_number((record.get("descriptors") or {}).get(key))
        return default if value is None else value

    alternative_rankings = {
        "ligand_efficiency_score_per_heavy_atom": _alternative_ranking(
            ranked_records,
            formula="score / max(1, heavy_atoms)",
            transform=lambda record: float(record["score"]) / max(1.0, desc(record, "heavy_atoms", 1.0)),
        ),
        "charge_penalty_score_plus_2_abs_charge": _alternative_ranking(
            ranked_records,
            formula="score + 2.0 * abs_formal_charge",
            transform=lambda record: float(record["score"]) + 2.0 * desc(record, "abs_formal_charge"),
        ),
        "charge_size_penalty_score_plus_4_abs_charge_plus_0_3_heavy_atoms": _alternative_ranking(
            ranked_records,
            formula="score + 4.0 * abs_formal_charge + 0.3 * heavy_atoms",
            transform=lambda record: float(record["score"]) + 4.0 * desc(record, "abs_formal_charge") + 0.3 * desc(record, "heavy_atoms"),
        ),
        "bias_corrected_score_plus_8_abs_charge_plus_0_1_heavy_atoms_minus_0_03_tpsa": _alternative_ranking(
            ranked_records,
            formula=BIAS_CORRECTION_FORMULA,
            transform=lambda record: ligand_bias_corrected_score(float(record["score"]), record.get("descriptors") or {}),
        ),
        "balanced_corrected_score_plus_25_abs_charge_minus_0_8_heavy_atoms_minus_0_03_tpsa_minus_0_4_rotatable_bonds": _alternative_ranking(
            ranked_records,
            formula=BALANCED_CORRECTION_FORMULA,
            transform=lambda record: ligand_balanced_corrected_score(float(record["score"]), record.get("descriptors") or {}),
        ),
    }

    top1_charged_fraction = _window_diagnostics(ranked_records, top1).get("charged_fraction")
    all_charged = [record for record in ranked_records if desc(record, "abs_formal_charge") > 0]
    all_charged_fraction = round(len(all_charged) / total, 4) if total else 0.0
    flags = []
    if isinstance(top1_charged_fraction, float) and top1_charged_fraction > max(0.5, all_charged_fraction * 1.5):
        flags.append("top_1_percent_enriched_for_formally_charged_ligands")
    if top_decoys and active_records and float(top_decoys[0]["score"]) < float(active_records[0]["score"]):
        flags.append("best_scoring_ligand_is_decoy")

    return {
        "official_ranking_note": "Official EF/AUC use the raw score column; alternative rankings below are diagnostics only and do not replace benchmark metrics.",
        "flags": flags,
        "score_quantiles": {
            "actives": _quantiles([float(record["score"]) for record in active_records]),
            "decoys": _quantiles([float(record["score"]) for record in decoy_records]),
        },
        "descriptor_quantiles": {
            "abs_formal_charge": descriptor_split("abs_formal_charge"),
            "heavy_atoms": descriptor_split("heavy_atoms"),
            "molecular_weight": descriptor_split("molecular_weight"),
            "tpsa": descriptor_split("tpsa"),
        },
        "top_windows": {
            "top_1_percent": _window_diagnostics(ranked_records, top1),
            "top_5_percent": _window_diagnostics(ranked_records, top5),
        },
        "top_decoy_examples": top_decoys,
        "alternative_rankings": alternative_rankings,
    }


def _pose_ids_from_sdf(poses_sdf: Path) -> set[str]:
    pose_ids: set[str] = set()
    if not poses_sdf.exists() or poses_sdf.stat().st_size == 0:
        return pose_ids
    try:
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False, sanitize=False)
        for index, mol in enumerate(supplier):
            if mol is None:
                continue
            for prop in ("lig_id", "name", "id", "_Name"):
                if mol.HasProp(prop):
                    value = mol.GetProp(prop).strip()
                    if value:
                        pose_ids.add(value)
            if mol.HasProp("_Name"):
                value = mol.GetProp("_Name").strip()
                if value:
                    pose_ids.add(value)
            pose_ids.add(f"pose_{index + 1}")
    except Exception:
        return set()
    return pose_ids


def _pose_molecule_count(poses_sdf: Path) -> int:
    if not poses_sdf.exists() or poses_sdf.stat().st_size == 0:
        return 0
    try:
        from rdkit import Chem

        return sum(1 for mol in Chem.SDMolSupplier(str(poses_sdf), removeHs=False, sanitize=False) if mol is not None)
    except Exception:
        return 0


def pose_retention_diagnostics(
    actives_path: Path,
    scores_csv: Path,
    *,
    score_col: str | None = None,
    smiles_col: str = "smiles",
    poses_sdf: Path | None = None,
    top_n: int = 20,
) -> dict[str, object]:
    """Report whether benchmark top-window rows have retained poses.

    EF/AUC remain score-only metrics. This diagnostic marks when top-ranked
    ligands, score-floor rows, or actives cannot be inspected because poses were
    not archived.
    """
    actives_path = actives_path.expanduser().resolve()
    scores_csv = scores_csv.expanduser().resolve()
    poses_sdf = poses_sdf.expanduser().resolve() if poses_sdf else scores_csv.parent / "poses.sdf"

    active_smiles, active_ids, _active_entries = _load_actives(actives_path)
    with scores_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"status": "no_scores", "flags": ["pose_retention_no_scores"]}

    headers = list(rows[0].keys())
    score_col = score_col or _autodetect_score_col(headers)
    score_direction = score_direction_for_col(score_col)
    pose_ids = _pose_ids_from_sdf(poses_sdf)
    pose_molecule_count = _pose_molecule_count(poses_sdf)

    records: list[dict[str, object]] = []
    for row in rows:
        try:
            score = ranking_score(float(row.get(score_col, "")), score_col, score_direction)
        except (TypeError, ValueError):
            continue
        smi = row.get(smiles_col, "")
        canonical = _canonical(smi) if smi else None
        row_ids = {
            str(row.get(key, "")).strip()
            for key in ("name", "id", "ligand", "chembl_id", "compound_id")
            if str(row.get(key, "")).strip()
        }
        label = 1 if ((canonical and canonical in active_smiles) or bool(_identifier_matches(active_ids, row_ids))) else 0
        has_pose = bool(row_ids & pose_ids)
        records.append(
            {
                "score": score,
                "label": label,
                "name": str(row.get("name", "")),
                "ligand": str(row.get("ligand", "")),
                "score_qc": str(row.get("score_qc", "")),
                "has_pose": has_pose,
            }
        )

    ranked = sorted(records, key=lambda record: float(record["score"]))
    top = ranked[: max(1, min(top_n, len(ranked)))]
    score_floor = [record for record in ranked if "score_floor" in str(record.get("score_qc", ""))]
    score_ceiling = [record for record in ranked if "score_ceiling" in str(record.get("score_qc", ""))]
    active_records = [record for record in ranked if record.get("label") == 1]
    retained_active = [record for record in active_records if record.get("has_pose")]

    def count_pose(items: list[dict[str, object]]) -> int:
        return sum(1 for item in items if item.get("has_pose"))

    flags: list[str] = []
    if not poses_sdf.exists() or not pose_ids:
        flags.append("poses_sdf_missing_or_empty")
    if top and count_pose(top) < len(top):
        flags.append("top_window_missing_retained_poses")
    if score_floor and count_pose(score_floor) < len(score_floor):
        flags.append("score_floor_rows_missing_retained_poses")
    if score_ceiling:
        flags.append("score_ceiling_rows_present")
    if active_records and not retained_active:
        flags.append("no_retained_active_poses")
    if len(ranked) >= 1000 and pose_molecule_count and pose_molecule_count / max(1, len(ranked)) < 0.01:
        flags.append("low_pose_retention_fraction")

    def examples(items: list[dict[str, object]], limit: int = 10) -> list[dict[str, object]]:
        out = []
        for rank, item in enumerate(ranked, 1):
            if item not in items:
                continue
            out.append(
                {
                    "rank": rank,
                    "name": item.get("name", ""),
                    "ligand": item.get("ligand", ""),
                    "score": round(float(item["score"]), 4),
                    "score_qc": item.get("score_qc", ""),
                    "has_pose": bool(item.get("has_pose")),
                    "label": "active" if item.get("label") == 1 else "decoy",
                }
            )
            if len(out) >= limit:
                break
        return out

    missing_top = [record for record in top if not record.get("has_pose")]
    return {
        "status": "complete",
        "poses_sdf": str(poses_sdf),
        "score_rows": len(ranked),
        "retained_pose_count": pose_molecule_count,
        "retained_pose_identifier_count": len(pose_ids),
        "retained_pose_fraction": round(pose_molecule_count / max(1, len(ranked)), 6),
        "top_window_size": len(top),
        "top_window_pose_count": count_pose(top),
        "top_window_missing_pose_count": len(missing_top),
        "active_rows": len(active_records),
        "retained_active_pose_count": len(retained_active),
        "score_floor_rows": len(score_floor),
        "score_floor_pose_count": count_pose(score_floor),
        "score_ceiling_rows": len(score_ceiling),
        "score_ceiling_pose_count": count_pose(score_ceiling),
        "flags": flags,
        "missing_top_window_examples": examples(missing_top),
        "score_floor_examples": examples(score_floor),
        "score_ceiling_examples": examples(score_ceiling),
    }


def _load_intended_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8') as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts or line.lstrip().startswith('#'):
                continue
            rows.append(
                {
                    'smiles': parts[0],
                    'name': parts[1] if len(parts) > 1 else '',
                }
            )
    return rows


def run_benchmark(
    actives_path: Path,
    scores_csv: Path,
    smiles_col: str = "smiles",
    score_col: str | None = None,
    score_direction: str | None = None,
    intended_compounds_path: Path | None = None,
) -> BenchmarkResult:
    active_smiles, active_ids, active_entries = _load_actives(actives_path)

    rows: list[dict] = []
    with scores_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {scores_csv}")

    headers = list(rows[0].keys())

    # auto-detect score column if not given
    if score_col is None:
        score_col = _autodetect_score_col(headers)
    score_direction = score_direction or score_direction_for_col(score_col)

    metric_rows: list[dict] = []
    for source_row in rows:
        row = dict(source_row)
        raw_score = _finite_number(row.get(score_col))
        qc_text = ' '.join(
            str(row.get(key, '')).lower()
            for key in ('score_qc', 'status', 'reason', 'stage')
        )
        filtered = 'filter' in qc_text
        if raw_score is not None:
            row['_ranking_score'] = ranking_score(
                raw_score, score_col, score_direction
            )
        row['_filtered'] = filtered
        row['_scored'] = raw_score is not None and not filtered
        metric_rows.append(row)

    unscored_path = scores_csv.parent / 'unscored_reasons.csv'
    if unscored_path.exists():
        with unscored_path.open(newline='', encoding='utf-8') as handle:
            for source_row in csv.DictReader(handle):
                row = dict(source_row)
                row['_scored'] = False
                row['_filtered'] = 'filter' in ' '.join(
                    str(row.get(key, '')).lower()
                    for key in ('score_qc', 'status', 'reason', 'stage')
                )
                metric_rows.append(row)

    for entry_index, (entry_smiles, entry_ids) in enumerate(active_entries):
        entry_found = False
        for row in metric_rows:
            row_ids = {
                str(row.get(key, '')).strip()
                for key in (
                    'name',
                    'id',
                    'ligand',
                    'chembl_id',
                    'compound_id',
                    'source_compound_id',
                )
                if str(row.get(key, '')).strip()
            }
            row_smiles = str(row.get(smiles_col, '')).strip()
            canonical = _canonical(row_smiles) if row_smiles else None
            if (
                (entry_smiles and entry_smiles == canonical)
                or bool(_identifier_matches(entry_ids, row_ids))
            ):
                entry_found = True
                break
        if not entry_found:
            metric_rows.append(
                {
                    'name': sorted(entry_ids)[0] if entry_ids else f'active_{entry_index}',
                    smiles_col: entry_smiles or '',
                    'reason': 'prep_or_output_missing',
                    '_scored': False,
                    '_filtered': False,
                }
            )

    postprep_rows = [
        row
        for row in metric_rows
        if 'prep' not in ' '.join(
            str(row.get(key, '')).lower()
            for key in ('score_qc', 'status', 'reason', 'stage')
        )
    ]
    postprep_count = aggregate_source_compounds(
        postprep_rows
    ).counts['after_aggregation']

    intended_rows: list[dict] = []
    if intended_compounds_path:
        intended_rows = _load_intended_rows(intended_compounds_path)
        for source_row in intended_rows:
            row = dict(source_row)
            row['_scored'] = False
            row['_filtered'] = False
            metric_rows.append(row)

    aggregation = aggregate_source_compounds(metric_rows)
    rows = aggregation.records
    source_score_direction = score_direction
    for row in rows:
        row[score_col] = row['_ranking_score']
    metric_score_direction = 'lower'
    intended_count = (
        aggregate_source_compounds(
            [
                {**row, '_scored': False, '_filtered': False}
                for row in intended_rows
            ]
        ).counts['after_aggregation']
        if intended_rows
        else aggregation.counts['after_aggregation']
    )

    scores: list[float] = []
    labels: list[int] = []
    scored_flags: list[bool] = []
    diagnostics = {
        'counts_before_prep': intended_count,
        'counts_after_prep': postprep_count,
        'counts_after_aggregation': aggregation.counts['after_aggregation'],
        'counts_scored': aggregation.counts['scored'],
        'counts_failed': aggregation.counts['failed'],
        'counts_filtered': aggregation.counts['filtered'],
        'counts_unmatched': aggregation.counts['unmatched'],
        'variant_rows_collapsed': aggregation.counts['variant_rows_collapsed'],
        'intended_compounds_path': (
            str(intended_compounds_path) if intended_compounds_path else ''
        ),
        'source_score_direction': source_score_direction,
        "actives_file_smiles_count": len(active_smiles),
        "actives_file_identifier_count": len(active_ids),
        "actives_file_entry_count": len(active_entries),
        "score_column": score_col,
        "score_direction": source_score_direction,
        "smiles_column": smiles_col,
        "scored_rows": 0,
        "skipped_rows": 0,
        "matched_active_rows": 0,
        "matched_by_identifier": 0,
        "matched_by_smiles_only": 0,
        "matched_by_both": 0,
        "matched_active_identifiers": 0,
        "unmatched_active_identifiers": len(active_ids),
        "matched_active_entries": 0,
        "unmatched_active_entries": len(active_entries),
        "first_active_rank": None,
        "top_1_percent_actives": 0,
        "top_5_percent_actives": 0,
    }
    matched_ids: set[str] = set()
    matched_entry_indexes: set[int] = set()
    ranked_pairs: list[tuple[float, int]] = []
    ranked_records: list[dict] = []

    for row in rows:
        raw_score = row.get(score_col, "")
        try:
            score = ranking_score(
                float(raw_score), score_col, metric_score_direction
            )
        except (ValueError, TypeError):
            diagnostics["skipped_rows"] += 1
            continue

        smi = row.get(smiles_col, "")
        canonical = _canonical(smi) if smi else None
        row_ids = {
            str(row.get(key, "")).strip()
            for key in ("name", "id", "ligand", "chembl_id", "compound_id")
            if str(row.get(key, "")).strip()
        }
        row_ids.update(
            str(row.get(key, '')).strip()
            for key in ('source_compound_id', 'source_id', 'parent_id', 'original_id')
            if str(row.get(key, '')).strip()
        )
        id_matches = _identifier_matches(active_ids, row_ids)
        smiles_match = bool(canonical and canonical in active_smiles)
        id_match = bool(id_matches)
        label = 1 if (smiles_match or id_match) else 0

        if label:
            diagnostics["matched_active_rows"] += 1
            if id_match:
                diagnostics["matched_by_identifier"] += 1
            if smiles_match and not id_match:
                diagnostics["matched_by_smiles_only"] += 1
            if id_match and smiles_match:
                diagnostics["matched_by_both"] += 1
            matched_ids.update(id_matches)
            for index, (entry_smiles, entry_ids) in enumerate(active_entries):
                if (entry_smiles and entry_smiles == canonical) or bool(_identifier_matches(entry_ids, row_ids)):
                    matched_entry_indexes.add(index)

        scores.append(score)
        labels.append(label)
        scored_flags.append(bool(row.get('_scored')))
        ranked_pairs.append((score, label))
        ranked_records.append(
            {
                "score": score,
                "label": label,
                "name": str(row.get("name", "")),
                "ligand": str(row.get("ligand", "")),
                "smiles": smi,
                "score_qc": str(row.get("score_qc", "")),
                "descriptors": _ligand_descriptors(smi),
            }
        )
        diagnostics["scored_rows"] += 1

    n_matched = sum(labels)
    if n_matched == 0:
        raise ValueError(
            f"None of the {len(active_smiles)} known active SMILES or {len(active_ids)} active IDs matched anything in {scores_csv.name}. "
            "Check that the SMILES or identifiers in your actives file appear in the docking results."
        )

    ranked_labels = [label for _, label in sorted(ranked_pairs, key=lambda item: item[0])]
    if 1 in ranked_labels:
        diagnostics["first_active_rank"] = ranked_labels.index(1) + 1
    top1 = max(1, int(len(ranked_labels) * 0.01))
    top5 = max(1, int(len(ranked_labels) * 0.05))
    diagnostics["top_1_percent_actives"] = sum(ranked_labels[:top1])
    diagnostics["top_5_percent_actives"] = sum(ranked_labels[:top5])
    diagnostics["matched_active_identifiers"] = len(matched_ids)
    diagnostics["unmatched_active_identifiers"] = max(0, len(active_ids) - len(matched_ids))
    diagnostics["matched_active_entries"] = len(matched_entry_indexes)
    diagnostics["unmatched_active_entries"] = max(0, len(active_entries) - len(matched_entry_indexes))

    diagnostics['scored_rows'] = aggregation.counts['scored']
    diagnostics['failed_rows'] = aggregation.counts['failed']
    result = compute_metrics(scores, labels, scored=scored_flags)
    diagnostics['first_active_rank'] = result.first_active_rank
    diagnostics['coverage'] = result.coverage
    result.label_diagnostics = diagnostics
    result.ranking_diagnostics = _ranking_diagnostics(ranked_records)
    return result
