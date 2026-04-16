from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _load_index(shard_dirs: list[Path]) -> dict[str, dict]:
    """Collect ligand identity (name, smiles) from index.csv files next to shards."""
    index: dict[str, dict] = {}
    for d in shard_dirs:
        idx_path = d / "index.csv"
        if not idx_path.exists():
            continue
        with idx_path.open() as f:
            for row in csv.DictReader(f):
                index[row["ligand"]] = row
    return index


def _add_efficiency_cols(
    rows: list[dict],
    fieldnames: list[str],
    score_col: str | None,
) -> tuple[list[dict], list[str]]:
    if not score_col:
        return rows, fieldnames
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import MolWt
        from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
    except ImportError:
        return rows, fieldnames

    new_fields = list(fieldnames)
    if "LE" not in new_fields:
        new_fields += ["LE", "BEI"]

    for row in rows:
        smi = row.get("smiles", "")
        try:
            score = abs(float(row.get(score_col, 0)))
            mol   = Chem.MolFromSmiles(smi) if smi else None
            if mol and score > 0:
                ha  = CalcNumHeavyAtoms(mol)
                mw  = MolWt(mol)
                row["LE"]  = f"{score / ha:.3f}"  if ha  else ""
                row["BEI"] = f"{score / mw * 1000:.3f}" if mw else ""
            else:
                row["LE"] = row["BEI"] = ""
        except Exception:
            row["LE"] = row["BEI"] = ""

    return rows, new_fields


def merge_shard_results(shard_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    fieldnames: list[str] = []
    index = _load_index(shard_dirs)

    for d in shard_dirs:
        sf = d / "scores.csv"
        if not sf.exists():
            continue
        with sf.open() as f:
            reader = csv.DictReader(f)
            if not fieldnames and reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            all_rows.extend(reader)

    # Attach name/smiles from index if not already in scores.csv
    if index:
        for row in all_rows:
            lig_id = row.get("ligand", "")
            if lig_id in index and "smiles" not in row:
                row["name"]   = index[lig_id]["name"]
                row["smiles"] = index[lig_id]["smiles"]
        if all_rows and "smiles" in all_rows[0] and "name" not in fieldnames:
            fieldnames += ["name", "smiles"]

    id_col = fieldnames[0] if fieldnames else None
    score_col = next(
        (h for h in fieldnames if "score" in h.lower() or "affinity" in h.lower()),
        fieldnames[-1] if fieldnames else None,
    )

    def _score(row: dict) -> float:
        try:
            return float(row.get(score_col, 0)) if score_col else 0.0
        except (ValueError, TypeError):
            return 0.0

    _SCORE_FLOOR = -15.0
    all_rows = [r for r in all_rows if _score(r) >= _SCORE_FLOOR]

    best: dict[str, dict] = {}
    for row in all_rows:
        key = row.get(id_col, "") if id_col else str(id(row))
        if key not in best or _score(row) < _score(best[key]):
            best[key] = row

    deduped = sorted(best.values(), key=_score)
    deduped, fieldnames = _add_efficiency_cols(deduped, fieldnames, score_col)

    scores_out = output_dir / "scores.csv"
    if deduped and fieldnames:
        with scores_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(deduped)

    poses_out = output_dir / "poses.sdf"
    with poses_out.open("w") as f:
        for d in shard_dirs:
            pf = d / "poses.sdf"
            if pf.exists():
                content = pf.read_text()
                f.write(content)
                if content and not content.endswith("\n"):
                    f.write("\n")

    failed_out = output_dir / "failed_prep.sdf"
    has_failures = False
    with failed_out.open("w") as f:
        for d in shard_dirs:
            ff = d / "failed_prep.sdf"
            if ff.exists() and ff.stat().st_size > 0:
                f.write(ff.read_text())
                has_failures = True

    return {
        "scores_csv": scores_out,
        "poses_sdf": poses_out,
        "failed_prep_sdf": failed_out if has_failures else None,
        "total_hits": len(deduped),
        "score_col": score_col,
    }
