from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _load_filtered(shard_dirs: list[Path]) -> list[dict]:
    """Return molecules filtered out during prep (too large for GPU), deduped by SMILES."""
    rows: list[dict] = []
    seen: set[str] = set()
    for d in shard_dirs:
        fp = d / "filtered_gpu_size.csv"
        if not fp.exists():
            continue
        with fp.open() as f:
            for row in csv.DictReader(f):
                key = row.get("smiles") or row.get("name") or ""
                if key and key not in seen:
                    rows.append(row)
                    seen.add(key)
    return rows


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


def _find_receptor_pdb(output_dir: Path) -> Path | None:
    """Locate the prepared receptor PDB for a run, checking the known locations."""
    work_dir = output_dir.parent
    resume = work_dir / "resume.json"
    if resume.exists():
        try:
            import json
            info = json.loads(resume.read_text())
            p = info.get("receptor_pdb")
            if p and Path(p).exists():
                return Path(p)
        except Exception:
            pass
    for cand in (
        work_dir / "receptor" / "receptor_prep.pdb",
        output_dir / "receptor_prep.pdb",
    ):
        if cand.exists():
            return cand
    return None


def _add_pose_validity_cols(
    rows: list[dict],
    fieldnames: list[str],
    poses_sdf: Path,
    output_dir: Path,
    id_col: str | None,
) -> tuple[list[dict], list[str]]:
    """Tag each row with PoseBusters validity. No-op if posebusters is unavailable."""
    if not id_col:
        return rows, fieldnames
    try:
        from ezscreen.results.pose_validity import check_poses

        receptor_pdb = _find_receptor_pdb(output_dir)
        if receptor_pdb is None:
            return rows, fieldnames
        validity = check_poses(poses_sdf, receptor_pdb)
        if not validity:
            return rows, fieldnames

        new_fields = list(fieldnames)
        if "pb_valid" not in new_fields:
            new_fields += ["pb_valid", "pb_failed"]
        for row in rows:
            entry = validity.get(row.get(id_col, ""))
            if entry is None:
                row["pb_valid"] = ""
                row["pb_failed"] = ""
            else:
                row["pb_valid"] = entry["passed"]
                row["pb_failed"] = ";".join(entry["failed_checks"])
        return rows, new_fields
    except Exception:
        return rows, fieldnames


def _load_failed_docking(shard_dirs: list[Path]) -> list[dict]:
    """Return per-ligand docking failure rows from failed_docking.csv files."""
    rows: list[dict] = []
    for d in shard_dirs:
        fp = d / "failed_docking.csv"
        if not fp.exists():
            continue
        with fp.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def _write_unscored_reasons(
    output_dir: Path,
    index: dict[str, dict],
    scored_ids: set[str],
    score_filtered: list[dict],
    gpu_filtered: list[dict],
    failed_docking: list[dict],
    id_col: str | None,
    score_col: str | None,
) -> Path:
    reasons: list[dict] = []

    # Compounds that reached merger but were cut by floor/ceiling.
    for row in score_filtered:
        lig_id = row.get(id_col, "") if id_col else ""
        info = index.get(lig_id, {})
        reasons.append({
            "ligand":    lig_id,
            "name":      info.get("name", row.get("name", "")),
            "smiles":    info.get("smiles", row.get("smiles", "")),
            "reason":    row.get("_reason", "score_filtered"),
            "raw_score": row.get(score_col, "") if score_col else "",
        })

    # Compounds the Kaggle notebook rejected (score_ceiling, no_remark, etc.)
    # These IDs are in index but never reached merger's scores.csv.
    docking_failed_ids: set[str] = set()
    for row in failed_docking:
        lig_id = row.get("ligand", "")
        docking_failed_ids.add(lig_id)
        info = index.get(lig_id, {})
        reasons.append({
            "ligand":    lig_id,
            "name":      info.get("name", ""),
            "smiles":    info.get("smiles", ""),
            "reason":    row.get("reason", "docking_failed"),
            "raw_score": row.get("raw_score", ""),
        })

    # Compounds that have no output PDBQT at all (UniDock never produced a pose).
    all_accounted = scored_ids | docking_failed_ids
    for lig_id, info in index.items():
        if lig_id not in all_accounted:
            reasons.append({
                "ligand":    lig_id,
                "name":      info.get("name", ""),
                "smiles":    info.get("smiles", ""),
                "reason":    "no_pose",
                "raw_score": "",
            })

    for fmol in gpu_filtered:
        reasons.append({
            "ligand":    "",
            "name":      fmol.get("name", ""),
            "smiles":    fmol.get("smiles", ""),
            "reason":    "gpu_size_filter",
            "raw_score": "",
        })

    out = output_dir / "unscored_reasons.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ligand", "name", "smiles", "reason", "raw_score"])
        w.writeheader()
        w.writerows(reasons)
    return out


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

    # Track every ligand ID that appeared in any scores.csv (before floor/ceiling filter).
    # Used to identify compounds that simply got no docking pose from UniDock.
    all_scored_ids: set[str] = set()

    # Attach name/smiles from index if not already in scores.csv
    if index:
        for row in all_rows:
            lig_id = row.get("ligand", "")
            all_scored_ids.add(lig_id)
            if lig_id in index:
                if "smiles" not in row:
                    row["name"]   = index[lig_id]["name"]
                    row["smiles"] = index[lig_id]["smiles"]
                row["conformer_qc"] = index[lig_id].get("conformer_qc", "")
        if all_rows and "smiles" in all_rows[0] and "name" not in fieldnames:
            fieldnames += ["name", "smiles"]
        if any(r.get("conformer_qc") for r in all_rows) and "conformer_qc" not in fieldnames:
            fieldnames += ["conformer_qc"]
    else:
        for row in all_rows:
            all_scored_ids.add(row.get("ligand", ""))

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

    try:
        from ezscreen import config as _cfg
        _lc           = _cfg.load().get("local", {})
        _enable_floor = bool(_lc.get("enable_score_floor", True))
        _score_floor  = float(_lc.get("score_floor", -15.0))
        _score_ceil   = float(_lc.get("score_ceiling", 0.0))
    except Exception:
        _enable_floor, _score_floor, _score_ceil = True, -15.0, 0.0

    score_filtered: list[dict] = []
    if _enable_floor:
        # Vina-family scores are always negative for real binding events.
        # Scores below the floor are GPU underflow artifacts; scores at or above
        # the ceiling (≥ 0) are GPU overflow artifacts (e.g. 3.776e+06 kcal/mol).
        valid: list[dict] = []
        for r in all_rows:
            s = _score(r)
            if s < _score_floor:
                r["_reason"] = "score_floor"
                score_filtered.append(r)
            elif s >= _score_ceil:
                r["_reason"] = "score_ceiling"
                score_filtered.append(r)
            else:
                valid.append(r)
        all_rows = valid

    best: dict[str, dict] = {}
    for row in all_rows:
        key = row.get(id_col, "") if id_col else str(id(row))
        if key not in best or _score(row) < _score(best[key]):
            best[key] = row

    deduped = sorted(best.values(), key=_score)

    # Append GPU-filtered molecules with score=0 so they appear in results
    # ranked last rather than being excluded from AUC/EF calculations entirely.
    gpu_filtered = _load_filtered(shard_dirs)
    if gpu_filtered:
        if not fieldnames:
            fieldnames = ["ligand", "score", "name", "smiles"]
        if "name" not in fieldnames and any(r.get("name") for r in gpu_filtered):
            fieldnames += ["name", "smiles"]
        for i, fmol in enumerate(gpu_filtered):
            deduped.append({
                "ligand": f"gpu_filt_{i:05d}",
                "score":  0.0,
                "name":   fmol.get("name", ""),
                "smiles": fmol.get("smiles", ""),
            })

    deduped, fieldnames = _add_efficiency_cols(deduped, fieldnames, score_col)

    poses_out = output_dir / "poses.sdf"
    with poses_out.open("w") as f:
        for d in shard_dirs:
            pf = d / "poses.sdf"
            if pf.exists():
                content = pf.read_text()
                f.write(content)
                if content and not content.endswith("\n"):
                    f.write("\n")

    deduped, fieldnames = _add_pose_validity_cols(
        deduped, fieldnames, poses_out, output_dir, id_col,
    )

    scores_out = output_dir / "scores.csv"
    if deduped and fieldnames:
        with scores_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(deduped)

    failed_out = output_dir / "failed_prep.sdf"
    has_failures = False
    with failed_out.open("w") as f:
        for d in shard_dirs:
            ff = d / "failed_prep.sdf"
            if ff.exists() and ff.stat().st_size > 0:
                f.write(ff.read_text())
                has_failures = True

    failed_docking = _load_failed_docking(shard_dirs)
    docking_failed_ids = {r.get("ligand", "") for r in failed_docking}

    reasons_out = _write_unscored_reasons(
        output_dir, index, all_scored_ids, score_filtered, gpu_filtered, failed_docking, id_col, score_col,
    )

    reason_counts: dict[str, int] = {}
    for r in score_filtered:
        k = r.get("_reason", "score_filtered")
        reason_counts[k] = reason_counts.get(k, 0) + 1
    for r in failed_docking:
        k = r.get("reason", "docking_failed")
        reason_counts[k] = reason_counts.get(k, 0) + 1
    all_accounted = all_scored_ids | docking_failed_ids
    reason_counts["no_pose"] = sum(1 for lid in index if lid not in all_accounted)
    reason_counts["gpu_size_filter"] = len(gpu_filtered)

    # Record the native score type so the viewer/report can label the scale.
    # Default is Vina kcal/mol; the engine selector (Phase 32) will set this per engine.
    import json

    from ezscreen.results.score_types import DEFAULT_SCORE_TYPE
    (output_dir / "results_meta.json").write_text(
        json.dumps({"score_type": DEFAULT_SCORE_TYPE, "score_col": score_col}, indent=2)
    )

    return {
        "scores_csv": scores_out,
        "poses_sdf": poses_out,
        "failed_prep_sdf": failed_out if has_failures else None,
        "total_hits": len(deduped),
        "score_col": score_col,
        "score_type": DEFAULT_SCORE_TYPE,
        "unscored_reasons_csv": reasons_out,
        "unscored_reason_counts": reason_counts,
    }
