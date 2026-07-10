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




def _infer_engine(shard_dirs: list[Path], rows: list[dict], explicit: str | None) -> str:
    if explicit:
        return explicit
    for row in rows:
        engine = row.get("engine")
        if engine:
            return str(engine)
    try:
        import json

        for d in shard_dirs:
            meta = d / "results_meta.json"
            if meta.exists():
                engine = json.loads(meta.read_text()).get("engine")
                if engine:
                    return str(engine)
    except Exception:
        pass
    return "unidock"

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


def _add_cnn_score_cols(
    rows: list[dict],
    fieldnames: list[str],
    output_dir: Path,
    id_col: str | None,
) -> tuple[list[dict], list[str]]:
    """Left-join GNINA CNN scores from cnn_scores.csv. No-op if the file is absent."""
    if not id_col:
        return rows, fieldnames
    cnn_path = output_dir / "cnn_scores.csv"
    if not cnn_path.exists():
        return rows, fieldnames
    try:
        cnn: dict[str, dict] = {}
        with cnn_path.open() as f:
            for r in csv.DictReader(f):
                key = r.get("lig_id") or r.get("ligand") or ""
                if key:
                    cnn[key] = r
        if not cnn:
            return rows, fieldnames

        new_fields = list(fieldnames)
        for col in ("CNNscore", "CNNaffinity"):
            if col not in new_fields:
                new_fields.append(col)
        for row in rows:
            entry = cnn.get(row.get(id_col, ""))
            if entry:
                # fresh score for this row (a top-N rescore) — overwrite
                row["CNNscore"]    = entry.get("CNNscore", row.get("CNNscore", ""))
                row["CNNaffinity"] = entry.get("CNNaffinity", row.get("CNNaffinity", ""))
            else:
                # not in this batch — keep any existing value (e.g. from GNINA docking),
                # don't blank it when only the top-N were rescored
                row.setdefault("CNNscore", "")
                row.setdefault("CNNaffinity", "")
        return rows, new_fields
    except Exception:
        return rows, fieldnames


def join_cnn_scores(output_dir: Path) -> bool:
    """Left-join cnn_scores.csv into an existing scores.csv in place.

    Used by the results viewer after an on-demand GNINA rescore, so CNN columns
    appear without a full re-merge. Returns True if CNN columns were added.
    """
    scores_csv = output_dir / "scores.csv"
    if not scores_csv.exists():
        return False
    with scores_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows or not fieldnames:
        return False
    id_col = fieldnames[0]
    new_rows, new_fields = _add_cnn_score_cols(rows, fieldnames, output_dir, id_col)
    if "CNNscore" not in new_fields:
        return False
    with scores_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(new_rows)
    return True


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


def _load_failed_prep(shard_dirs: list[Path]) -> list[dict]:
    """Return Kaggle-side ligand prep failures from failed_prep.csv files."""
    rows: list[dict] = []
    for d in shard_dirs:
        fp = d / "failed_prep.csv"
        if not fp.exists():
            continue
        with fp.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def _write_unscored_reasons(
    output_dir: Path,
    index: dict[str, dict],
    successful_ids: set[str],
    score_filtered: list[dict],
    gpu_filtered: list[dict],
    failed_prep: list[dict],
    failed_docking: list[dict],
    id_col: str | None,
    score_col: str | None,
) -> Path:
    reasons: list[dict] = []
    keyed_reasons: dict[tuple[str, str], dict] = {}

    def _append_ligand_reason(row: dict, reason: str, raw_score: str = "") -> None:
        lig_id = row.get("ligand", "")
        key = (lig_id, reason)
        if lig_id and key in keyed_reasons:
            if raw_score:
                existing = keyed_reasons[key].get("raw_score", "")
                scores = [s for s in existing.split("; ") if s]
                if raw_score not in scores:
                    scores.append(raw_score)
                keyed_reasons[key]["raw_score"] = "; ".join(scores)
            return
        reasons.append(row)
        if lig_id:
            keyed_reasons[key] = row

    # Compounds that reached merger but were cut by floor/ceiling.
    score_filtered_ids: set[str] = set()
    for row in score_filtered:
        lig_id = row.get(id_col, "") if id_col else ""
        score_filtered_ids.add(lig_id)
        if lig_id in successful_ids:
            continue
        info = index.get(lig_id, {})
        reason = row.get("_reason", "score_filtered")
        raw_score = row.get(score_col, "") if score_col else ""
        if row.get("receptor_variant") and raw_score:
            raw_score = f"{row['receptor_variant']}:{raw_score}"
        _append_ligand_reason({
            "ligand":    lig_id,
            "name":      info.get("name", row.get("name", "")),
            "smiles":    info.get("smiles", row.get("smiles", "")),
            "reason":    reason,
            "raw_score": raw_score,
        }, reason, raw_score)

    # Compounds the Kaggle notebook rejected (score_ceiling, no_remark, etc.)
    # These IDs are in index but never reached merger's scores.csv.
    docking_failed_ids: set[str] = set()
    for row in failed_docking:
        lig_id = row.get("ligand", "")
        docking_failed_ids.add(lig_id)
        if lig_id in successful_ids:
            continue
        info = index.get(lig_id, {})
        reason = row.get("reason", "docking_failed")
        raw_score = row.get("raw_score", "")
        if row.get("receptor_variant") and raw_score:
            raw_score = f"{row['receptor_variant']}:{raw_score}"
        _append_ligand_reason({
            "ligand":    lig_id,
            "name":      info.get("name", ""),
            "smiles":    info.get("smiles", ""),
            "reason":    reason,
            "raw_score": raw_score,
        }, reason, raw_score)

    # Compounds that have no output PDBQT at all (UniDock never produced a pose).
    all_accounted = successful_ids | docking_failed_ids | score_filtered_ids
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

    for row in failed_prep:
        reasons.append({
            "ligand":    "",
            "name":      row.get("name", ""),
            "smiles":    row.get("smiles", ""),
            "reason":    row.get("reason", "prep_failed"),
            "raw_score": "",
        })

    out = output_dir / "unscored_reasons.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ligand", "name", "smiles", "reason", "raw_score"])
        w.writeheader()
        w.writerows(reasons)
    return out


def _score_is_higher_better(engine: str) -> bool:
    return engine == "diffdock"


def _score_col_from_headers(headers: list[str]) -> str:
    return next(
        (h for h in headers if "score" in h.lower() or "affinity" in h.lower() or "confidence" in h.lower()),
        headers[-1] if headers else "",
    )


def _row_identity(row: dict) -> str:
    return row.get("smiles") or row.get("name") or row.get("ligand") or ""


def build_consensus_results(engine_output_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Rank ligands by mean normalized rank across completed engine outputs."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    by_identity: dict[str, dict[str, Any]] = {}
    engines_used: list[str] = []
    fieldnames = [
        "consensus_rank",
        "consensus_norm_rank",
        "consensus_engines",
        "ligand",
        "name",
        "smiles",
    ]

    for engine_dir in engine_output_dirs:
        scores_csv = engine_dir / "scores.csv"
        if not scores_csv.exists():
            continue
        meta_path = engine_dir / "results_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        engine = str(meta.get("engine") or engine_dir.name)
        engines_used.append(engine)

        with scores_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
            headers = list(rows[0].keys()) if rows else []
        if not rows:
            continue
        score_col = str(meta.get("score_col") or _score_col_from_headers(headers))

        def _score_value(row: dict) -> float:
            try:
                return float(row.get(score_col, "0"))
            except (TypeError, ValueError):
                return 0.0

        ranked = sorted(
            rows,
            key=_score_value,
            reverse=_score_is_higher_better(engine),
        )
        denom = max(len(ranked) - 1, 1)
        score_field = f"{engine}_score"
        rank_field = f"{engine}_rank"
        for h in (score_field, rank_field):
            if h not in fieldnames:
                fieldnames.append(h)

        for rank, row in enumerate(ranked, 1):
            identity = _row_identity(row)
            if not identity:
                continue
            entry = by_identity.setdefault(
                identity,
                {
                    "ligand": row.get("ligand", ""),
                    "name": row.get("name", "") or row.get("ligand", ""),
                    "smiles": row.get("smiles", ""),
                    "_norms": [],
                    "_engines": [],
                },
            )
            for base in ("ligand", "name", "smiles"):
                if not entry.get(base) and row.get(base):
                    entry[base] = row.get(base, "")
            entry[score_field] = row.get(score_col, "")
            entry[rank_field] = str(rank)
            entry["_norms"].append((rank - 1) / denom)
            entry["_engines"].append(engine)

    rows_out: list[dict[str, Any]] = []
    for entry in by_identity.values():
        norms = entry.pop("_norms")
        seen_engines = entry.pop("_engines")
        if not norms:
            continue
        entry["consensus_norm_rank"] = f"{sum(norms) / len(norms):.6f}"
        entry["consensus_engines"] = ",".join(seen_engines)
        rows_out.append(entry)

    rows_out.sort(key=lambda r: float(r["consensus_norm_rank"]))
    for rank, row in enumerate(rows_out, 1):
        row["consensus_rank"] = str(rank)

    scores_out = output_dir / "scores.csv"
    if rows_out:
        with scores_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)
    (output_dir / "results_meta.json").write_text(
        json.dumps(
            {
                "score_type": "consensus_rank",
                "score_col": "consensus_norm_rank",
                "engine": "consensus",
                "engines": engines_used,
            },
            indent=2,
        )
    )
    return {
        "scores_csv": scores_out,
        "total_hits": len(rows_out),
        "score_col": "consensus_norm_rank",
        "score_type": "consensus_rank",
        "engine": "consensus",
        "engines": engines_used,
    }

def merge_shard_results(shard_dirs: list[Path], output_dir: Path, engine: str | None = None) -> dict[str, Any]:
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

    result_engine = _infer_engine(shard_dirs, all_rows, engine)
    if all_rows:
        if "engine" not in fieldnames:
            fieldnames.append("engine")
        for row in all_rows:
            if not row.get("engine"):
                row["engine"] = result_engine
    # Attach name/smiles from index if not already in scores.csv
    if index:
        for row in all_rows:
            lig_id = row.get("ligand", "")
            if lig_id in index:
                if "smiles" not in row:
                    row["name"]   = index[lig_id]["name"]
                    row["smiles"] = index[lig_id]["smiles"]
                row["conformer_qc"] = index[lig_id].get("conformer_qc", "")
        if all_rows and "smiles" in all_rows[0] and "name" not in fieldnames:
            fieldnames += ["name", "smiles"]
        if any(r.get("conformer_qc") for r in all_rows) and "conformer_qc" not in fieldnames:
            fieldnames += ["conformer_qc"]

    id_col = fieldnames[0] if fieldnames else None
    score_col = _score_col_from_headers(fieldnames) if fieldnames else None

    higher_is_better = _score_is_higher_better(result_engine)

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
    if _enable_floor and result_engine != "diffdock":
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
    valid_scored_ids = {row.get(id_col, "") for row in all_rows} if id_col else set()

    receptor_variant_mode = any(row.get("receptor_variant") for row in all_rows)
    if receptor_variant_mode and "receptor_variant" not in fieldnames:
        fieldnames.append("receptor_variant")

    best: dict[str, dict] = {}
    for row in all_rows:
        if receptor_variant_mode:
            key = row.get("smiles") or row.get("name") or row.get(id_col, "")
        else:
            key = row.get(id_col, "") if id_col else str(id(row))
        if key not in best:
            best[key] = row
            continue
        better = _score(row) > _score(best[key]) if higher_is_better else _score(row) < _score(best[key])
        if better:
            best[key] = row

    deduped = sorted(best.values(), key=_score, reverse=higher_is_better)

    # Append GPU-filtered molecules with score=0 so they appear in results
    # ranked last rather than being excluded from AUC/EF calculations entirely.
    gpu_filtered = _load_filtered(shard_dirs)
    if gpu_filtered:
        if not fieldnames:
            fieldnames = ["ligand", "score", "name", "smiles"]
        if "name" not in fieldnames and any(r.get("name") for r in gpu_filtered):
            fieldnames += ["name", "smiles"]
        if "engine" not in fieldnames:
            fieldnames.append("engine")
        for i, fmol in enumerate(gpu_filtered):
            deduped.append({
                "ligand": f"gpu_filt_{i:05d}",
                "score":  0.0,
                "name":   fmol.get("name", ""),
                "smiles": fmol.get("smiles", ""),
                "engine": result_engine,
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

    deduped, fieldnames = _add_cnn_score_cols(deduped, fieldnames, output_dir, id_col)

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
    failed_prep = _load_failed_prep(shard_dirs)

    reasons_out = _write_unscored_reasons(
        output_dir, index, valid_scored_ids, score_filtered, gpu_filtered, failed_prep, failed_docking, id_col, score_col,
    )

    reason_counts: dict[str, int] = {}
    with reasons_out.open() as f:
        for r in csv.DictReader(f):
            k = r.get("reason", "unknown")
            reason_counts[k] = reason_counts.get(k, 0) + 1
    if "no_pose" not in reason_counts:
        reason_counts["no_pose"] = 0
    reason_counts["gpu_size_filter"] = len(gpu_filtered)

    # Record native score metadata so the viewer/report can label the scale.
    import json

    from ezscreen.backends import engines as _engines

    score_type = _engines.native_score_type(result_engine)
    (output_dir / "results_meta.json").write_text(
        json.dumps(
            {"score_type": score_type, "score_col": score_col, "engine": result_engine},
            indent=2,
        )
    )
    return {
        "scores_csv": scores_out,
        "poses_sdf": poses_out,
        "failed_prep_sdf": failed_out if has_failures else None,
        "total_hits": len(deduped),
        "score_col": score_col,
        "score_type": score_type,
        "engine": result_engine,
        "unscored_reasons_csv": reasons_out,
        "unscored_reason_counts": reason_counts,
    }
