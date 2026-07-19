from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT = Path(__file__).resolve().parents[1]
CELL = AUDIT.parent
REPO = AUDIT.parents[4]
SOURCE = CELL / "run-allrescore-kaggle-4shards-001"
ORIGINAL = CELL / "run-multiacct-t4-4x2-001"
EXTERNAL = REPO / "benchmarks" / "external_sources" / "dude_ppard"

BOX_CENTER = [15.656, 2.308, 39.483]
BOX_SIZE = [21.337, 22.566, 22.752]

REDOCKING = AUDIT / "redocking"
INPUTS = AUDIT / "inputs"
PANEL = AUDIT / "audit_panel"
POSE_VALIDITY = AUDIT / "pose_validity"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _read_scores() -> list[dict[str, str]]:
    path = SOURCE / "merged_outputs" / "scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_inputs() -> dict[str, str]:
    INPUTS.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for src in [
        SOURCE / "inputs" / "receptor_fixed.pdbqt",
        SOURCE / "inputs" / "receptor_prep.pdb",
        SOURCE / "inputs" / "receptor_fixed_clean.pdb",
        SOURCE / "inputs" / "actives.smi",
        SOURCE / "inputs" / "decoys.smi",
        SOURCE / "inputs" / "active_and_decoy.ism",
        EXTERNAL / "crystal_ligand.mol2",
        EXTERNAL / "receptor.pdb",
        EXTERNAL / "pdb_selection.txt",
    ]:
        if not src.exists():
            raise FileNotFoundError(src)
        dest = INPUTS / src.name
        shutil.copy2(src, dest)
        copied[src.name] = str(dest)
    return copied


def _load_crystal_mol():
    from rdkit import Chem

    mol2 = INPUTS / "crystal_ligand.mol2"
    mol = Chem.MolFromMol2File(str(mol2), sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMol2File(str(mol2), sanitize=False, removeHs=False)
        if mol is None:
            raise RuntimeError(f"RDKit could not read {mol2}")
        Chem.SanitizeMol(mol, catchErrors=True)
    mol.SetProp("_Name", "2ZNP_crystal_ligand")
    return mol


def _heavy_coords(mol) -> list[tuple[float, float, float]]:
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return coords


def _centroid(coords: list[tuple[float, float, float]]) -> list[float]:
    return [sum(pt[i] for pt in coords) / len(coords) for i in range(3)]


def _extent(coords: list[tuple[float, float, float]]) -> list[float]:
    return [max(pt[i] for pt in coords) - min(pt[i] for pt in coords) for i in range(3)]


def _mol_to_pdbqt(mol) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit.Chem import AllChem

    try:
        mol_h = AllChem.AddHs(mol, addCoords=True)
    except Exception:
        mol_h = mol
    setups = MoleculePreparation().prepare(mol_h)
    if not setups:
        raise RuntimeError("Meeko produced no ligand setup")
    pdbqt, ok, detail = PDBQTWriterLegacy.write_string(setups[0])
    if not ok or not pdbqt:
        raise RuntimeError(f"Meeko PDBQT writer failed: {detail}")
    return "REMARK lig_id cocrystal_2znp\n" + pdbqt


def _prepare_crystal_ligand() -> dict[str, Any]:
    from rdkit import Chem

    REDOCKING.mkdir(parents=True, exist_ok=True)
    mol = _load_crystal_mol()
    heavy = Chem.RemoveHs(mol)
    heavy.SetProp("_Name", "2ZNP_crystal_ligand")
    sdf = REDOCKING / "crystal_ligand.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(heavy)
    writer.close()
    Chem.MolToPDBFile(heavy, str(REDOCKING / "crystal_ligand.pdb"))
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    (REDOCKING / "crystal_ligand.smi").write_text(
        f"{smiles} 2ZNP_crystal_ligand\n", encoding="utf-8"
    )

    pdbqt_path = REDOCKING / "cocrystal_2znp.pdbqt"
    prep_backend = "meeko"
    try:
        pdbqt_path.write_text(_mol_to_pdbqt(mol), encoding="utf-8")
    except Exception as exc:
        prep_backend = f"obabel_fallback_after_meeko:{type(exc).__name__}"
        result = subprocess.run(
            ["obabel", str(INPUTS / "crystal_ligand.mol2"), "-O", str(pdbqt_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not pdbqt_path.exists():
            raise RuntimeError(result.stderr[-1000:] or "obabel PDBQT conversion failed")
        text = pdbqt_path.read_text(errors="ignore")
        if not text.startswith("REMARK lig_id"):
            pdbqt_path.write_text(
                "REMARK lig_id cocrystal_2znp\n" + text, encoding="utf-8"
            )

    coords = _heavy_coords(heavy)
    centroid = _centroid(coords)
    extent = _extent(coords)
    return {
        "source": str(INPUTS / "crystal_ligand.mol2"),
        "sdf": str(sdf),
        "pdbqt": str(pdbqt_path),
        "prep_backend": prep_backend,
        "heavy_atoms": heavy.GetNumHeavyAtoms(),
        "smiles": smiles,
        "centroid": [round(x, 3) for x in centroid],
        "extent": [round(x, 3) for x in extent],
        "current_box_center": BOX_CENTER,
        "current_box_size": BOX_SIZE,
        "center_delta_from_ligand_centroid": [
            round(BOX_CENTER[i] - centroid[i], 3) for i in range(3)
        ],
        "nominal_padding_each_axis": [
            round((BOX_SIZE[i] - extent[i]) / 2.0, 3) for i in range(3)
        ],
    }


def _row_key(row: dict[str, str]) -> str:
    return _sha1(f"{row.get('name','')}|{row.get('smiles','')}")


def _pick_unique(
    rows: list[dict[str, str]],
    picked: dict[str, dict[str, str]],
    count: int,
    reason: str,
) -> None:
    for row in rows:
        key = _row_key(row)
        if key in picked:
            continue
        row = dict(row)
        row["audit_reason"] = reason
        picked[key] = row
        if sum(1 for r in picked.values() if r["audit_reason"] == reason) >= count:
            return


def _select_audit_panel(scores: list[dict[str, str]]) -> dict[str, Any]:
    PANEL.mkdir(parents=True, exist_ok=True)
    active_rows = [r for r in scores if str(r.get("label", "")).strip() == "1"]
    decoy_rows = [r for r in scores if str(r.get("label", "")).strip() != "1"]
    picked: dict[str, dict[str, str]] = {}

    _pick_unique(sorted(active_rows, key=lambda r: _float(r, "score")), picked, 5, "active_best_raw")
    _pick_unique(sorted(active_rows, key=lambda r: -_float(r, "CNNscore")), picked, 5, "active_best_cnn")
    _pick_unique(sorted(active_rows, key=_row_key), picked, 5, "active_hash_sample")
    _pick_unique(sorted(decoy_rows, key=lambda r: _float(r, "score")), picked, 20, "hard_top_raw_decoy")

    med = {
        key: sorted(_float(r, key) for r in active_rows)[len(active_rows) // 2]
        for key in ["molecular_weight", "heavy_atoms", "tpsa", "rotatable_bonds"]
    }

    def neutral_distance(row: dict[str, str]) -> float:
        return (
            abs(_float(row, "molecular_weight") - med["molecular_weight"]) / 50.0
            + abs(_float(row, "heavy_atoms") - med["heavy_atoms"]) / 5.0
            + abs(_float(row, "tpsa") - med["tpsa"]) / 25.0
            + abs(_float(row, "rotatable_bonds") - med["rotatable_bonds"]) / 4.0
        )

    neutral = [
        r
        for r in decoy_rows
        if round(_float(r, "abs_formal_charge"), 6) == 0
    ]
    _pick_unique(sorted(neutral, key=neutral_distance), picked, 15, "neutral_property_matched_decoy")

    panel_rows = list(picked.values())
    fields = [
        "audit_id",
        "audit_reason",
        "label",
        "ligand",
        "name",
        "smiles",
        "score",
        "CNNscore",
        "CNNaffinity",
        "prolif_score",
        "abs_formal_charge",
        "heavy_atoms",
        "molecular_weight",
        "tpsa",
        "rotatable_bonds",
        "shard",
    ]
    for idx, row in enumerate(panel_rows):
        row["audit_id"] = f"ppard_audit_{idx:03d}"
    with (PANEL / "audit_panel.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(panel_rows)
    with (PANEL / "audit_panel.smi").open("w", encoding="utf-8") as handle:
        for row in panel_rows:
            handle.write(f"{row.get('smiles','')} {row['audit_id']} {row.get('name','')}\n")
    counts: dict[str, int] = {}
    for row in panel_rows:
        counts[row["audit_reason"]] = counts.get(row["audit_reason"], 0) + 1
    return {"panel_rows": len(panel_rows), "reason_counts": counts, "active_rows": sum(int(r["label"]) for r in panel_rows)}


def _pose_validity_snapshot(scores: list[dict[str, str]]) -> dict[str, Any]:
    POSE_VALIDITY.mkdir(parents=True, exist_ok=True)
    rows = sorted(scores, key=lambda r: _float(r, "score"))[:20]
    fields = [
        "rank",
        "label",
        "ligand",
        "name",
        "score",
        "pose_min_heavy_atom_distance",
        "contact_count_4a",
        "contact_count_6a",
        "gaussian_contact_score",
        "score_qc",
        "shard",
        "smiles",
    ]
    clash_rows = 0
    active_rows = 0
    with (POSE_VALIDITY / "current_raw_top20_pose_validity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            out = dict(row)
            out["rank"] = idx
            writer.writerow(out)
            if _float(row, "pose_min_heavy_atom_distance", 999.0) < 2.0:
                clash_rows += 1
            active_rows += int(str(row.get("label", "")).strip() == "1")
    return {
        "top20_rows": len(rows),
        "top20_active_rows": active_rows,
        "top20_clash_like_rows_min_heavy_distance_lt_2a": clash_rows,
        "min_pose_min_heavy_atom_distance": min(
            _float(r, "pose_min_heavy_atom_distance", 999.0) for r in rows
        ),
        "median_contact_count_4a": sorted(_float(r, "contact_count_4a") for r in rows)[len(rows) // 2],
    }


def _write_seed_metric_row() -> None:
    fields = [
        "ablation_id",
        "status",
        "redocking_rmsd",
        "redocking_reliable",
        "ef1",
        "ef5",
        "auc_roc",
        "active_survival",
        "top20_actives",
        "top20_decoys",
        "pose_clash_like_top20",
        "runtime_seconds",
        "notes",
    ]
    seed = {
        "ablation_id": "prior_full_screen_current_box_balance",
        "status": "prior_full_screen_metrics_only",
        "redocking_rmsd": "",
        "redocking_reliable": "",
        "ef1": "0.420",
        "ef5": "1.001",
        "auc_roc": "0.7011",
        "active_survival": "240/240",
        "top20_actives": "0",
        "top20_decoys": "20",
        "pose_clash_like_top20": "",
        "runtime_seconds": "",
        "notes": "Seed row from completed PPARD all-pose rescoring.",
    }
    path = AUDIT / "docking_ablation_metrics.csv"
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not any(row.get("ablation_id") == seed["ablation_id"] for row in rows):
        rows.insert(0, seed)
    else:
        rows = [seed if row.get("ablation_id") == seed["ablation_id"] else row for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    copied = _copy_inputs()
    crystal = _prepare_crystal_ligand()
    scores = _read_scores()
    panel = _select_audit_panel(scores)
    pose_validity = _pose_validity_snapshot(scores)
    _write_seed_metric_row()

    _write_json(REDOCKING / "box_sanity.json", crystal)
    redock_status_path = REDOCKING / "redocking_status.json"
    if not (REDOCKING / "redocking_rmsd_latest.json").exists():
        _write_json(
            redock_status_path,
            {
                "status": "prepared_not_docked",
                "updated_at": _now(),
                "reason": "local UniDock binary unavailable; use submit_redocking_kaggle.py for actual UniDock redocking.",
                "prepared_ligand_pdbqt": crystal["pdbqt"],
                "current_receptor_pdbqt": copied["receptor_fixed.pdbqt"],
                "current_box_center": BOX_CENTER,
                "current_box_size": BOX_SIZE,
            },
        )
    summary_path = AUDIT / "docking_ablation_summary.json"
    existing_summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_summary = {}
    summary = {
        **existing_summary,
        "status": existing_summary.get("status", "prepared"),
        "updated_at": _now(),
        "scope": "standalone PPARD docking audit; no core code paths changed",
        "prior_full_screen_metrics": {
            "raw_unidock": {"ef1": 0.420, "ef5": 1.001, "auc_roc": 0.7011},
            "active_survival": "240/240",
            "top20_raw": "0 actives / 20 decoys",
        },
        "crystal_ligand": crystal,
        "audit_panel": panel,
        "pose_validity_snapshot": pose_validity,
        "local_tool_availability": {
            "rdkit": True,
            "meeko": True,
            "obabel": shutil.which("obabel") is not None,
            "unidock": shutil.which("unidock") is not None,
        },
        "next_command": "python scripts/submit_redocking_kaggle.py --ablation-id current_box_balance",
    }
    _write_json(summary_path, summary)
    print(json.dumps({"status": "prepared", "audit_dir": str(AUDIT), "panel": panel}, indent=2))


if __name__ == "__main__":
    main()
