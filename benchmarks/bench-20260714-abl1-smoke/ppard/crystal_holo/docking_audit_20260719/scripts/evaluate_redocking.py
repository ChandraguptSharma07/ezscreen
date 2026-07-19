from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT = Path(__file__).resolve().parents[1]
REDOCKING = AUDIT / "redocking"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _coords(mol) -> list[tuple[float, float, float]]:
    conf = mol.GetConformer()
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        out.append((float(pos.x), float(pos.y), float(pos.z)))
    return out


def _centroid(coords: list[tuple[float, float, float]]) -> list[float]:
    return [sum(pt[i] for pt in coords) / len(coords) for i in range(3)]


def _receptor_coords(path: Path) -> list[tuple[float, float, float]]:
    coords = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if element == "H" or atom_name.startswith("H"):
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return coords


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _pose_contacts(ligand_coords: list[tuple[float, float, float]], receptor_coords: list[tuple[float, float, float]]) -> dict[str, Any]:
    min_dist = 999.0
    contacts_4a = 0
    contacts_6a = 0
    clashes_2a = 0
    for lc in ligand_coords:
        for rc in receptor_coords:
            dist = _distance(lc, rc)
            min_dist = min(min_dist, dist)
            if dist < 2.0:
                clashes_2a += 1
            if dist <= 4.0:
                contacts_4a += 1
            if dist <= 6.0:
                contacts_6a += 1
    return {
        "min_heavy_atom_distance": round(min_dist, 3),
        "contacts_4a": contacts_4a,
        "contacts_6a": contacts_6a,
        "clashes_lt_2a": clashes_2a,
    }


def _load_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _update_metrics(ablation_id: str, payload: dict[str, Any]) -> None:
    path = AUDIT / "docking_ablation_metrics.csv"
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
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("ablation_id") != ablation_id]
    rows.append(
        {
            "ablation_id": ablation_id,
            "status": str(payload.get("status", "")),
            "redocking_rmsd": "" if payload.get("redocking_rmsd") is None else str(payload.get("redocking_rmsd")),
            "redocking_reliable": str(payload.get("redocking_reliable", "")),
            "ef1": "",
            "ef5": "",
            "auc_roc": "",
            "active_survival": "",
            "top20_actives": "",
            "top20_decoys": "",
            "pose_clash_like_top20": str(payload.get("clashes_lt_2a", "")),
            "runtime_seconds": str(payload.get("runtime_seconds", "")),
            "notes": str(payload.get("notes", "")),
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output_dir: Path, ablation_id: str, runtime_seconds: float | None = None) -> dict[str, Any]:
    from rdkit import Chem
    from ezscreen.benchmark.redock import symmetry_rmsd

    ref = next(m for m in Chem.SDMolSupplier(str(REDOCKING / "crystal_ligand.sdf"), removeHs=False) if m is not None)
    poses_path = output_dir / "poses.sdf"
    pose_mols = [m for m in Chem.SDMolSupplier(str(poses_path), removeHs=False) if m is not None]
    if not pose_mols:
        raise RuntimeError(f"No poses parsed from {poses_path}")
    pose = pose_mols[0]
    rmsd = symmetry_rmsd(ref, pose)
    receptor = AUDIT / "inputs" / "receptor_prep.pdb"
    contacts = _pose_contacts(_coords(pose), _receptor_coords(receptor))
    ref_centroid = _centroid(_coords(ref))
    pose_centroid = _centroid(_coords(pose))
    rows = _load_scores(output_dir / "scores.csv") if (output_dir / "scores.csv").exists() else []
    result = {
        "status": "ok" if rmsd is not None else "rmsd_failed",
        "updated_at": _now(),
        "ablation_id": ablation_id,
        "output_dir": str(output_dir),
        "scores_rows": len(rows),
        "redocking_rmsd": None if rmsd is None else round(float(rmsd), 3),
        "redocking_reliable": bool(rmsd is not None and rmsd <= 2.0),
        "reference_ligand": str(REDOCKING / "crystal_ligand.sdf"),
        "redocked_pose_sdf": str(poses_path),
        "crystal_centroid": [round(x, 3) for x in ref_centroid],
        "redocked_centroid": [round(x, 3) for x in pose_centroid],
        "centroid_delta": [round(pose_centroid[i] - ref_centroid[i], 3) for i in range(3)],
        "pose_validity": contacts,
        "runtime_seconds": runtime_seconds,
        "notes": "Heavy-atom symmetry RMSD via RDKit CalcRMS, no alignment.",
    }
    _write_json(REDOCKING / f"redocking_rmsd_{ablation_id}.json", result)
    _write_json(REDOCKING / "redocking_rmsd_latest.json", result)
    _update_metrics(ablation_id, {**result, **contacts})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ablation-id", default="current_box_balance")
    parser.add_argument("--runtime-seconds", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output_dir, args.ablation_id, args.runtime_seconds), indent=2))


if __name__ == "__main__":
    main()
