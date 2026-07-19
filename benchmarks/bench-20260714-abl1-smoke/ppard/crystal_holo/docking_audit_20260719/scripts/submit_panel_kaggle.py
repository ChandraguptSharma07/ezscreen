from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT = Path(__file__).resolve().parents[1]
REPO = AUDIT.parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOX_CENTER = [15.656, 2.308, 39.483]
BOX_SIZE = [21.337, 22.566, 22.752]
RECEPTOR_PDBQT = AUDIT / "inputs" / "receptor_fixed.pdbqt"
PANEL_SMI = AUDIT / "audit_panel" / "audit_panel.smi"
PANEL_LABEL_CSV = AUDIT / "audit_panel" / "audit_panel.csv"
POSE_RECEPTOR_PDB = AUDIT / "inputs" / "receptor_prep.pdb"
PREP_ON_KAGGLE = True
ENUMERATE_OPTS: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_primary_kaggle() -> tuple[str, Path]:
    from ezscreen.auth import get_kaggle_json_path, load_credentials

    path = get_kaggle_json_path(load_credentials())
    if path is None or not path.exists():
        raise RuntimeError("No primary Kaggle credential path configured")
    data = json.loads(path.read_text(encoding="utf-8"))
    os.environ["KAGGLE_USERNAME"] = data["username"]
    os.environ["KAGGLE_KEY"] = data["key"]
    return data["username"], path


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _load_panel_labels() -> tuple[dict[str, int], dict[str, dict[str, str]], dict[str, int], dict[str, dict[str, str]], dict[str, int], dict[str, dict[str, str]], int]:
    labels: dict[str, int] = {}
    meta: dict[str, dict[str, str]] = {}
    labels_by_ligand: dict[str, int] = {}
    meta_by_ligand: dict[str, dict[str, str]] = {}
    labels_by_name: dict[str, int] = {}
    meta_by_name: dict[str, dict[str, str]] = {}
    active_total = 0
    with PANEL_LABEL_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row.get("label", "0") or 0)
            c = _canonical(row.get("smiles", ""))
            if c:
                labels[c] = label
                meta[c] = dict(row)
            ligand = row.get("ligand", "")
            if ligand:
                labels_by_ligand[ligand] = label
                meta_by_ligand[ligand] = dict(row)
            for key in (row.get("name", ""), row.get("audit_id", "")):
                if key:
                    labels_by_name[key] = label
                    meta_by_name[key] = dict(row)
            active_total += label
    return labels, meta, labels_by_ligand, meta_by_ligand, labels_by_name, meta_by_name, active_total
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


def _mol_coords(mol) -> list[tuple[float, float, float]]:
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return coords


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _contact_stats(lig_coords: list[tuple[float, float, float]], rec_coords: list[tuple[float, float, float]]) -> dict[str, Any]:
    min_dist = 999.0
    contacts_4a = 0
    contacts_6a = 0
    clashes_2a = 0
    for lc in lig_coords:
        for rc in rec_coords:
            dist = _distance(lc, rc)
            min_dist = min(min_dist, dist)
            if dist < 2.0:
                clashes_2a += 1
            if dist <= 4.0:
                contacts_4a += 1
            if dist <= 6.0:
                contacts_6a += 1
    return {
        "pose_min_heavy_atom_distance": round(min_dist, 3),
        "contact_count_4a": contacts_4a,
        "contact_count_6a": contacts_6a,
        "clashes_lt_2a": clashes_2a,
    }


def _pose_validity(output_dir: Path, rows: list[dict[str, str]], ablation_id: str) -> dict[str, Any]:
    from rdkit import Chem

    poses = [m for m in Chem.SDMolSupplier(str(output_dir / "poses.sdf"), removeHs=False) if m is not None]
    rec = _receptor_coords(POSE_RECEPTOR_PDB)
    by_name = {m.GetProp("_Name"): m for m in poses if m.HasProp("_Name")}
    out_path = AUDIT / "pose_validity" / f"panel_pose_validity_{ablation_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "label", "ligand", "name", "score", "pose_min_heavy_atom_distance", "contact_count_4a", "contact_count_6a", "clashes_lt_2a", "smiles"]
    written = 0
    clash_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            mol = by_name.get(row.get("ligand", ""))
            stats = _contact_stats(_mol_coords(mol), rec) if mol is not None else {}
            clash_rows += int(float(stats.get("pose_min_heavy_atom_distance", 999.0)) < 2.0)
            writer.writerow({
                "rank": idx,
                "label": row.get("label", ""),
                "ligand": row.get("ligand", ""),
                "name": row.get("name", ""),
                "score": row.get("score", ""),
                "pose_min_heavy_atom_distance": stats.get("pose_min_heavy_atom_distance", ""),
                "contact_count_4a": stats.get("contact_count_4a", ""),
                "contact_count_6a": stats.get("contact_count_6a", ""),
                "clashes_lt_2a": stats.get("clashes_lt_2a", ""),
                "smiles": row.get("smiles", ""),
            })
            written += 1
    return {"pose_validity_csv": str(out_path), "pose_rows": written, "clash_like_rows_min_heavy_distance_lt_2a": clash_rows}


def _metrics(rows: list[dict[str, str]], active_total: int, ablation_id: str, runtime: float) -> dict[str, Any]:
    from ezscreen.benchmark.metrics import compute_metrics

    scored = [r for r in rows if r.get("label") in {"0", "1"}]
    ranked = sorted(scored, key=lambda r: _float(r, "score"))
    result = compute_metrics([_float(r, "score") for r in ranked], [int(r["label"]) for r in ranked])
    top20 = ranked[:20]
    top20_path = AUDIT / f"top20_{ablation_id}.csv"
    fields = ["rank", "label", "audit_id", "audit_reason", "ligand", "name", "smiles", "score", "shard"]
    with top20_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for idx, row in enumerate(top20, 1):
            out = dict(row)
            out["rank"] = idx
            writer.writerow(out)
    pose = _pose_validity(AUDIT / "panel_runs" / f"output_{ablation_id}", top20, ablation_id)
    summary = {
        "status": "ok",
        "updated_at": _now(),
        "ablation_id": ablation_id,
        "ef1": result.ef1,
        "ef5": result.ef5,
        "auc_roc": result.auc_roc,
        "active_survival": f"{sum(int(r['label']) for r in scored)}/{active_total}",
        "top20_actives": sum(int(r["label"]) for r in top20),
        "top20_decoys": len(top20) - sum(int(r["label"]) for r in top20),
        "runtime_seconds": runtime,
        "top20_csv": str(top20_path),
        **pose,
    }
    return summary


def _update_metric_row(summary: dict[str, Any]) -> None:
    path = AUDIT / "docking_ablation_metrics.csv"
    fields = ["ablation_id", "status", "redocking_rmsd", "redocking_reliable", "ef1", "ef5", "auc_roc", "active_survival", "top20_actives", "top20_decoys", "pose_clash_like_top20", "runtime_seconds", "notes"]
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("ablation_id") == summary["ablation_id"]:
            row.update({
                "status": summary.get("status", row.get("status", "")),
                "ef1": str(summary.get("ef1", "")),
                "ef5": str(summary.get("ef5", "")),
                "auc_roc": str(summary.get("auc_roc", "")),
                "active_survival": str(summary.get("active_survival", "")),
                "top20_actives": str(summary.get("top20_actives", "")),
                "top20_decoys": str(summary.get("top20_decoys", "")),
                "pose_clash_like_top20": str(summary.get("clash_like_rows_min_heavy_distance_lt_2a", "")),
                "runtime_seconds": str(summary.get("runtime_seconds", "")),
                "notes": "Redocking RMSD plus 50-ligand panel baseline.",
            })
            break
    else:
        rows.append({k: "" for k in fields} | {
            "ablation_id": str(summary["ablation_id"]),
            "status": str(summary.get("status", "")),
            "ef1": str(summary.get("ef1", "")),
            "ef5": str(summary.get("ef5", "")),
            "auc_roc": str(summary.get("auc_roc", "")),
            "active_survival": str(summary.get("active_survival", "")),
            "top20_actives": str(summary.get("top20_actives", "")),
            "top20_decoys": str(summary.get("top20_decoys", "")),
            "pose_clash_like_top20": str(summary.get("clash_like_rows_min_heavy_distance_lt_2a", "")),
            "runtime_seconds": str(summary.get("runtime_seconds", "")),
            "notes": "50-ligand panel baseline only.",
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def _parent_name(name: str) -> str:
    return re.sub(r"_v\d+$", "", name or "")


def _collapse_best_by_parent(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    by_parent: dict[str, dict[str, str]] = {}
    duplicates = 0
    for row in rows:
        parent = row.get("audit_id") or row.get("parent_name") or _parent_name(row.get("name", "")) or row.get("ligand", "")
        row["parent_key"] = parent
        current = by_parent.get(parent)
        if current is not None:
            duplicates += 1
        if current is None or _float(row, "score", 999.0) < _float(current, "score", 999.0):
            by_parent[parent] = row
    collapsed = sorted(by_parent.values(), key=lambda r: _float(r, "score", 999.0))
    return collapsed, duplicates


def _evaluate_panel(output_dir: Path, ablation_id: str, runtime: float) -> dict[str, Any]:
    labels, meta, labels_by_ligand, meta_by_ligand, labels_by_name, meta_by_name, active_total = _load_panel_labels()
    with (output_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        c = _canonical(row.get("smiles", ""))
        lig = row.get("ligand", "")
        input_name = row.get("name", "")
        parent_name = _parent_name(input_name)
        row_meta = meta.get(c) if c in meta else meta_by_ligand.get(lig)
        if row_meta is None:
            row_meta = meta_by_name.get(input_name) or meta_by_name.get(parent_name)
        row_label = labels.get(c) if c in labels else labels_by_ligand.get(lig)
        if row_label is None:
            row_label = labels_by_name.get(input_name) if input_name in labels_by_name else labels_by_name.get(parent_name)
        row["label"] = "" if row_label is None else str(row_label)
        row["input_name"] = input_name
        row["parent_name"] = parent_name
        if row_meta:
            row["audit_id"] = row_meta.get("audit_id", "")
            row["audit_reason"] = row_meta.get("audit_reason", "")
            row["name"] = row_meta.get("name", row.get("name", ""))
            row["smiles"] = row.get("smiles") or row_meta.get("smiles", "")
    evaluated = [r for r in rows if r.get("label") in {"0", "1"}]
    collapsed, duplicate_variant_rows = _collapse_best_by_parent(evaluated)
    out_csv = AUDIT / "panel_runs" / f"scores_labeled_{ablation_id}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = _metrics(collapsed, active_total, ablation_id, runtime)
    summary["scores_labeled_csv"] = str(out_csv)
    summary["raw_score_rows"] = len(rows)
    summary["evaluated_score_rows"] = len(evaluated)
    summary["collapsed_parent_rows"] = len(collapsed)
    summary["duplicate_variant_rows"] = duplicate_variant_rows
    (AUDIT / "panel_runs" / f"panel_summary_{ablation_id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _update_metric_row(summary)
    return summary

def submit(ablation_id: str, search_mode: str, accelerator: str) -> dict[str, object]:
    from ezscreen.backends.kaggle.runner import render_vina_notebook, run_screening_job

    username, key_path = _load_primary_kaggle()
    run_id = f"ezs-ppard-panel-{ablation_id}".lower().replace("_", "-")
    work = AUDIT / "panel_runs" / f"kaggle_{ablation_id}"
    work.mkdir(parents=True, exist_ok=True)
    notebook = work / "panel_unidock.ipynb"
    notebook.write_text(
        render_vina_notebook(
            run_id=run_id,
            shard_filenames=[PANEL_SMI.name],
            notebook_index=0,
            total_notebooks=1,
            box_center=BOX_CENTER,
            box_size=BOX_SIZE,
            search_mode=search_mode,
            ph=7.4,
            prep_on_kaggle=PREP_ON_KAGGLE,
            accelerator=accelerator,
            enumerate_opts=ENUMERATE_OPTS,
            results_cfg={"poses_returned": 1000000, "prolif_rescore_enabled": False},
            engine="unidock",
            scoring="vina",
            receptor_filename="receptor.pdbqt",
        ),
        encoding="utf-8",
    )
    started = time.time()
    result = run_screening_job(
        run_id=run_id,
        receptor_pdbqt=RECEPTOR_PDBQT,
        shard_paths=[PANEL_SMI],
        notebook_path=notebook,
        username=username,
        work_dir=work,
        retry_limit=1,
        accelerator=accelerator,
        engine="unidock",
    )
    runtime = round(time.time() - started, 3)
    summary: dict[str, Any] = {
        "status": result.get("status"),
        "ablation_id": ablation_id,
        "run_id": run_id,
        "username": username,
        "kaggle_key_path": str(key_path),
        "work_dir": str(work),
        "runtime_seconds": runtime,
        "result": {k: str(v) if isinstance(v, Path) else v for k, v in result.items()},
    }
    if result.get("status") == "complete" and result.get("output_dir"):
        output_dir = Path(str(result["output_dir"]))
        archived = AUDIT / "panel_runs" / f"output_{ablation_id}"
        if archived.exists():
            shutil.rmtree(archived)
        shutil.copytree(output_dir, archived)
        summary["panel_evaluation"] = _evaluate_panel(archived, ablation_id, runtime)
    (work / "submission_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    global RECEPTOR_PDBQT, PANEL_SMI, PANEL_LABEL_CSV, POSE_RECEPTOR_PDB, PREP_ON_KAGGLE, ENUMERATE_OPTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-id", default="current_box_balance")
    parser.add_argument("--search-mode", default="balance", choices=["fast", "balance", "detail"])
    parser.add_argument("--accelerator", default="nvidiaTeslaT4")
    parser.add_argument("--receptor-pdbqt", type=Path, default=RECEPTOR_PDBQT)
    parser.add_argument("--panel-smi", type=Path, default=PANEL_SMI)
    parser.add_argument("--panel-label-csv", type=Path, default=PANEL_LABEL_CSV)
    parser.add_argument("--pose-receptor-pdb", type=Path, default=POSE_RECEPTOR_PDB)
    parser.add_argument("--prep-on-kaggle", action=argparse.BooleanOptionalAction, default=PREP_ON_KAGGLE)
    parser.add_argument("--enumerate-ligands", action="store_true")
    parser.add_argument("--enumerate-max-variants", type=int, default=2)
    args = parser.parse_args()
    RECEPTOR_PDBQT = args.receptor_pdbqt
    PANEL_SMI = args.panel_smi
    PANEL_LABEL_CSV = args.panel_label_csv
    POSE_RECEPTOR_PDB = args.pose_receptor_pdb
    PREP_ON_KAGGLE = args.prep_on_kaggle
    ENUMERATE_OPTS = {"enabled": True, "protonation": True, "tautomers": True, "stereo": False, "ring": False, "max_variants": args.enumerate_max_variants} if args.enumerate_ligands else None
    print(json.dumps(submit(args.ablation_id, args.search_mode, args.accelerator), indent=2))

if __name__ == "__main__":
    main()