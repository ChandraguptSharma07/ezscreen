from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT = Path(__file__).resolve().parents[1]
INPUTS = AUDIT / "inputs"
PANEL = AUDIT / "audit_panel"
REDOCKING = AUDIT / "redocking"
VARIANTS = AUDIT / "prep_variants"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _formal_charge(mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _uncharge_smiles(smiles: str) -> tuple[str | None, int | None, int | None]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    before = _formal_charge(mol)
    uncharged = rdMolStandardize.Uncharger().uncharge(mol)
    after = _formal_charge(uncharged)
    return Chem.MolToSmiles(uncharged, isomericSmiles=True), before, after


def _write_uncharge_panel() -> dict[str, Any]:
    src = PANEL / "audit_panel.csv"
    out_csv = VARIANTS / "ligands" / "audit_panel_uncharge.csv"
    out_smi = VARIANTS / "ligands" / "audit_panel_uncharge.smi"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(handle.name and rows[0].keys() if rows else [])
    extra = ["original_smiles", "original_formal_charge", "variant_formal_charge", "smiles_changed"]
    fieldnames = list(dict.fromkeys(fields + extra))
    changed = 0
    failed = 0
    with out_csv.open("w", newline="", encoding="utf-8") as csv_handle, out_smi.open("w", encoding="utf-8") as smi_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            original = row.get("smiles", "")
            variant, before, after = _uncharge_smiles(original)
            if not variant:
                failed += 1
                variant = original
            changed += int(_canonical(original) != _canonical(variant))
            row = dict(row)
            row["original_smiles"] = original
            row["original_formal_charge"] = "" if before is None else str(before)
            row["variant_formal_charge"] = "" if after is None else str(after)
            row["smiles_changed"] = str(_canonical(original) != _canonical(variant)).lower()
            row["smiles"] = variant
            writer.writerow(row)
            name = row.get("name") or row.get("audit_id") or "ligand"
            smi_handle.write(f"{variant} {name}\n")
    return {
        "variant_id": "ligand_uncharge",
        "kind": "ligand_charge_protonation",
        "panel_csv": str(out_csv),
        "panel_smi": str(out_smi),
        "input_rows": len(rows),
        "changed_rows": changed,
        "failed_rows": failed,
    }


def _mol_to_pdbqt(mol, lig_id: str) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit.Chem import AllChem

    mol_h = AllChem.AddHs(mol, addCoords=True)
    setups = MoleculePreparation().prepare(mol_h)
    if not setups:
        raise RuntimeError("Meeko produced no ligand setup")
    pdbqt, ok, detail = PDBQTWriterLegacy.write_string(setups[0])
    if not ok or not pdbqt:
        raise RuntimeError(f"Meeko PDBQT writer failed: {detail}")
    return f"REMARK lig_id {lig_id}\n" + pdbqt


def _write_uncharge_redock_ligand() -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromMol2File(str(INPUTS / "crystal_ligand.mol2"), sanitize=True, removeHs=False)
    if mol is None:
        raise RuntimeError("RDKit could not read crystal_ligand.mol2")
    before = _formal_charge(mol)
    variant = rdMolStandardize.Uncharger().uncharge(mol)
    after = _formal_charge(variant)
    out = VARIANTS / "ligands" / "cocrystal_2znp_uncharge.pdbqt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_mol_to_pdbqt(variant, "cocrystal_2znp"), encoding="utf-8")
    return {
        "variant_id": "redock_ligand_uncharge",
        "kind": "ligand_charge_protonation",
        "redock_ligand_pdbqt": str(out),
        "original_formal_charge": before,
        "variant_formal_charge": after,
    }


def _run_obabel(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("obabel")
    if not exe:
        return {"status": "unavailable", "reason": "obabel not on PATH", "source": str(src), "pdbqt": str(dst)}
    result = subprocess.run(
        [exe, str(src), "-O", str(dst)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "status": "ok" if result.returncode == 0 and dst.exists() else "failed",
        "source": str(src),
        "pdbqt": str(dst),
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-1000:],
    }


def _write_receptor_variants() -> list[dict[str, Any]]:
    specs = [
        ("receptor_fixed_clean_obabel", INPUTS / "receptor_fixed_clean.pdb"),
        ("receptor_prep_obabel", INPUTS / "receptor_prep.pdb"),
        ("receptor_raw_obabel", INPUTS / "receptor.pdb"),
    ]
    rows = []
    for variant_id, src in specs:
        rows.append({"variant_id": variant_id, "kind": "receptor_prep", **_run_obabel(src, VARIANTS / "receptors" / f"{variant_id}.pdbqt")})
    return rows

def _conformer_qc(mol_h) -> str:
    import math

    try:
        conf = mol_h.GetConformer()
    except Exception:
        return "no_conformer"
    pos = [conf.GetAtomPosition(i) for i in range(conf.GetNumAtoms())]
    for p in pos:
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            return "non_finite_coords"
    excluded = set()
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = pos[i].Distance(pos[j])
        if dist < 0.5 or dist > 3.0:
            return "bad_bond_length"
        excluded.add((min(i, j), max(i, j)))
    for atom in mol_h.GetAtoms():
        nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        for a in range(len(nbrs)):
            for c in range(a + 1, len(nbrs)):
                excluded.add((min(nbrs[a], nbrs[c]), max(nbrs[a], nbrs[c])))
    heavy = [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() > 1]
    for a in range(len(heavy)):
        for c in range(a + 1, len(heavy)):
            i, j = heavy[a], heavy[c]
            if (min(i, j), max(i, j)) in excluded:
                continue
            if pos[i].Distance(pos[j]) < 0.9:
                return "steric_clash"
    return ""


def _mol_h_to_pdbqt(mol_h, lig_id: str) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol_h)
    if not setups:
        raise RuntimeError("Meeko produced no ligand setup")
    pdbqt, ok, detail = PDBQTWriterLegacy.write_string(setups[0])
    if not ok or not pdbqt:
        raise RuntimeError(f"Meeko PDBQT writer failed: {detail}")
    clean = "\n".join(ln for ln in pdbqt.splitlines() if ln.strip() != "END")
    return f"REMARK lig_id {lig_id}\n" + clean + "\n"


def _write_local_mmff_panel() -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    src = PANEL / "audit_panel.csv"
    out_dir = VARIANTS / "ligands"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "audit_panel_local_mmff200.csv"
    out_index = out_dir / "audit_panel_local_mmff200_index.csv"
    out_pdbqt = out_dir / "audit_panel_local_mmff200.pdbqt"

    with src.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        base_fields = list(rows[0].keys()) if rows else []

    extra_fields = ["original_ligand", "input_smiles", "conformer_qc", "prep_note", "mmff_status"]
    fieldnames = list(dict.fromkeys(base_fields + extra_fields))
    index_fields = ["ligand", "name", "smiles", "input_smiles", "conformer_qc", "prep_note", "mmff_status"]
    written = 0
    failed = []
    csv_rows = []
    index_rows = []
    pdbqt_blocks = []

    for idx, row in enumerate(rows):
        smiles = row.get("smiles", "")
        lig_id = f"local_mmff_lig_{idx:05d}"
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError("invalid_smiles")
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == -1:
                raise RuntimeError("embed_failed")
            mmff_status = AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94", maxIters=200)
            qc = _conformer_qc(mol_h)
            pdbqt_blocks.append(_mol_h_to_pdbqt(mol_h, lig_id))
        except Exception as exc:
            failed.append({"audit_id": row.get("audit_id", ""), "smiles": smiles, "reason": str(exc)[:240]})
            continue

        out_row = dict(row)
        out_row["original_ligand"] = row.get("ligand", "")
        out_row["ligand"] = lig_id
        out_row["input_smiles"] = smiles
        out_row["conformer_qc"] = qc
        out_row["prep_note"] = "local_etkdg_seed42_mmff94_200"
        out_row["mmff_status"] = str(mmff_status)
        csv_rows.append(out_row)
        index_rows.append({k: out_row.get(k, "") for k in index_fields})
        written += 1

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    with out_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(index_rows)
    out_pdbqt.write_text("\n".join(pdbqt_blocks), encoding="utf-8")
    fail_csv = out_dir / "audit_panel_local_mmff200_failed.csv"
    if failed:
        with fail_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audit_id", "smiles", "reason"])
            writer.writeheader()
            writer.writerows(failed)

    return {
        "variant_id": "ligprep_local_mmff200",
        "kind": "ligand_conformer_prep",
        "panel_csv": str(out_csv),
        "panel_index_csv": str(out_index),
        "panel_pdbqt": str(out_pdbqt),
        "input_rows": len(rows),
        "prepared_rows": written,
        "failed_rows": len(failed),
        "failed_csv": str(fail_csv) if failed else "",
    }

def main() -> None:
    VARIANTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "updated_at": _now(),
        "purpose": "PPARD prep/protonation/charge audit variants; same box/search settings as current baseline.",
        "ligand_variants": [_write_uncharge_panel(), _write_uncharge_redock_ligand(), _write_local_mmff_panel()],
        "receptor_variants": _write_receptor_variants(),
    }
    out = VARIANTS / "prep_variant_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()