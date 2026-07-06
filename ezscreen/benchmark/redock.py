from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Redock the co-crystal ligand into the detected box and measure how well the
# docked pose reproduces the crystal pose (symmetry-aware RMSD, no re-alignment).
# A low RMSD means the box + engine reproduce a known binding mode, so the box is
# trustworthy; a high RMSD (> ~2 Å) warns the box may be off. Everything is
# fail-soft: any missing dependency or parse error returns None, never raising.

RMSD_WARN_THRESHOLD = 2.0
_CCD_CACHE_DIR = Path.home() / ".ezscreen" / "cache" / "ccd"
_CCD_IDEAL_SDF_URL = "https://files.rcsb.org/ligands/view/{resname}_ideal.sdf"
_CCD_CIF_URL = "https://files.rcsb.org/ligands/view/{resname}.cif"


def symmetry_rmsd(ref, probe) -> float | None:
    """Symmetry-aware heavy-atom RMSD between two poses, without re-aligning them.

    Uses RDKit CalcRMS (best atom mapping over molecular symmetry) on the current
    coordinates, so a pose that sits away from the crystal scores high — that's the
    point for box validation. Returns None if the molecules can't be compared.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolAlign

        ref_h = Chem.RemoveHs(ref)
        probe_h = Chem.RemoveHs(probe)
        return float(rdMolAlign.CalcRMS(probe_h, ref_h))
    except Exception:
        return None


def extract_cocrystal_mol(pdb_path: Path, resname: str, chain: str, resseq: int):
    """Build an RDKit mol (with crystal coords) for one co-crystal ligand residue."""
    try:
        from rdkit import Chem
    except Exception:
        return None

    ccd = _ccd_reference_mol(pdb_path, resname, chain, resseq)
    if ccd is not None:
        return ccd

    lines = []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith(("HETATM", "ATOM")):
            continue
        try:
            if (
                line[17:20].strip() == resname
                and line[21:22].strip() == chain
                and int(line[22:26]) == resseq
            ):
                lines.append(line)
        except (ValueError, IndexError):
            continue
    if not lines:
        return None

    block = "\n".join(lines) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    return mol


def _ccd_cache_path(resname: str, suffix: str) -> Path:
    _CCD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CCD_CACHE_DIR / f"{resname.upper()}{suffix}"


def _fetch_ccd_text(resname: str, *, artifact: str) -> str | None:
    try:
        import requests
    except Exception:
        return None

    resname = resname.strip().upper()
    if artifact == "ideal_sdf":
        url = _CCD_IDEAL_SDF_URL.format(resname=resname)
        cache_path = _ccd_cache_path(resname, "_ideal.sdf")
    elif artifact == "cif":
        url = _CCD_CIF_URL.format(resname=resname)
        cache_path = _ccd_cache_path(resname, ".cif")
    else:
        return None

    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass

    try:
        response = requests.get(url, timeout=20)
        if not response.ok:
            return None
        cache_path.write_text(response.text, encoding="utf-8")
        return response.text
    except Exception:
        return None


def _parse_cif_loop_rows(cif_text: str, prefix: str) -> list[dict[str, str]]:
    lines = cif_text.splitlines()
    rows: list[dict[str, str]] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "loop_":
            idx += 1
            continue
        idx += 1
        headers: list[str] = []
        while idx < len(lines) and lines[idx].startswith("_"):
            headers.append(lines[idx].strip())
            idx += 1
        if not headers or not all(header.startswith(prefix) for header in headers):
            while idx < len(lines):
                raw = lines[idx].strip()
                if not raw or raw == "#":
                    idx += 1
                    break
                if raw == "loop_" or raw.startswith("_"):
                    break
                idx += 1
            continue
        short_headers = [header.removeprefix(prefix) for header in headers]
        while idx < len(lines):
            raw = lines[idx].strip()
            if not raw or raw == "#":
                idx += 1
                break
            if raw == "loop_" or raw.startswith("_"):
                break
            parts = raw.split()
            if len(parts) != len(short_headers):
                idx += 1
                continue
            rows.append(dict(zip(short_headers, parts, strict=False)))
            idx += 1
        if rows:
            return rows
    return rows


def _pdb_residue_atom_coords(
    pdb_path: Path, resname: str, chain: str, resseq: int
) -> dict[str, tuple[float, float, float]]:
    coords: dict[str, tuple[float, float, float]] = {}
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith(("HETATM", "ATOM")):
            continue
        try:
            if (
                line[17:20].strip() == resname
                and line[21:22].strip() == chain
                and int(line[22:26]) == resseq
            ):
                atom_id = line[12:16].strip()
                coords[atom_id] = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
        except (ValueError, IndexError):
            continue
    return coords


def _map_mol_atoms_to_ccd_ids(
    mol, atom_rows: list[dict[str, str]]
) -> dict[int, str] | None:
    conf = mol.GetConformer()
    unmatched = [
        {
            "atom_id": row["atom_id"],
            "type_symbol": row["type_symbol"].upper(),
            "coords": (
                float(row["pdbx_model_Cartn_x_ideal"]),
                float(row["pdbx_model_Cartn_y_ideal"]),
                float(row["pdbx_model_Cartn_z_ideal"]),
            ),
        }
        for row in atom_rows
        if row.get("type_symbol", "").upper() != "H"
    ]
    atom_map: dict[int, str] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        symbol = atom.GetSymbol().upper()
        candidates = [row for row in unmatched if row["type_symbol"] == symbol]
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda row: (
                (pos.x - row["coords"][0]) ** 2
                + (pos.y - row["coords"][1]) ** 2
                + (pos.z - row["coords"][2]) ** 2
            ),
        )
        dist = math.sqrt(
            (pos.x - best["coords"][0]) ** 2
            + (pos.y - best["coords"][1]) ** 2
            + (pos.z - best["coords"][2]) ** 2
        )
        if dist > 0.25:
            return None
        atom_map[idx] = best["atom_id"]
        unmatched.remove(best)
    return atom_map


def _ccd_reference_mol(pdb_path: Path, resname: str, chain: str, resseq: int):
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except Exception:
        return None

    residue_coords = _pdb_residue_atom_coords(pdb_path, resname, chain, resseq)
    if not residue_coords:
        return None

    cif_text = _fetch_ccd_text(resname, artifact="cif")
    sdf_text = _fetch_ccd_text(resname, artifact="ideal_sdf")
    if not cif_text or not sdf_text:
        return None

    atom_rows = _parse_cif_loop_rows(cif_text, "_chem_comp_atom.")
    if not atom_rows:
        return None

    ideal = Chem.MolFromMolBlock(sdf_text, sanitize=True, removeHs=False)
    if ideal is None:
        return None
    ideal = Chem.RemoveHs(ideal)
    atom_map = _map_mol_atoms_to_ccd_ids(ideal, atom_rows)
    if atom_map is None:
        return None

    conf = Chem.Conformer(ideal.GetNumAtoms())
    conf.Set3D(True)
    for atom in ideal.GetAtoms():
        atom_id = atom_map.get(atom.GetIdx())
        if atom_id is None or atom_id not in residue_coords:
            return None
        x, y, z = residue_coords[atom_id]
        conf.SetAtomPosition(atom.GetIdx(), Point3D(x, y, z))
    ideal.RemoveAllConformers()
    ideal.AddConformer(conf, assignId=True)
    return ideal


def _mol_to_pdbqt(mol) -> str | None:
    from rdkit.Chem import AllChem

    from ezscreen.prep.ligands import _to_pdbqt

    try:
        mol_h = AllChem.AddHs(mol, addCoords=True)
    except Exception:
        mol_h = mol
    return _to_pdbqt(mol_h)


def _docked_pose_mols(out_pdbqt: Path) -> list[Any]:
    """Read docked Vina poses from a PDBQT into RDKit mol objects."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate

        pm = PDBQTMolecule.from_file(str(out_pdbqt), skip_typing=True)
        result = RDKitMolCreate.from_pdbqt_mol(pm)
        mols = result[0] if isinstance(result, tuple) else result
        mols = mols if isinstance(mols, list) else [mols]
        return [m for m in mols if m is not None]
    except Exception:
        pass
    return []


def _failure_result(
    ligand: dict[str, Any],
    reason: str,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "rmsd": None,
        "reliable": False,
        "reference_ligand": ligand.get("resname", ""),
        "reference_chain": ligand.get("chain", ""),
        "reference_resseq": ligand.get("resseq"),
        "status": "skipped",
        "reason": reason,
        "detail": detail,
    }


def redock_cocrystal(
    receptor_pdbqt: Path,
    pdb_path: Path,
    ligand: dict[str, Any],
    box: dict[str, Any],
    work_dir: Path,
    exhaustiveness: int = 8,
    num_modes: int = 5,
) -> dict[str, Any]:
    """Redock the co-crystal ligand into the box; return {rmsd, reliable, reference_ligand}.

    Local, single-ligand AutoDock Vina run reusing the local backend. Fail-soft:
    returns None if the ligand can't be extracted/prepped/docked.
    """
    import tempfile

    try:
        resseq = int(ligand.get("resseq", 0))
    except (TypeError, ValueError):
        return _failure_result(
            ligand, "invalid_ligand_resseq", str(ligand.get("resseq"))
        )

    ref = extract_cocrystal_mol(
        pdb_path,
        ligand.get("resname", ""),
        ligand.get("chain", ""),
        resseq,
    )
    if ref is None:
        return _failure_result(
            ligand,
            "extract_failed",
            f"{ligand.get('resname', '')}:{ligand.get('chain', '')}:{resseq}",
        )

    lig_pdbqt_text = _mol_to_pdbqt(ref)
    if not lig_pdbqt_text:
        return _failure_result(ligand, "ligand_pdbqt_failed")

    try:
        from ezscreen.backends.local.runner import _run_vina
        from ezscreen.backends.local.vina_binary import get_vina_binary

        vina = get_vina_binary()
    except Exception as exc:
        return _failure_result(ligand, "vina_setup_failed", str(exc))

    with tempfile.TemporaryDirectory(dir=str(work_dir)) as td:
        tdp = Path(td)
        lig_in = tdp / "cocrystal.pdbqt"
        lig_in.write_text(lig_pdbqt_text)
        out_pdbqt = tdp / "cocrystal_out.pdbqt"
        text = _run_vina(
            vina,
            Path(receptor_pdbqt),
            lig_in,
            out_pdbqt,
            box["center"],
            box["size"],
            exhaustiveness,
            num_modes=max(1, int(num_modes)),
            ligand_name=ligand.get("resname", "cocrystal"),
        )
        if text is None or not out_pdbqt.exists():
            detail = None if text is None else text.strip()[-400:]
            return _failure_result(ligand, "vina_docking_failed", detail)
        docked_mols = _docked_pose_mols(out_pdbqt)

    if not docked_mols:
        return _failure_result(ligand, "docked_pose_parse_failed")
    mode_rmsds: list[float | None] = [
        symmetry_rmsd(ref, docked) for docked in docked_mols
    ]
    scored_modes = [
        (idx, rmsd) for idx, rmsd in enumerate(mode_rmsds, start=1) if rmsd is not None
    ]
    if not scored_modes:
        return _failure_result(ligand, "rmsd_failed")
    best_mode_rank, best_rmsd = min(scored_modes, key=lambda item: item[1])
    return {
        "rmsd": round(best_rmsd, 3),
        "reliable": best_rmsd <= RMSD_WARN_THRESHOLD,
        "reference_ligand": ligand.get("resname", ""),
        "reference_chain": ligand.get("chain", ""),
        "reference_resseq": resseq,
        "best_mode_rank": best_mode_rank,
        "num_modes_evaluated": len(scored_modes),
        "mode_rmsds": [None if rmsd is None else round(rmsd, 3) for rmsd in mode_rmsds],
        "status": "ok",
        "reason": None,
        "detail": None,
    }
