from __future__ import annotations

from pathlib import Path
from typing import Any

# Redock the co-crystal ligand into the detected box and measure how well the
# docked pose reproduces the crystal pose (symmetry-aware RMSD, no re-alignment).
# A low RMSD means the box + engine reproduce a known binding mode, so the box is
# trustworthy; a high RMSD (> ~2 Å) warns the box may be off. Everything is
# fail-soft: any missing dependency or parse error returns None, never raising.

RMSD_WARN_THRESHOLD = 2.0


def symmetry_rmsd(ref, probe) -> float | None:
    """Symmetry-aware heavy-atom RMSD between two poses, without re-aligning them.

    Uses RDKit CalcRMS (best atom mapping over molecular symmetry) on the current
    coordinates, so a pose that sits away from the crystal scores high — that's the
    point for box validation. Returns None if the molecules can't be compared.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolAlign

        ref_h   = Chem.RemoveHs(ref)
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

    lines = []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith(("HETATM", "ATOM")):
            continue
        try:
            if (line[17:20].strip() == resname
                    and line[21:22].strip() == chain
                    and int(line[22:26]) == resseq):
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


def _mol_to_pdbqt(mol) -> str | None:
    from rdkit.Chem import AllChem

    from ezscreen.prep.ligands import _to_pdbqt
    try:
        mol_h = AllChem.AddHs(mol, addCoords=True)
    except Exception:
        mol_h = mol
    return _to_pdbqt(mol_h)


def _docked_pose_mol(out_pdbqt: Path):
    """Read the top docked pose from a Vina output PDBQT into an RDKit mol."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        pm = PDBQTMolecule.from_file(str(out_pdbqt), skip_typing=True)
        result = RDKitMolCreate.from_pdbqt_mol(pm)
        mols = result[0] if isinstance(result, tuple) else result
        mols = mols if isinstance(mols, list) else [mols]
        for m in mols:
            if m is not None:
                return m
    except Exception:
        pass
    return None


def redock_cocrystal(
    receptor_pdbqt: Path,
    pdb_path: Path,
    ligand: dict[str, Any],
    box: dict[str, Any],
    work_dir: Path,
    exhaustiveness: int = 8,
) -> dict[str, Any] | None:
    """Redock the co-crystal ligand into the box; return {rmsd, reliable, reference_ligand}.

    Local, single-ligand AutoDock Vina run reusing the local backend. Fail-soft:
    returns None if the ligand can't be extracted/prepped/docked.
    """
    import tempfile

    ref = extract_cocrystal_mol(
        pdb_path, ligand.get("resname", ""), ligand.get("chain", ""),
        int(ligand.get("resseq", 0)),
    )
    if ref is None:
        return None

    lig_pdbqt_text = _mol_to_pdbqt(ref)
    if not lig_pdbqt_text:
        return None

    try:
        from ezscreen.backends.local.runner import _run_vina
        from ezscreen.backends.local.vina_binary import get_vina_binary
        vina = get_vina_binary()
    except Exception:
        return None

    with tempfile.TemporaryDirectory(dir=str(work_dir)) as td:
        tdp = Path(td)
        lig_in = tdp / "cocrystal.pdbqt"
        lig_in.write_text(lig_pdbqt_text)
        out_pdbqt = tdp / "cocrystal_out.pdbqt"
        text = _run_vina(
            vina, Path(receptor_pdbqt), lig_in, out_pdbqt,
            box["center"], box["size"], exhaustiveness, num_modes=1,
            ligand_name=ligand.get("resname", "cocrystal"),
        )
        if text is None or not out_pdbqt.exists():
            return None
        docked = _docked_pose_mol(out_pdbqt)

    if docked is None:
        return None
    rmsd = symmetry_rmsd(ref, docked)
    if rmsd is None:
        return None
    return {
        "rmsd": round(rmsd, 3),
        "reliable": rmsd <= RMSD_WARN_THRESHOLD,
        "reference_ligand": ligand.get("resname", ""),
    }
