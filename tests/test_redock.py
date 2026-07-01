from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from ezscreen.benchmark.redock import extract_cocrystal_mol, symmetry_rmsd  # noqa: E402


def _embedded_benzene():
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(m, randomSeed=42)
    return m


def test_symmetry_rmsd_identical_is_zero():
    m = _embedded_benzene()
    assert symmetry_rmsd(Chem.Mol(m), Chem.Mol(m)) == pytest.approx(0.0, abs=1e-6)


def test_symmetry_rmsd_translated_is_nonzero():
    m = _embedded_benzene()
    probe = Chem.Mol(m)
    conf = probe.GetConformer()
    for i in range(conf.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + 5.0, p.y, p.z))
    # a rigid 5 Å translation is not re-aligned away — RMSD is the shift
    assert symmetry_rmsd(m, probe) == pytest.approx(5.0, abs=0.05)


def _het(serial, name, resn, chain, resseq, x, y, z, elem):
    line = list(" " * 80)
    line[0:6] = "HETATM"
    line[6:11] = f"{serial:>5}"
    line[12:16] = f"{name:<4}"
    line[17:20] = f"{resn:>3}"
    line[21] = chain
    line[22:26] = f"{resseq:>4}"
    line[30:38] = f"{x:8.3f}"
    line[38:46] = f"{y:8.3f}"
    line[46:54] = f"{z:8.3f}"
    line[76:78] = f"{elem:>2}"
    return "".join(line)


def test_extract_cocrystal_mol_selects_the_right_residue(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text("\n".join([
        # a protein atom and a water — must be ignored
        "ATOM      1  CA  ALA A  10      0.000   0.000   0.000  1.00  0.00           C",
        _het(2, "O", "HOH", "A", 700, 9.0, 9.0, 9.0, "O"),
        # the target co-crystal ligand LIG/A/500 — three bonded heavy atoms
        _het(3, "C1", "LIG", "A", 500, 0.0, 0.0, 0.0, "C"),
        _het(4, "C2", "LIG", "A", 500, 1.5, 0.0, 0.0, "C"),
        _het(5, "O1", "LIG", "A", 500, 2.9, 0.0, 0.0, "O"),
        "END",
    ]) + "\n")

    mol = extract_cocrystal_mol(pdb, "LIG", "A", 500)
    assert mol is not None
    assert mol.GetNumAtoms() == 3  # only the LIG residue's atoms

    assert extract_cocrystal_mol(pdb, "LIG", "A", 999) is None  # no such residue
