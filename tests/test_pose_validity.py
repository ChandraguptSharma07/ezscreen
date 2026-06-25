from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ezscreen.results.pose_validity import check_poses

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _translate(mol: Chem.Mol, dx: float, dy: float, dz: float) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + dx, p.y + dy, p.z + dz))
    return out


def _pdb_atom(serial: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:>5}  CA  ALA A{serial:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_protein(path: Path, centre: tuple[float, float, float]) -> None:
    cx, cy, cz = centre
    lines = ["REMARK synthetic test protein\n"]
    serial = 1
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                lines.append(
                    _pdb_atom(serial, cx + 2.5 * i, cy + 2.5 * j, cz + 2.5 * k)
                )
                serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_poses(path: Path, mols: dict[str, Chem.Mol]) -> None:
    writer = Chem.SDWriter(str(path))
    for lig_id, mol in mols.items():
        mol.SetProp("lig_id", lig_id)
        writer.write(mol)
    writer.close()


def test_missing_inputs_return_none(tmp_path):
    # No posebusters needed: missing files always short-circuit to None.
    assert check_poses(tmp_path / "nope.sdf", tmp_path / "nope.pdb") is None


def test_valid_and_clashing_poses(tmp_path):
    pytest.importorskip("posebusters")

    receptor = tmp_path / "receptor_prep.pdb"
    _write_protein(receptor, centre=(0.0, 0.0, 0.0))

    base = _embed(IBUPROFEN)
    # Valid pose sits clear of the protein block; clashing pose overlaps its centre.
    valid_pose = _translate(base, 0.0, 0.0, 8.0)
    clash_pose = _translate(base, 0.0, 0.0, 0.0)

    poses = tmp_path / "poses.sdf"
    _write_poses(poses, {"valid_lig": valid_pose, "clash_lig": clash_pose})

    result = check_poses(poses, receptor)
    assert result is not None
    assert set(result) == {"valid_lig", "clash_lig"}

    assert result["valid_lig"]["passed"] is True
    assert result["valid_lig"]["failed_checks"] == []

    assert result["clash_lig"]["passed"] is False
    assert result["clash_lig"]["failed_checks"]
