from __future__ import annotations

import pytest
from rdkit import Chem

from ezscreen.prep.ligands import _embed_3d

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.mark.parametrize("force_field", ["MMFF94", "MMFF94s", "UFF"])
def test_embed_3d_produces_finite_coords(force_field):
    mol = Chem.MolFromSmiles(IBUPROFEN)
    mol_3d = _embed_3d(mol, mmff_max_iters=200, force_field=force_field)
    assert mol_3d is not None
    conf = mol_3d.GetConformer()
    assert conf.GetNumAtoms() == mol_3d.GetNumAtoms()
    for i in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        for c in (pos.x, pos.y, pos.z):
            assert c == c  # not NaN
            assert abs(c) != float("inf")


def test_embed_3d_unknown_force_field_falls_back_to_mmff():
    mol = Chem.MolFromSmiles(IBUPROFEN)
    # an unrecognised name should still embed (MMFF94 branch), not raise
    mol_3d = _embed_3d(mol, mmff_max_iters=50, force_field="bogus")
    assert mol_3d is not None
    assert mol_3d.GetNumConformers() == 1


def test_prep_ligands_records_force_field_override(tmp_path):
    from ezscreen.prep.ligands import prep_ligands

    smi = tmp_path / "in.smi"
    smi.write_text(f"{IBUPROFEN}\tibuprofen\n")
    out = tmp_path / "shards"
    # explicit per-run override should win over the config default and be reported
    result = prep_ligands(input_path=smi, output_dir=out, force_field="UFF")
    assert result["report"]["force_field"] == "UFF"
    assert result["report"]["prep_passed"] == 1
