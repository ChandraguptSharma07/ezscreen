from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from ezscreen.prep.ligands import _conformer_qc, _embed_3d

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _two_carbons(dist: float, bonded: bool) -> Chem.Mol:
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(6))
    rw.AddAtom(Chem.Atom(6))
    if bonded:
        rw.AddBond(0, 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(dist, 0.0, 0.0))
    mol.AddConformer(conf)
    return mol


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


def test_conformer_qc_passes_clean_molecule():
    mol = _embed_3d(Chem.MolFromSmiles(IBUPROFEN), force_field="MMFF94")
    assert _conformer_qc(mol) is None


def test_conformer_qc_flags_steric_clash():
    # two non-bonded carbons overlapping → clash; no bonds means no bond-length flag
    assert _conformer_qc(_two_carbons(0.5, bonded=False)) == "steric_clash"
    # same atoms far apart → clean
    assert _conformer_qc(_two_carbons(2.0, bonded=False)) is None


def test_conformer_qc_flags_bad_bond_length():
    assert _conformer_qc(_two_carbons(4.0, bonded=True)) == "bad_bond_length"


def test_conformer_qc_flags_non_finite_coords():
    mol = _two_carbons(2.0, bonded=True)
    mol.GetConformer().SetAtomPosition(1, Point3D(float("inf"), 0.0, 0.0))
    assert _conformer_qc(mol) == "non_finite_coords"


def test_prep_ligands_expands_enumerated_variants(tmp_path, monkeypatch):
    import ezscreen.prep.enumerate as enum
    from ezscreen.prep.ligands import prep_ligands

    # stub enumeration: one acid → two protomers, so prep should embed both
    monkeypatch.setattr(
        enum, "enumerate_variants",
        lambda smi, *a, **k: ["CC(=O)O", "CC(=O)[O-]"],
    )

    src = tmp_path / "in.smi"
    src.write_text("CC(=O)O\tacid\n")
    out = tmp_path / "shards"
    opts = {"enabled": True, "protonation": True, "tautomers": True,
            "stereo": False, "ring": False, "max_variants": 4}
    result = prep_ligands(input_path=src, output_dir=out, enumerate_opts=opts)

    assert result["report"]["enumeration_enabled"] is True
    assert result["report"]["variants_generated"] == 2
    assert result["report"]["prep_passed"] == 2
