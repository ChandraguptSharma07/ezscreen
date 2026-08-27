from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from ezscreen.errors import LigandPrepError
from ezscreen.prep import ligands as ligand_prep
from ezscreen.prep.ligands import _conformer_qc, _embed_3d

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _write_sdf(path, smiles: str, name: str = "ligand") -> None:
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


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


def test_prep_ligands_reports_when_protonation_is_not_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_prep, "_get_scrubber", lambda: None)
    smi = tmp_path / "in.smi"
    smi.write_text(f"{IBUPROFEN}\tibuprofen\n")

    result = ligand_prep.prep_ligands(
        input_path=smi,
        output_dir=tmp_path / "shards",
        enumerate_opts={"enabled": False},
    )

    assert result["report"]["protonation_status"] == (
        "not_applied_input_state_preserved"
    )
    assert result["report"]["protonation_method"] is None


def test_generalized_normalization_removes_only_recognized_counterion():
    mol = Chem.MolFromSmiles("CC(=O)[O-].[Na+]")

    normalized, record = ligand_prep._normalize_ligand(mol, "generalized_v1")

    assert Chem.MolToSmiles(normalized) == "CC(=O)[O-]"
    assert record["status"] == "normalized"
    assert record["removed_counterions"] == ["[Na+]"]
    assert record["protonation_uncertainty"] == "not_assessed"


def test_generalized_normalization_rejects_ambiguous_organic_fragments():
    mol = Chem.MolFromSmiles("CCO.CN")

    normalized, record = ligand_prep._normalize_ligand(mol, "generalized_v1")

    assert normalized is None
    assert record["status"] == "rejected"
    assert record["reason"] == "ambiguous_fragments"


def test_generalized_normalization_reports_unspecified_stereo():
    unspecified = Chem.MolFromSmiles("CC(O)C(=O)O")
    specified = Chem.MolFromSmiles("C[C@H](O)C(=O)O")

    _, unspecified_record = ligand_prep._normalize_ligand(unspecified, "generalized_v1")
    _, specified_record = ligand_prep._normalize_ligand(specified, "generalized_v1")

    assert unspecified_record["unspecified_stereocenters"] == 1
    assert specified_record["unspecified_stereocenters"] == 0


def test_generalized_normalization_rejection_stays_in_denominator(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ligand_prep, "_get_scrubber", lambda: None)
    smi = tmp_path / "in.smi"
    smi.write_text("CCO.CN\tmixture\nCC(=O)[O-].[Na+]\tacetate\n")

    result = ligand_prep.prep_ligands(
        input_path=smi,
        output_dir=tmp_path / "shards",
        enumerate_opts={"enabled": False},
        normalization_profile="generalized_v1",
    )

    report = result["report"]
    assert report["total_input"] == 2
    assert report["prep_passed"] == 1
    assert report["prep_failed"] == 1
    assert report["prep_failures"]["ambiguous_fragments"] == 1
    assert report["normalization_profile"] == "generalized_v1"
    assert report["counterions_removed"] == 1
    assert report["normalization_rejected"] == 1

    index = (tmp_path / "shards" / "index.csv").read_text()
    assert "source_smiles" in index
    assert "CC(=O)[O-].[Na+]" in index
    assert "[Na+]" in index


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
        enum,
        "enumerate_variants",
        lambda smi, *a, **k: ["CC(=O)O", "CC(=O)[O-]"],
    )

    src = tmp_path / "in.smi"
    src.write_text("CC(=O)O\tacid\n")
    out = tmp_path / "shards"
    opts = {
        "enabled": True,
        "protonation": True,
        "tautomers": True,
        "stereo": False,
        "ring": False,
        "max_variants": 4,
    }
    result = prep_ligands(input_path=src, output_dir=out, enumerate_opts=opts)

    assert result["report"]["enumeration_enabled"] is True
    assert result["report"]["variants_generated"] == 2
    assert result["report"]["prep_passed"] == 2


def test_scan_input_accepts_openbabel_ligand_formats(tmp_path):
    for name in ("ligand.sdf", "ligand.smi", "ligand.mol2", "ligand.mol", "ligand.pdb"):
        (tmp_path / name).write_text("placeholder")

    suffixes = {path.suffix for path in ligand_prep.scan_input(tmp_path)}

    assert {".sdf", ".smi", ".mol2", ".mol", ".pdb"} <= suffixes


def test_openbabel_formats_are_kept_for_kaggle_raw_sharding(tmp_path, monkeypatch):
    src_dir = tmp_path / "inputs"
    src_dir.mkdir()
    ref_sdf = src_dir / "reference.sdf"
    _write_sdf(ref_sdf, "CCO", "ethanol")
    (src_dir / "from_mol2.mol2").write_text("@<TRIPOS>MOLECULE\nethanol\n")
    (src_dir / "from_pdb.pdb").write_text("HEADER    ETHANOL\n")

    def fail_if_local_convert(*_args, **_kwargs):
        raise AssertionError("Kaggle raw sharding should not require local Open Babel")

    monkeypatch.setattr(ligand_prep, "_convert_ligand_to_sdf", fail_if_local_convert)

    result = ligand_prep.shard_raw(src_dir, tmp_path / "out", shard_size=10)
    suffixes = sorted(path.suffix for path in result["shard_paths"])

    assert suffixes == [".mol2", ".pdb", ".smi"]
    assert result["report"]["input_files"] == 3
    assert result["report"]["backend_input_files"] == 2
    assert result["report"]["converted_input_files"] == 0
    assert result["report"]["total_input"] == 3
    assert (tmp_path / "out" / "raw_0000_from_mol2.mol2").exists()
    assert (tmp_path / "out" / "raw_0001_from_pdb.pdb").exists()


def test_openbabel_formats_are_converted_for_local_prep_loading(tmp_path, monkeypatch):
    src_dir = tmp_path / "inputs"
    src_dir.mkdir()
    ref_sdf = src_dir / "reference.sdf"
    _write_sdf(ref_sdf, "CCO", "ethanol")
    (src_dir / "from_mol2.mol2").write_text("@<TRIPOS>MOLECULE\nethanol\n")
    (src_dir / "from_pdb.pdb").write_text("HEADER    ETHANOL\n")

    converted: list[str] = []

    def fake_convert(src, output_dir, index):
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / f"{index:04d}_{src.stem}.sdf"
        dst.write_text(ref_sdf.read_text())
        converted.append(src.suffix)
        return dst

    monkeypatch.setattr(ligand_prep, "_convert_ligand_to_sdf", fake_convert)

    normalised = ligand_prep._normalise_input_files(
        ligand_prep.scan_input(src_dir), tmp_path / "out"
    )
    smiles = []
    for path in normalised:
        for mol in ligand_prep._load_supplier(path):
            smiles.append(Chem.MolToSmiles(mol))

    assert sorted(converted) == [".mol2", ".pdb"]
    assert smiles == [Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))] * 3


def test_openbabel_formats_report_missing_obabel(tmp_path, monkeypatch):
    monkeypatch.setattr(ligand_prep.shutil, "which", lambda _name: None)
    src = tmp_path / "ligand.mol2"
    src.write_text("@<TRIPOS>MOLECULE\n")

    with pytest.raises(LigandPrepError, match="Open Babel"):
        ligand_prep._convert_ligand_to_sdf(src, tmp_path / "out", 0)
