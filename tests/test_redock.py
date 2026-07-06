from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from ezscreen.benchmark.redock import (  # noqa: E402
    extract_cocrystal_mol,
    redock_cocrystal,
    symmetry_rmsd,
)


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
    pdb.write_text(
        "\n".join(
            [
                # a protein atom and a water — must be ignored
                "ATOM      1  CA  ALA A  10      0.000   0.000   0.000  1.00  0.00           C",
                _het(2, "O", "HOH", "A", 700, 9.0, 9.0, 9.0, "O"),
                # the target co-crystal ligand LIG/A/500 — three bonded heavy atoms
                _het(3, "C1", "LIG", "A", 500, 0.0, 0.0, 0.0, "C"),
                _het(4, "C2", "LIG", "A", 500, 1.5, 0.0, 0.0, "C"),
                _het(5, "O1", "LIG", "A", 500, 2.9, 0.0, 0.0, "O"),
                "END",
            ]
        )
        + "\n"
    )

    mol = extract_cocrystal_mol(pdb, "LIG", "A", 500)
    assert mol is not None
    assert mol.GetNumAtoms() == 3  # only the LIG residue's atoms

    assert extract_cocrystal_mol(pdb, "LIG", "A", 999) is None  # no such residue


def test_extract_cocrystal_mol_uses_ccd_chemistry_when_available(tmp_path, monkeypatch):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "\n".join(
            [
                _het(3, "C1", "LIG", "A", 500, 0.0, 0.0, 0.0, "C"),
                _het(4, "C2", "LIG", "A", 500, 1.5, 0.0, 0.0, "C"),
                _het(5, "O1", "LIG", "A", 500, 2.9, 0.0, 0.0, "O"),
                "CONECT    3    4",
                "CONECT    4    3    5",
                "CONECT    5    4",
                "END",
            ]
        )
        + "\n"
    )

    cif_text = "\n".join(
        [
            "data_LIG",
            "loop_",
            "_chem_comp_atom.comp_id",
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.alt_atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_align",
            "_chem_comp_atom.pdbx_aromatic_flag",
            "_chem_comp_atom.pdbx_leaving_atom_flag",
            "_chem_comp_atom.pdbx_stereo_config",
            "_chem_comp_atom.model_Cartn_x",
            "_chem_comp_atom.model_Cartn_y",
            "_chem_comp_atom.model_Cartn_z",
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
            "_chem_comp_atom.pdbx_component_atom_id",
            "_chem_comp_atom.pdbx_component_comp_id",
            "_chem_comp_atom.pdbx_ordinal",
            "LIG C1 C1 C 0 1 N N N 0.0 0.0 0.0 0.0 0.0 0.0 C1 LIG 1",
            "LIG C2 C2 C 0 1 N N N 1.2 0.0 0.0 1.2 0.0 0.0 C2 LIG 2",
            "LIG O1 O1 O 0 1 N N N 2.4 0.0 0.0 2.4 0.0 0.0 O1 LIG 3",
            "#",
        ]
    )
    sdf_text = "\n".join(
        [
            "LIG",
            "  TEST",
            "",
            "  3  2  0  0  0  0  0  0  0  0999 V2000",
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
            "    1.2000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
            "    2.4000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
            "  1  2  1  0",
            "  2  3  2  0",
            "M  END",
            "$$$$",
        ]
    )

    monkeypatch.setattr(
        "ezscreen.benchmark.redock._fetch_ccd_text",
        lambda resname, artifact: cif_text if artifact == "cif" else sdf_text,
    )

    mol = extract_cocrystal_mol(pdb, "LIG", "A", 500)
    assert mol is not None
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == "CC=O"
    conf = mol.GetConformer()
    assert conf.GetAtomPosition(0).x == pytest.approx(0.0)
    assert conf.GetAtomPosition(1).x == pytest.approx(1.5)
    assert conf.GetAtomPosition(2).x == pytest.approx(2.9)


def test_redock_cocrystal_reports_extract_failure(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  10      0.000   0.000   0.000  1.00  0.00           C",
                _het(2, "O", "HOH", "A", 700, 9.0, 9.0, 9.0, "O"),
                "END",
            ]
        )
        + "\n"
    )

    result = redock_cocrystal(
        tmp_path / "missing_receptor.pdbqt",
        pdb,
        {"resname": "LIG", "chain": "A", "resseq": 500},
        {"center": [0.0, 0.0, 0.0], "size": [10.0, 10.0, 10.0]},
        tmp_path,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "extract_failed"
    assert result["reference_ligand"] == "LIG"
    assert result["reference_chain"] == "A"
    assert result["reference_resseq"] == 500
    assert result["rmsd"] is None


def test_redock_cocrystal_keeps_best_mode(tmp_path, monkeypatch):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "\n".join(
            [
                _het(3, "C1", "LIG", "A", 500, 0.0, 0.0, 0.0, "C"),
                _het(4, "C2", "LIG", "A", 500, 1.5, 0.0, 0.0, "C"),
                _het(5, "O1", "LIG", "A", 500, 2.9, 0.0, 0.0, "O"),
                "END",
            ]
        )
        + "\n"
    )

    ref = extract_cocrystal_mol(pdb, "LIG", "A", 500)
    assert ref is not None
    better = Chem.Mol(ref)
    worse = Chem.Mol(ref)
    worse_conf = worse.GetConformer()
    for idx in range(worse_conf.GetNumAtoms()):
        pos = worse_conf.GetAtomPosition(idx)
        worse_conf.SetAtomPosition(idx, Point3D(pos.x + 4.0, pos.y, pos.z))

    monkeypatch.setattr(
        "ezscreen.benchmark.redock._mol_to_pdbqt",
        lambda mol: "MODEL 1\nENDMDL\n",
    )
    monkeypatch.setattr(
        "ezscreen.benchmark.redock._docked_pose_mols",
        lambda out_pdbqt: [worse, better],
    )
    monkeypatch.setattr(
        "ezscreen.backends.local.vina_binary.get_vina_binary",
        lambda: "vina",
    )

    def _fake_run_vina(*args, **kwargs):
        out_pdbqt = args[3]
        out_pdbqt.write_text("MODEL 1\nENDMDL\n")
        return "ok"

    monkeypatch.setattr("ezscreen.backends.local.runner._run_vina", _fake_run_vina)

    result = redock_cocrystal(
        tmp_path / "receptor.pdbqt",
        pdb,
        {"resname": "LIG", "chain": "A", "resseq": 500},
        {"center": [0.0, 0.0, 0.0], "size": [10.0, 10.0, 10.0]},
        tmp_path,
        num_modes=5,
    )

    assert result["status"] == "ok"
    assert result["rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert result["best_mode_rank"] == 2
    assert result["num_modes_evaluated"] == 2
    assert result["mode_rmsds"] == [
        pytest.approx(4.0, abs=0.05),
        pytest.approx(0.0, abs=1e-6),
    ]
