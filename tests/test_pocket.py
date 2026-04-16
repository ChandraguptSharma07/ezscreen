from __future__ import annotations

from pathlib import Path

import pytest

from ezscreen.pocket.detect import (
    _box_from_coords,
    box_from_cocrystal,
    box_from_residues,
    find_cocrystal_ligands,
)

# ---------------------------------------------------------------------------
# Minimal PDB fixtures
# ---------------------------------------------------------------------------

def _write_pdb(path: Path, lines: list[str]) -> Path:
    pdb = path / "receptor.pdb"
    pdb.write_text("\n".join(lines) + "\n")
    return pdb


def _atom_line(serial, name, resname, chain, resseq, x, y, z, record="ATOM  "):
    # Standard PDB column layout (1-indexed):
    # 1-6 record, 7-11 serial, 12 blank, 13-16 name, 17 altLoc, 18-20 resname,
    # 21 blank, 22 chain, 23-26 resseq, 27-30 blank, 31-38 x, 39-46 y, 47-54 z
    return (
        f"{record:<6}{serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def _make_ca_pdb(tmp_path: Path) -> Path:
    """Minimal PDB with three CA atoms in chain A at known positions."""
    lines = [
        _atom_line(1, "CA", "ALA", "A", 10, 1.0, 2.0, 3.0),
        _atom_line(2, "CA", "GLY", "A", 20, 5.0, 6.0, 7.0),
        _atom_line(3, "CA", "LEU", "A", 30, 3.0, 4.0, 5.0),
    ]
    return _write_pdb(tmp_path, lines)


def _make_hetatm_pdb(tmp_path: Path) -> Path:
    """Minimal PDB with a drug-like HETATM (8 atoms, not an additive)."""
    lines = [
        _atom_line(1, "CA",  "ALA", "A", 5, 0.0, 0.0, 0.0),
        _atom_line(2, "N1",  "LIG", "A", 1, 10.0, 10.0, 10.0, "HETATM"),
        _atom_line(3, "C1",  "LIG", "A", 1, 11.0, 10.0, 10.0, "HETATM"),
        _atom_line(4, "C2",  "LIG", "A", 1, 12.0, 10.0, 10.0, "HETATM"),
        _atom_line(5, "C3",  "LIG", "A", 1, 13.0, 10.0, 10.0, "HETATM"),
        _atom_line(6, "C4",  "LIG", "A", 1, 14.0, 10.0, 10.0, "HETATM"),
        _atom_line(7, "O1",  "LIG", "A", 1, 15.0, 10.0, 10.0, "HETATM"),
        _atom_line(8, "O2",  "LIG", "A", 1, 16.0, 10.0, 10.0, "HETATM"),
        _atom_line(9, "S1",  "LIG", "A", 1, 17.0, 10.0, 10.0, "HETATM"),
    ]
    return _write_pdb(tmp_path, lines)


def _make_additive_pdb(tmp_path: Path) -> Path:
    """HETATM residue that is a crystallographic additive (HOH) — should be filtered."""
    lines = [
        _atom_line(1, "O", "HOH", "A", 1, 5.0, 5.0, 5.0, "HETATM"),
        _atom_line(2, "O", "HOH", "A", 2, 5.0, 5.0, 6.0, "HETATM"),
        _atom_line(3, "O", "HOH", "A", 3, 5.0, 5.0, 7.0, "HETATM"),
        _atom_line(4, "O", "HOH", "A", 4, 5.0, 5.0, 8.0, "HETATM"),
        _atom_line(5, "O", "HOH", "A", 5, 5.0, 5.0, 9.0, "HETATM"),
        _atom_line(6, "O", "HOH", "A", 6, 5.0, 5.0, 10.0, "HETATM"),
    ]
    return _write_pdb(tmp_path, lines)


# ---------------------------------------------------------------------------
# _box_from_coords
# ---------------------------------------------------------------------------

def test_box_center_single_point():
    box = _box_from_coords([(5.0, 10.0, 15.0)], padding=0.0)
    assert box["center"] == [5.0, 10.0, 15.0]


def test_box_center_two_points():
    box = _box_from_coords([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)], padding=0.0)
    assert box["center"] == [5.0, 5.0, 5.0]


def test_box_size_includes_padding():
    box = _box_from_coords([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)], padding=5.0)
    assert box["size"] == [20.0, 20.0, 20.0]


def test_box_empty_coords_raises():
    with pytest.raises(ValueError):
        _box_from_coords([], padding=5.0)


def test_box_volume_computed():
    box = _box_from_coords([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)], padding=0.0)
    assert box["volume_angstrom3"] == pytest.approx(1000.0, abs=0.5)


# ---------------------------------------------------------------------------
# box_from_cocrystal
# ---------------------------------------------------------------------------

def test_box_from_cocrystal_sets_method():
    ligand = {
        "resname": "LIG", "chain": "A", "resseq": 1,
        "coords": [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)],
    }
    box = box_from_cocrystal(ligand, padding=5.0)
    assert box["method"] == "co_crystal"
    assert box["reference_ligand"] == "LIG"
    assert box["reference_chain"] == "A"


def test_box_from_cocrystal_center():
    ligand = {
        "resname": "LIG", "chain": "A", "resseq": 1,
        "coords": [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    }
    box = box_from_cocrystal(ligand, padding=0.0)
    assert box["center"][0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# box_from_residues
# ---------------------------------------------------------------------------

def test_box_from_residues_correct_center(tmp_path):
    pdb = _make_ca_pdb(tmp_path)
    # residues 10 and 30 → CA at (1,2,3) and (3,4,5)
    box = box_from_residues(pdb, residue_ids=[10, 30], chains=["A"], padding=0.0)
    assert box["center"][0] == pytest.approx(2.0)
    assert box["center"][1] == pytest.approx(3.0)


def test_box_from_residues_method_label(tmp_path):
    pdb = _make_ca_pdb(tmp_path)
    box = box_from_residues(pdb, residue_ids=[10], chains=["A"], padding=0.0)
    assert box["method"] == "residue_defined"
    assert 10 in box["residues"]


def test_box_from_residues_raises_if_no_match(tmp_path):
    pdb = _make_ca_pdb(tmp_path)
    with pytest.raises(ValueError, match="No C"):
        box_from_residues(pdb, residue_ids=[999], chains=["A"], padding=0.0)


def test_box_from_residues_chain_filter(tmp_path):
    pdb = _make_ca_pdb(tmp_path)
    with pytest.raises(ValueError):
        box_from_residues(pdb, residue_ids=[10], chains=["B"], padding=0.0)


# ---------------------------------------------------------------------------
# find_cocrystal_ligands
# ---------------------------------------------------------------------------

def test_find_cocrystal_ligands_finds_drug_like(tmp_path):
    pdb = _make_hetatm_pdb(tmp_path)
    ligands = find_cocrystal_ligands(pdb)
    assert len(ligands) == 1
    assert ligands[0]["resname"] == "LIG"


def test_find_cocrystal_ligands_filters_additives(tmp_path):
    pdb = _make_additive_pdb(tmp_path)
    ligands = find_cocrystal_ligands(pdb)
    assert ligands == []


def test_find_cocrystal_ligands_centroid_computed(tmp_path):
    pdb = _make_hetatm_pdb(tmp_path)
    ligands = find_cocrystal_ligands(pdb)
    cx, cy, cz = ligands[0]["centroid"]
    # atoms span x=10..17, centroid x should be in that range
    assert 10.0 <= cx <= 17.0


def test_blind_box_covers_whole_protein(tmp_path):
    """box_blind is not separately exported, but box_from_residues with all residues
    effectively gives a whole-protein box — verify volume grows with padding."""
    pdb = _make_ca_pdb(tmp_path)
    box_tight = box_from_residues(pdb, [10, 20, 30], ["A"], padding=0.0)
    box_padded = box_from_residues(pdb, [10, 20, 30], ["A"], padding=10.0)
    assert box_padded["volume_angstrom3"] > box_tight["volume_angstrom3"]
