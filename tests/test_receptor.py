from __future__ import annotations

from ezscreen.prep.receptor import (
    _apply_hetero_selection,
    heteroatom_inventory,
)


def _line(rec, serial, name, resn, chain, resseq, elem):
    L = list(" " * 80)
    L[0:6] = f"{rec:<6}"
    L[6:11] = f"{serial:>5}"
    L[12:16] = f"{name:<4}"
    L[17:20] = f"{resn:>3}"
    L[21] = chain
    L[22:26] = f"{resseq:>4}"
    L[30:38] = f"{0.0:8.3f}"
    L[38:46] = f"{0.0:8.3f}"
    L[46:54] = f"{0.0:8.3f}"
    L[76:78] = f"{elem:>2}"
    return "".join(L)


def _sample_pdb(tmp_path):
    lines = [
        _line("ATOM", 1, "CA", "ALA", "A", 10, "C"),   # protein
        _line("HETATM", 2, "O", "HOH", "A", 200, "O"),  # water
        _line("HETATM", 3, "ZN", "ZN", "A", 300, "ZN"),  # metal
        _line("HETATM", 4, "FE", "HEM", "A", 400, "FE"),  # cofactor
    ]
    # a 5-atom drug-like ligand
    for i in range(5):
        lines.append(_line("HETATM", 10 + i, f"C{i}", "LIG", "A", 500, "C"))
    lines.append("END")
    p = tmp_path / "complex.pdb"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_heteroatom_inventory_classifies(tmp_path):
    inv = {(h["resname"], h["resseq"]): h for h in heteroatom_inventory(_sample_pdb(tmp_path))}
    assert inv[("HOH", 200)]["category"] == "water"    and inv[("HOH", 200)]["keep_default"] is False
    assert inv[("ZN", 300)]["category"]  == "metal"    and inv[("ZN", 300)]["keep_default"] is False
    assert inv[("HEM", 400)]["category"] == "cofactor" and inv[("HEM", 400)]["keep_default"] is True
    assert inv[("LIG", 500)]["category"] == "ligand"   and inv[("LIG", 500)]["n_atoms"] == 5


def test_apply_hetero_selection_keeps_only_selected(tmp_path):
    src = _sample_pdb(tmp_path)
    dst = tmp_path / "out.pdb"
    removed = _apply_hetero_selection(src, dst, {("LIG", "A", 500)})
    text = dst.read_text()

    assert removed == 3                       # HOH + ZN + HEM dropped
    assert "ATOM" in text                     # protein retained
    assert "LIG" in text                      # kept ligand retained
    assert "HOH" not in text and " ZN " not in text  # de-selected hetero gone
