from __future__ import annotations

import math
from pathlib import Path

import pytest

from ezscreen.errors import InvalidReceptorError
from ezscreen.prep.receptor import (
    _apply_hetero_selection,
    _apply_pqr_residue_names_to_pdb,
    _apply_residue_state_overrides,
    _hetero_retention_report,
    _meeko_receptor_command,
    _pqr_atom_to_pdb_line,
    _run_meeko_receptor,
    _run_pdb2pqr_protonation,
    heteroatom_inventory,
    metal_site_qc,
    ramachandran_angles,
    receptor_sanity_checks,
    use_prepared_receptor,
    validate_prepared_receptor,
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


def _pdbqt_atom(serial: int = 1, x: float = 1.0) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A   1    "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00  0.00     0.100 C \n"
    )


def test_use_prepared_receptor_preserves_bytes_and_writes_provenance(tmp_path):
    source = tmp_path / "expert_receptor.pdbqt"
    source.write_text("REMARK expert protocol\n" + _pdbqt_atom(), encoding="utf-8")

    result = use_prepared_receptor(source, tmp_path / "staged")

    staged = result["pdbqt_path"]
    assert staged.read_bytes() == source.read_bytes()
    assert result["report"]["preparation_skipped"] is True
    assert result["report"]["byte_identical"] is True
    assert result["report"]["source_sha256"] == result["report"]["staged_sha256"]
    assert result["report"]["validation"]["atom_count"] == 1
    assert (tmp_path / "staged" / "prepared_receptor_provenance.json").is_file()


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("receptor.pdb", _pdbqt_atom(), "requires a .pdbqt"),
        ("receptor.pdbqt", "REMARK no atoms\n", "contains no ATOM"),
        (
            "receptor.pdbqt",
            _pdbqt_atom().replace("   1.000", "     nan", 1),
            "non-finite coordinates",
        ),
    ],
)
def test_validate_prepared_receptor_rejects_invalid_input(
    tmp_path, filename, contents, message
):
    source = tmp_path / filename
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(InvalidReceptorError, match=message):
        validate_prepared_receptor(source)


def _sample_pdb(tmp_path):
    lines = [
        _line("ATOM", 1, "CA", "ALA", "A", 10, "C"),  # protein
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
    inv = {
        (h["resname"], h["resseq"]): h
        for h in heteroatom_inventory(_sample_pdb(tmp_path))
    }
    assert (
        inv[("HOH", 200)]["category"] == "water"
        and inv[("HOH", 200)]["keep_default"] is False
    )
    assert (
        inv[("ZN", 300)]["category"] == "metal"
        and inv[("ZN", 300)]["keep_default"] is False
    )
    assert (
        inv[("HEM", 400)]["category"] == "cofactor"
        and inv[("HEM", 400)]["keep_default"] is True
    )
    assert (
        inv[("LIG", 500)]["category"] == "ligand" and inv[("LIG", 500)]["n_atoms"] == 5
    )


def test_apply_hetero_selection_keeps_only_selected(tmp_path):
    src = _sample_pdb(tmp_path)
    dst = tmp_path / "out.pdb"
    removed = _apply_hetero_selection(src, dst, {("LIG", "A", 500)})
    text = dst.read_text()

    assert removed == 3  # HOH + ZN + HEM dropped
    assert "ATOM" in text  # protein retained
    assert "LIG" in text  # kept ligand retained
    assert "HOH" not in text and " ZN " not in text  # de-selected hetero gone


def test_meeko_receptor_command_accepts_modern_script_name(monkeypatch):
    monkeypatch.setattr("ezscreen.prep.receptor.sys.prefix", "/missing-env")
    monkeypatch.setattr(
        "ezscreen.prep.receptor.shutil.which",
        lambda name: (
            "/venv/bin/mk_prepare_receptor.py"
            if name == "mk_prepare_receptor.py"
            else None
        ),
    )

    assert _meeko_receptor_command() == "/venv/bin/mk_prepare_receptor.py"


def test_run_meeko_uses_modern_read_pdb_interface(tmp_path, monkeypatch):
    import subprocess

    src = tmp_path / "fixed.pdb"
    src.write_text(_pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)) + "\nEND\n")
    captured = {}

    monkeypatch.setattr(
        "ezscreen.prep.receptor._meeko_receptor_command",
        lambda: "/venv/bin/mk_prepare_receptor.py",
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[cmd.index("-p") + 1]).write_text("ATOM      1  CA  ALA A   1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("ezscreen.prep.receptor.subprocess.run", fake_run)

    _run_meeko_receptor(src, tmp_path)

    assert "--read_pdb" in captured["cmd"]
    assert "-i" not in captured["cmd"]
    assert "-a" in captured["cmd"]


def test_hetero_retention_report_requires_every_selected_group(tmp_path):
    pdbqt = tmp_path / "receptor.pdbqt"
    pdbqt.write_text(_line("HETATM", 1, "O", "HOH", "A", 301, "O") + "\nEND\n")

    report = _hetero_retention_report(
        pdbqt,
        [
            {"resname": "HOH", "chain": "A", "resseq": 301},
            {"resname": "HOH", "chain": "A", "resseq": 566},
        ],
    )

    assert report["complete"] is False
    assert report["retained"] == [{"resname": "HOH", "chain": "A", "resseq": 301}]
    assert report["missing"] == [{"resname": "HOH", "chain": "A", "resseq": 566}]


def test_residue_state_overrides_are_exact_and_user_controlled(tmp_path):
    src = tmp_path / "propka.pdb"
    src.write_text(
        "\n".join(
            [
                _pdb_atom(1, "CA", "ASP", "A", 25, (0.0, 0.0, 0.0)),
                _pdb_atom(2, "CA", "ASH", "B", 25, (1.0, 0.0, 0.0)),
                "END",
            ]
        )
        + "\n"
    )
    dst = tmp_path / "manual.pdb"

    result = _apply_residue_state_overrides(
        src,
        dst,
        [
            {"chain": "A", "resseq": 25, "state": "ASH"},
            {"chain": "B", "resseq": 25, "state": "ASP"},
        ],
    )

    assert result["assignment_count"] == 2
    assert " ASH A  25" in dst.read_text()
    assert " ASP B  25" in dst.read_text()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"chain": "A", "resseq": 99, "state": "ASH"}, "not found"),
        ({"chain": "A", "resseq": 25, "state": "LYN"}, "incompatible"),
    ],
)
def test_residue_state_overrides_fail_closed(tmp_path, override, message):
    src = tmp_path / "fixed.pdb"
    src.write_text(_pdb_atom(1, "CA", "ASP", "A", 25, (0.0, 0.0, 0.0)) + "\nEND\n")

    with pytest.raises(Exception, match=message):
        _apply_residue_state_overrides(src, tmp_path / "out.pdb", [override])


def test_prep_receptor_rejects_ambiguous_manual_and_automatic_states(tmp_path):
    from ezscreen.prep import receptor as rec

    src = tmp_path / "fixed.pdb"
    src.write_text(_pdb_atom(1, "CA", "ASP", "A", 25, (0.0, 0.0, 0.0)) + "\nEND\n")

    with pytest.raises(Exception, match="either automatic.*or manual"):
        rec.prep_receptor(
            src,
            ["A"],
            tmp_path / "out",
            receptor_isomers="single",
            residue_state_overrides=[{"chain": "A", "resseq": 25, "state": "ASH"}],
        )


def _pdb_atom(serial, name, resn, chain, resseq, xyz, elem="C"):
    x, y, z = xyz
    L = list(" " * 80)
    L[0:6] = "ATOM  "
    L[6:11] = f"{serial:>5}"
    L[12:16] = f"{name:<4}"
    L[17:20] = f"{resn:>3}"
    L[21] = chain
    L[22:26] = f"{resseq:>4}"
    L[30:38] = f"{x:8.3f}"
    L[38:46] = f"{y:8.3f}"
    L[46:54] = f"{z:8.3f}"
    L[76:78] = f"{elem:>2}"
    return "".join(L)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    length = math.sqrt(_dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length)


def _rotate(v, axis, degrees):
    theta = math.radians(degrees)
    cost = math.cos(theta)
    sint = math.sin(theta)
    cross = _cross(axis, v)
    dot = _dot(axis, v)
    return (
        v[0] * cost + cross[0] * sint + axis[0] * dot * (1 - cost),
        v[1] * cost + cross[1] * sint + axis[1] * dot * (1 - cost),
        v[2] * cost + cross[2] * sint + axis[2] * dot * (1 - cost),
    )


def _fourth_point_for_dihedral(p0, p1, p2, degrees):
    axis = _norm(_sub(p2, p1))
    v = _sub(p0, p1)
    v = _norm(
        (
            v[0] - _dot(v, axis) * axis[0],
            v[1] - _dot(v, axis) * axis[1],
            v[2] - _dot(v, axis) * axis[2],
        )
    )
    w = _rotate(v, axis, degrees)
    return (p2[0] + w[0], p2[1] + w[1], p2[2] + w[2])


def test_ramachandran_angles_for_known_geometry(tmp_path):
    c1 = (0.0, 1.0, 0.0)
    n2 = (0.0, 0.0, 0.0)
    ca2 = (1.0, 0.0, 0.0)
    c2 = (1.0, math.cos(math.radians(-60)), math.sin(math.radians(-60)))
    n3 = _fourth_point_for_dihedral(n2, ca2, c2, 120)

    pdb = tmp_path / "rama.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_atom(1, "N", "ALA", "A", 1, (-0.5, 1.0, 0.0), "N"),
                _pdb_atom(2, "CA", "ALA", "A", 1, (-0.2, 1.2, 0.0), "C"),
                _pdb_atom(3, "C", "ALA", "A", 1, c1, "C"),
                _pdb_atom(4, "N", "GLY", "A", 2, n2, "N"),
                _pdb_atom(5, "CA", "GLY", "A", 2, ca2, "C"),
                _pdb_atom(6, "C", "GLY", "A", 2, c2, "C"),
                _pdb_atom(7, "N", "SER", "A", 3, n3, "N"),
                _pdb_atom(8, "CA", "SER", "A", 3, (n3[0] + 0.5, n3[1], n3[2]), "C"),
                _pdb_atom(9, "C", "SER", "A", 3, (n3[0] + 1.0, n3[1], n3[2]), "C"),
                "END",
            ]
        )
        + "\n"
    )

    angles = ramachandran_angles(pdb)

    assert len(angles) == 1
    assert angles[0]["resname"] == "GLY"
    assert angles[0]["phi"] == pytest.approx(-60.0, abs=0.2)
    assert angles[0]["psi"] == pytest.approx(120.0, abs=0.2)


def test_pqr_atom_to_pdb_line_writes_alpha_carbon_as_carbon():
    line = "ATOM      2  CA  PRO     1      30.307  38.663   5.319  0.3300 2.0000"

    pdb_line = _pqr_atom_to_pdb_line(line, 2)

    assert pdb_line is not None
    assert pdb_line[12:16].strip() == "CA"
    assert pdb_line[17:20].strip() == "PRO"
    assert pdb_line[76:78].strip() == "C"


def test_apply_pqr_residue_names_preserves_source_pdb_geometry(tmp_path):
    src = tmp_path / "fixed.pdb"
    src.write_text(
        "\n".join(
            [
                _pdb_atom(1, "N", "HIS", "A", 7, (0.0, 0.0, 0.0), "N"),
                _pdb_atom(2, "CA", "HIS", "A", 7, (1.0, 0.0, 0.0), "C"),
                _pdb_atom(3, "C", "ASP", "A", 8, (2.0, 0.0, 0.0), "C"),
                "END",
            ]
        )
        + "\n"
    )
    pqr = tmp_path / "propka.pqr"
    pqr.write_text(
        "\n".join(
            [
                "ATOM      1  N   HIE     7      9.000   9.000   9.000  0.0000 1.5000",
                "ATOM      2  CA  HIE     7      9.000   9.000   9.000  0.0000 1.7000",
                "ATOM      3  C   ASH     8      9.000   9.000   9.000  0.0000 1.7000",
                "END",
            ]
        )
        + "\n"
    )
    dst = tmp_path / "propka_mapped.pdb"

    result = _apply_pqr_residue_names_to_pdb(src, pqr, dst)

    lines = dst.read_text().splitlines()
    assert result["assignment_count"] == 2
    assert lines[0][17:20].strip() == "HIE"
    assert lines[1][17:20].strip() == "HIE"
    assert lines[2][17:20].strip() == "ASH"
    assert lines[1][21] == "A"
    assert lines[1][12:16].strip() == "CA"
    assert lines[1][30:38].strip() == "1.000"
    assert lines[1][76:78].strip() == "C"


def test_apply_pqr_residue_names_infers_histidine_tautomer_from_hydrogens(tmp_path):
    src = tmp_path / "fixed.pdb"
    src.write_text(
        "\n".join(
            [
                _pdb_atom(1, "N", "HIS", "A", 69, (0.0, 0.0, 0.0), "N"),
                _pdb_atom(2, "CA", "HIS", "A", 69, (1.0, 0.0, 0.0), "C"),
                _pdb_atom(3, "ND1", "HIS", "A", 69, (2.0, 0.0, 0.0), "N"),
                "END",
            ]
        )
        + "\n"
    )
    pqr = tmp_path / "propka.pqr"
    pqr.write_text(
        "\n".join(
            [
                "ATOM      1  N   HIS A  69      0.000   0.000   0.000 -0.4000 1.5000",
                "ATOM      2  CA  HIS A  69      1.000   0.000   0.000  0.0000 2.0000",
                "ATOM      3  ND1 HIS A  69      2.000   0.000   0.000 -0.4000 1.5000",
                "ATOM      4  HD1 HIS A  69      2.200   0.000   0.000  0.4000 1.0000",
                "END",
            ]
        )
        + "\n"
    )
    dst = tmp_path / "mapped.pdb"

    result = _apply_pqr_residue_names_to_pdb(src, pqr, dst)

    assert result["assignment_count"] == 1
    assert result["assignments"][0]["state"] == "HID"
    assert " HID A  69" in dst.read_text()


def test_receptor_sanity_checks_flags_missing_backbone_and_clash(tmp_path):
    pdb = tmp_path / "bad_receptor.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_atom(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0), "N"),
                _pdb_atom(2, "CA", "ALA", "A", 1, (1.4, 0.0, 0.0), "C"),
                _pdb_atom(3, "C", "ALA", "A", 1, (2.5, 0.0, 0.0), "C"),
                _pdb_atom(4, "N", "GLY", "A", 2, (0.2, 0.0, 0.0), "N"),
                _pdb_atom(5, "CA", "GLY", "A", 2, (3.9, 0.0, 0.0), "C"),
                _pdb_atom(6, "C", "GLY", "A", 2, (5.0, 0.0, 0.0), "C"),
                _pdb_atom(7, "O", "GLY", "A", 2, (6.0, 0.0, 0.0), "O"),
                "END",
            ]
        )
        + "\n"
    )

    qc = receptor_sanity_checks(pdb)

    assert qc["missing_backbone"][0]["missing"] == ["O"]
    assert qc["steric_clash_count"] == 1
    assert qc["steric_clashes"][0]["distance"] == pytest.approx(0.2, abs=0.01)


def test_metal_site_qc_reports_atom_coded_zinc_and_protein_donors(tmp_path):
    pdb = tmp_path / "zinc_receptor.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_atom(1, "ZN", "ZN", "A", 500, (0.0, 0.0, 0.0), "ZN"),
                _pdb_atom(2, "NE2", "HIS", "A", 383, (2.1, 0.0, 0.0), "N"),
                _pdb_atom(3, "OE1", "GLU", "A", 411, (0.0, 2.2, 0.0), "O"),
                _pdb_atom(4, "CA", "ALA", "A", 1, (0.0, 0.0, 2.2), "C"),
                "END",
            ]
        )
        + "\n"
    )

    qc = metal_site_qc(pdb)

    assert qc["metal_elements"] == ["ZN"]
    assert qc["score_validation_status"] == "zinc_specialized_available"
    assert qc["metals"][0]["record"] == "ATOM"
    assert [donor["atom"] for donor in qc["metals"][0]["nearby_protein_donors"]] == [
        "NE2",
        "OE1",
    ]


def test_metal_site_qc_marks_non_zinc_metals_unvalidated(tmp_path):
    pdb = tmp_path / "calcium_receptor.pdb"
    pdb.write_text(
        _pdb_atom(1, "CA", "CA", "A", 700, (0.0, 0.0, 0.0), "CA") + "\nEND\n"
    )

    qc = metal_site_qc(pdb)

    assert qc["metal_elements"] == ["CA"]
    assert qc["score_validation_status"] == "element_specific_validation_needed"
    assert qc["metals"][0]["nearby_protein_donors"] == []


def test_pdb2pqr_protonation_assigns_titratable_states(tmp_path, monkeypatch):
    import subprocess

    src = tmp_path / "fixed.pdb"
    src.write_text("END\n")
    dst = tmp_path / "protonated.pdb"

    def fake_which(name):
        return "pdb2pqr" if name == "pdb2pqr" else None

    def fake_run(cmd, **kwargs):
        assert "--with-ph=6.5" in cmd
        assert "--titration-state-method=propka" in cmd
        assert "--keep-chain" in cmd
        pqr = tmp_path / "protonated.pqr"
        pqr.write_text(
            "\n".join(
                [
                    "ATOM      1  N   HIE A   7      0.000   0.000   0.000  0.0000 1.5000",
                    "ATOM      2  CA  HIE A   7      1.000   0.000   0.000  0.0000 1.7000",
                    "ATOM      3  C   ASH A   8      2.000   0.000   0.000  0.0000 1.7000",
                    "END",
                ]
            )
            + "\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("ezscreen.prep.receptor.shutil.which", fake_which)
    monkeypatch.setattr("ezscreen.prep.receptor.subprocess.run", fake_run)

    result = _run_pdb2pqr_protonation(src, dst, 6.5)

    assert result["applied"] is True
    assert result["assignment_count"] == 2
    assert {a["state"] for a in result["assignments"]} == {"HIE", "ASH"}
    text = dst.read_text()
    assert " HIE A   7" in text
    assert " ASH A   8" in text


def test_pdb2pqr_protonation_falls_back_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("ezscreen.prep.receptor.shutil.which", lambda _name: None)

    result = _run_pdb2pqr_protonation(tmp_path / "fixed.pdb", tmp_path / "out.pdb", 7.4)

    assert result["applied"] is False
    assert result["assignment_count"] == 0
    assert "not found" in result["error"]


@pytest.mark.parametrize("missing_count", [0, 21])
def test_prep_receptor_defers_variant_prep_to_kaggle(
    tmp_path, monkeypatch, missing_count
):
    from ezscreen.prep import receptor as rec

    src = tmp_path / "input.pdb"
    src.write_text(_pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C") + "\nEND\n")

    def fake_pdbfixer(_src, dst, keep_waters, remove_heterogens=True):
        dst.write_text(src.read_text())
        return {
            "missing_residues": missing_count,
            "residue_count": 1,
            "pdbfixer_version": "test",
        }

    def fake_protonation(_src, dst, ph):
        raise AssertionError("variant mode should not run local PDB2PQR")

    def fake_meeko(src_pdb, output_dir):
        out = output_dir / f"{src_pdb.stem}.pdbqt"
        out.write_text(f"RECEPTOR {src_pdb.stem}\n")
        return out, "test-meeko"

    monkeypatch.setattr(rec, "_run_pdbfixer", fake_pdbfixer)
    monkeypatch.setattr(rec, "_run_pdb2pqr_protonation", fake_protonation)
    monkeypatch.setattr(rec, "_run_meeko_receptor", fake_meeko)

    result = rec.prep_receptor(
        pdb_path=src,
        chains=["A"],
        output_dir=tmp_path / "out",
        receptor_isomers="variants",
    )

    assert len(result["variant_pdbqt_paths"]) == 1
    assert result["receptor_variant_modes"] == ["default", "propka_ph"]
    assert result["report"]["receptor_variant_modes"] == ["default", "propka_ph"]
    assert result["report"]["receptor_protonation"]["error"] == "deferred_to_kaggle"
    unresolved = [
        w
        for w in result["report"]["warnings"]
        if w["category"] == "missing_residues_unresolved"
    ]
    assert len(unresolved) == bool(missing_count)
    if missing_count:
        assert unresolved[0]["affected_count"] == missing_count
        assert "not rebuilt" in unresolved[0]["message"]
    assert not any(
        w["category"] == "missing_residues_modelled"
        for w in result["report"]["warnings"]
    )
    assert any(
        w["category"] == "receptor_protonation_deferred"
        for w in result["report"]["warnings"]
    )


def test_prep_receptor_defers_single_state_when_local_propka_fails(
    tmp_path, monkeypatch
):
    from ezscreen.prep import receptor as rec

    src = tmp_path / "input.pdb"
    src.write_text(_pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C") + "\nEND\n")

    def fake_pdbfixer(_src, dst, keep_waters, remove_heterogens=True):
        dst.write_text(src.read_text())
        return {"missing_residues": 0, "residue_count": 1, "pdbfixer_version": "test"}

    def fake_protonation(_src, _dst, _ph):
        return {
            "applied": False,
            "tool": "pdb2pqr",
            "assignment_count": 0,
            "assignments": [],
            "error": "propka crashed under python 3.14",
        }

    def fake_meeko(src_pdb, output_dir):
        out = output_dir / f"{src_pdb.stem}.pdbqt"
        out.write_text(f"RECEPTOR {src_pdb.stem}\n")
        return out, "test-meeko"

    monkeypatch.setattr(rec, "_run_pdbfixer", fake_pdbfixer)
    monkeypatch.setattr(rec, "_run_pdb2pqr_protonation", fake_protonation)
    monkeypatch.setattr(rec, "_run_meeko_receptor", fake_meeko)

    result = rec.prep_receptor(
        pdb_path=src,
        chains=["A"],
        output_dir=tmp_path / "out",
        receptor_isomers="single",
    )

    assert result["receptor_variant_modes"] == ["propka_ph"]
    assert result["report"]["receptor_variant_modes"] == ["propka_ph"]
    assert result["report"]["receptor_protonation"]["error"].startswith(
        "local_failed_deferred_to_kaggle"
    )
    assert any(
        w["category"] == "receptor_protonation_deferred"
        for w in result["report"]["warnings"]
    )


def test_pdb2pqr_failure_includes_log_tail(tmp_path, monkeypatch):
    import subprocess

    src = tmp_path / "fixed.pdb"
    src.write_text("END\n")
    dst = tmp_path / "protonated.pdb"

    monkeypatch.setattr(
        "ezscreen.prep.receptor.shutil.which",
        lambda name: "pdb2pqr" if name == "pdb2pqr" else None,
    )

    def fake_run(cmd, **kwargs):
        (tmp_path / "protonated.log").write_text("PROPKA traceback detail\n")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr("ezscreen.prep.receptor.subprocess.run", fake_run)

    result = _run_pdb2pqr_protonation(src, dst, 7.4)

    assert result["applied"] is False
    assert "PROPKA traceback detail" in result["error"]
