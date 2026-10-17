from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from ezscreen.benchmark.vinardo_paper import (
    PaperTarget,
    load_targets,
    mol2_box,
    strip_pdb_residue,
    verify_dud_archives,
)


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _nested_mol2(count: int) -> bytes:
    payload = b"\n".join(
        b"@<TRIPOS>MOLECULE\ncompound\n0 0 0\nSMALL\nNO_CHARGES"
        for _ in range(count)
    )
    return gzip.compress(payload)


def test_load_targets_enforces_alphabetical_odd_even_split(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "position": 1,
                    "target_id": "ace",
                    "pdb_id": "1o86",
                    "split": "development",
                    "actives": 2,
                    "decoys": 3,
                    "library_stem": "ace",
                },
                {
                    "position": 2,
                    "target_id": "ada",
                    "pdb_id": "1ndw",
                    "split": "validation",
                    "actives": 4,
                    "decoys": 5,
                    "library_stem": "ada",
                },
            ]
        ),
        encoding="utf-8",
    )
    targets = load_targets(path)
    assert targets[0].intended_compounds == 5

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[1]["split"] = "development"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected validation"):
        load_targets(path)


def test_mol2_box_matches_smina_autobox_padding() -> None:
    payload = """@<TRIPOS>MOLECULE
lig
2 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 C1 -2.0 1.0 4.0 C.3
2 C2 4.0 5.0 10.0 C.3
@<TRIPOS>BOND
"""
    result = mol2_box(payload, padding=15.0)
    assert result["center"] == [1.0, 3.0, 7.0]
    assert result["size"] == [36.0, 34.0, 36.0]


def test_strip_pdb_residue_handles_atom_records() -> None:
    pdb = (
        "ATOM      1  N   SAM     1      0.0  0.0  0.0\n"
        "ATOM      2  CA  ALA A   2      0.0  0.0  0.0\n"
        "HETATM    3  C1  SAM     1      0.0  0.0  0.0\n"
    )
    cleaned, removed = strip_pdb_residue(pdb, "SAM")
    assert removed == 2
    assert "SAM" not in cleaned
    assert "ALA" in cleaned


def test_verify_dud_archives_checks_counts_and_structures(tmp_path: Path) -> None:
    target = PaperTarget(
        position=1,
        target_id="ace",
        pdb_id="1o86",
        split="development",
        actives=2,
        decoys=3,
        library_stem="ace",
    )
    ligands = tmp_path / "ligands.tar.gz"
    decoys = tmp_path / "decoys.tar.gz"
    receptors = tmp_path / "receptors.tar.gz"
    crystals = tmp_path / "crystals.tar.gz"
    _write_tar(
        ligands,
        {"dud_ligands2006/ace_ligands.mol2.gz": _nested_mol2(2)},
    )
    _write_tar(
        decoys,
        {"dud_decoys2006/ace_decoys.mol2.gz": _nested_mol2(3)},
    )
    _write_tar(receptors, {"targets/ace/rec.pdb": b"ATOM\n"})
    _write_tar(
        crystals,
        {"targets/ace/xtal-lig.mol2": b"@<TRIPOS>MOLECULE\nxtal\n"},
    )

    result = verify_dud_archives(
        [target],
        ligand_archive=ligands,
        decoy_archive=decoys,
        receptor_archive=receptors,
        crystal_ligand_archive=crystals,
    )
    assert result["mismatch_count"] == 0
    assert result["actual_actives"] == 2
    assert result["actual_decoys"] == 3
