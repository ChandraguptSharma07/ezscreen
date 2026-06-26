from __future__ import annotations

import csv

from ezscreen.results.merger import merge_shard_results


def _write(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_merge_carries_conformer_qc_from_index(tmp_path):
    sd = tmp_path / "shard"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score"], [
        {"ligand": "nb0_lig_00000", "score": "-8.1"},
        {"ligand": "nb0_lig_00001", "score": "-7.0"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles", "conformer_qc"], [
        {"ligand": "nb0_lig_00000", "name": "A", "smiles": "CCO", "conformer_qc": ""},
        {"ligand": "nb0_lig_00001", "name": "B", "smiles": "c1ccccc1", "conformer_qc": "steric_clash"},
    ])

    out = tmp_path / "output"
    merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    by_name = {r["name"]: r for r in rows}
    assert "conformer_qc" in rows[0]
    assert by_name["A"]["conformer_qc"] == ""
    assert by_name["B"]["conformer_qc"] == "steric_clash"


def test_merge_omits_conformer_qc_when_all_clean(tmp_path):
    sd = tmp_path / "shard"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score"], [
        {"ligand": "nb0_lig_00000", "score": "-8.1"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles", "conformer_qc"], [
        {"ligand": "nb0_lig_00000", "name": "A", "smiles": "CCO", "conformer_qc": ""},
    ])

    out = tmp_path / "output"
    merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    # no flags anywhere → column is omitted to avoid an all-empty column
    assert "conformer_qc" not in rows[0]
