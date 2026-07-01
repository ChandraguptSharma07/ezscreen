from __future__ import annotations

import csv

from ezscreen.results.merger import join_cnn_scores, merge_shard_results


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


def test_merge_joins_cnn_scores_when_present(tmp_path):
    sd = tmp_path / "shard"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score"], [
        {"ligand": "nb0_lig_00000", "score": "-8.1"},
        {"ligand": "nb0_lig_00001", "score": "-7.0"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles", "conformer_qc"], [
        {"ligand": "nb0_lig_00000", "name": "A", "smiles": "CCO", "conformer_qc": ""},
        {"ligand": "nb0_lig_00001", "name": "B", "smiles": "c1ccccc1", "conformer_qc": ""},
    ])

    out = tmp_path / "output"
    out.mkdir()
    # gnina_runner drops cnn_scores.csv into the run output before a re-merge
    _write(out / "cnn_scores.csv", ["lig_id", "CNNscore", "CNNaffinity"], [
        {"lig_id": "nb0_lig_00000", "CNNscore": "0.91", "CNNaffinity": "6.2"},
    ])

    merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    by_name = {r["name"]: r for r in rows}
    assert "CNNscore" in rows[0] and "CNNaffinity" in rows[0]
    assert by_name["A"]["CNNscore"] == "0.91"
    assert by_name["A"]["CNNaffinity"] == "6.2"
    # a ligand with no CNN entry gets blank cells, never an error
    assert by_name["B"]["CNNscore"] == ""


def test_join_cnn_scores_updates_existing_scores_csv(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    _write(out / "scores.csv", ["ligand", "score", "name", "smiles"], [
        {"ligand": "nb0_lig_00000", "score": "-8.1", "name": "A", "smiles": "CCO"},
        {"ligand": "nb0_lig_00001", "score": "-7.0", "name": "B", "smiles": "c1ccccc1"},
    ])
    # no cnn_scores.csv yet → no-op
    assert join_cnn_scores(out) is False
    assert "CNNscore" not in list(csv.DictReader((out / "scores.csv").open()))[0]

    _write(out / "cnn_scores.csv", ["lig_id", "CNNscore", "CNNaffinity"], [
        {"lig_id": "nb0_lig_00000", "CNNscore": "0.91", "CNNaffinity": "6.2"},
    ])
    assert join_cnn_scores(out) is True
    rows = list(csv.DictReader((out / "scores.csv").open()))
    by_name = {r["name"]: r for r in rows}
    assert by_name["A"]["CNNscore"] == "0.91"
    assert by_name["A"]["CNNaffinity"] == "6.2"
    assert by_name["B"]["CNNscore"] == ""


def test_join_cnn_scores_preserves_rows_outside_the_batch(tmp_path):
    # a GNINA-engine run already carries CNN for every hit; re-rescoring only the
    # top hit must update that hit and leave the others' CNN values intact.
    out = tmp_path / "output"
    out.mkdir()
    _write(out / "scores.csv", ["ligand", "score", "CNNscore", "CNNaffinity"], [
        {"ligand": "l0", "score": "-8.1", "CNNscore": "0.70", "CNNaffinity": "5.0"},
        {"ligand": "l1", "score": "-7.0", "CNNscore": "0.60", "CNNaffinity": "4.5"},
    ])
    _write(out / "cnn_scores.csv", ["lig_id", "CNNscore", "CNNaffinity"], [
        {"lig_id": "l0", "CNNscore": "0.95", "CNNaffinity": "6.8"},
    ])
    assert join_cnn_scores(out) is True
    rows = {r["ligand"]: r for r in csv.DictReader((out / "scores.csv").open())}
    assert rows["l0"]["CNNaffinity"] == "6.8"   # rescored hit updated
    assert rows["l1"]["CNNaffinity"] == "4.5"   # untouched hit preserved, not blanked


def test_merge_without_cnn_scores_has_no_cnn_cols(tmp_path):
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
    assert "CNNscore" not in rows[0]
