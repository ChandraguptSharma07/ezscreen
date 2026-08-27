from __future__ import annotations

import csv
import json

from ezscreen.results.merger import (
    build_consensus_results,
    join_cnn_scores,
    merge_shard_results,
)


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


def test_join_cnn_scores_reads_gnina_output_fallback(tmp_path):
    run_dir = tmp_path / "ezs-team"
    run_dir.mkdir()
    _write(run_dir / "scores.csv", ["ligand", "score", "name", "smiles"], [
        {"ligand": "l0", "score": "-8.1", "name": "A", "smiles": "CCO"},
        {"ligand": "l1", "score": "-7.0", "name": "B", "smiles": "CCN"},
    ])
    gnina_output = run_dir / "gnina_output"
    gnina_output.mkdir()
    _write(gnina_output / "cnn_scores.csv", ["lig_id", "CNNscore", "CNNaffinity"], [
        {"lig_id": "l0", "CNNscore": "0.95", "CNNaffinity": "6.8"},
    ])

    assert join_cnn_scores(run_dir) is True
    rows = {r["ligand"]: r for r in csv.DictReader((run_dir / "scores.csv").open())}
    assert rows["l0"]["CNNaffinity"] == "6.8"
    assert rows["l1"]["CNNaffinity"] == ""


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


def test_merge_reports_kaggle_failed_prep_csv(tmp_path):
    sd = tmp_path / "shard"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score"], [
        {"ligand": "nb0_lig_00000", "score": "-8.1"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles", "conformer_qc"], [
        {"ligand": "nb0_lig_00000", "name": "A", "smiles": "CCO", "conformer_qc": ""},
    ])
    _write(sd / "failed_prep.csv", ["name", "smiles", "reason"], [
        {"name": "salt", "smiles": "CCO.Cl", "reason": "meeko_failed"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd], out)

    reasons = list(csv.DictReader((out / "unscored_reasons.csv").open()))
    assert any(
        r["name"] == "salt"
        and r["smiles"] == "CCO.Cl"
        and r["reason"] == "meeko_failed"
        for r in reasons
    )
    assert result["unscored_reason_counts"]["meeko_failed"] == 1



def test_merge_receptor_variants_keeps_best_per_ligand_identity(tmp_path):
    sd0 = tmp_path / "rv0"
    sd1 = tmp_path / "rv1"
    sd0.mkdir()
    sd1.mkdir()
    _write(sd0 / "scores.csv", ["ligand", "score", "name", "smiles", "receptor_variant"], [
        {"ligand": "nb0_lig_00000", "score": "-7.0", "name": "A", "smiles": "CCO", "receptor_variant": "default"},
        {"ligand": "nb0_lig_00001", "score": "-8.0", "name": "B", "smiles": "CCN", "receptor_variant": "default"},
    ])
    _write(sd1 / "scores.csv", ["ligand", "score", "name", "smiles", "receptor_variant"], [
        {"ligand": "nb1_lig_00000", "score": "-9.0", "name": "A", "smiles": "CCO", "receptor_variant": "propka_ph"},
        {"ligand": "nb1_lig_00001", "score": "-6.5", "name": "B", "smiles": "CCN", "receptor_variant": "propka_ph"},
    ])

    out = tmp_path / "output"
    merge_shard_results([sd0, sd1], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    by_smiles = {r["smiles"]: r for r in rows}
    assert len(rows) == 2
    assert by_smiles["CCO"]["score"] == "-9.0"
    assert by_smiles["CCO"]["receptor_variant"] == "propka_ph"
    assert by_smiles["CCN"]["score"] == "-8.0"
    assert by_smiles["CCN"]["receptor_variant"] == "default"


def test_merge_retains_receptor_variant_score_ceiling_as_scored_qc(tmp_path):
    sd = tmp_path / "rv"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score", "name", "smiles", "receptor_variant"], [
        {"ligand": "nb0_lig_00000", "score": "-6.0", "name": "A", "smiles": "CCO", "receptor_variant": "propka_ph"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles", "conformer_qc"], [
        {"ligand": "nb0_lig_00000", "name": "A", "smiles": "CCO", "conformer_qc": ""},
        {"ligand": "nb0_lig_00001", "name": "B", "smiles": "CCN", "conformer_qc": ""},
    ])
    _write(sd / "failed_docking.csv", ["ligand", "raw_score", "reason", "receptor_variant"], [
        {"ligand": "nb0_lig_00000", "raw_score": "25.8", "reason": "score_ceiling", "receptor_variant": "default"},
        {"ligand": "nb0_lig_00001", "raw_score": "20.0", "reason": "score_ceiling", "receptor_variant": "default"},
        {"ligand": "nb0_lig_00001", "raw_score": "2.0", "reason": "score_ceiling", "receptor_variant": "propka_ph"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    by_smiles = {row["smiles"]: row for row in rows}
    reasons = list(csv.DictReader((out / "unscored_reasons.csv").open()))

    assert len(rows) == 2
    assert by_smiles["CCN"]["score"] == "2.0"
    assert by_smiles["CCN"]["receptor_variant"] == "propka_ph"
    assert by_smiles["CCN"]["score_qc"] == "score_ceiling"
    assert reasons == []
    assert result["unscored_reason_counts"].get("score_ceiling", 0) == 0



def test_merge_keeps_score_floor_failures_unscored_not_headline_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ezscreen.config.load",
        lambda: {"local": {"enable_score_floor": True, "score_floor": -15.0, "score_ceiling": 0.0}},
    )
    sd = tmp_path / "floor"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score", "name", "smiles"], [
        {"ligand": "nb0_lig_00000", "score": "-7.0", "name": "A", "smiles": "CCO"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles"], [
        {"ligand": "nb0_lig_00000", "name": "A", "smiles": "CCO"},
        {"ligand": "nb0_lig_00001", "name": "B", "smiles": "CCN"},
    ])
    _write(sd / "failed_docking.csv", ["ligand", "raw_score", "reason"], [
        {"ligand": "nb0_lig_00001", "raw_score": "-50.0", "reason": "score_floor"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    reasons = list(csv.DictReader((out / "unscored_reasons.csv").open()))

    assert len(rows) == 1
    assert rows[0]["smiles"] == "CCO"
    assert reasons == [
        {
            "ligand": "nb0_lig_00001",
            "name": "B",
            "smiles": "CCN",
            "reason": "score_floor",
            "raw_score": "-50.0",
        }
    ]
    assert result["unscored_reason_counts"].get("score_floor", 0) == 1


def test_merge_removes_existing_below_floor_score_from_ranked_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ezscreen.config.load",
        lambda: {"local": {"enable_score_floor": True, "score_floor": -15.0, "score_ceiling": 0.0}},
    )
    sd = tmp_path / "floor"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score", "name", "smiles"], [
        {"ligand": "valid", "score": "-8.0", "name": "valid", "smiles": "CCO"},
        {"ligand": "underflow", "score": "-50.0", "name": "underflow", "smiles": "CCN"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    reasons = list(csv.DictReader((out / "unscored_reasons.csv").open()))

    assert [row["ligand"] for row in rows] == ["valid"]
    assert reasons == [{
        "ligand": "underflow",
        "name": "underflow",
        "smiles": "CCN",
        "reason": "score_floor",
        "raw_score": "-50.0",
    }]
    assert result["unscored_reason_counts"]["score_floor"] == 1


def test_merge_marks_existing_finite_ceiling_score_with_qc(tmp_path):
    sd = tmp_path / "shard"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score"], [
        {"ligand": "nb0_lig_00000", "score": "0.5"},
    ])
    _write(sd / "index.csv", ["ligand", "name", "smiles"], [
        {"ligand": "nb0_lig_00000", "name": "weak", "smiles": "CCO"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    reasons = list(csv.DictReader((out / "unscored_reasons.csv").open()))

    assert rows[0]["score"] == "0.5"
    assert rows[0]["score_qc"] == "score_ceiling"
    assert reasons == []
    assert result["total_hits"] == 1

def test_merge_unions_later_shard_only_columns(tmp_path):
    sd0 = tmp_path / "s0"
    sd1 = tmp_path / "s1"
    sd0.mkdir()
    sd1.mkdir()
    _write(sd0 / "scores.csv", ["ligand", "score"], [
        {"ligand": "l0", "score": "-8.1"},
    ])
    _write(sd1 / "scores.csv", ["ligand", "score", "note"], [
        {"ligand": "l1", "score": "-7.0", "note": "late_col"},
    ])

    out = tmp_path / "output"
    merge_shard_results([sd0, sd1], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    by_ligand = {row["ligand"]: row for row in rows}
    assert "note" in rows[0]
    assert by_ligand["l0"]["note"] == ""
    assert by_ligand["l1"]["note"] == "late_col"


def test_merge_adds_engine_column_and_meta(tmp_path):
    sd = tmp_path / "shard"
    sd.mkdir()
    _write(sd / "scores.csv", ["ligand", "score"], [
        {"ligand": "nb0_lig_00000", "score": "-8.1"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd], out, engine="gnina")

    rows = list(csv.DictReader((out / "scores.csv").open()))
    meta = json.loads((out / "results_meta.json").read_text())
    assert rows[0]["engine"] == "gnina"
    assert meta["engine"] == "gnina"
    assert result["engine"] == "gnina"
    assert result["score_type"] == "vina_kcal_mol"


def test_merge_diffdock_confidence_keeps_highest_and_sorts_desc(tmp_path):
    sd0 = tmp_path / "d0"
    sd1 = tmp_path / "d1"
    sd0.mkdir()
    sd1.mkdir()
    _write(sd0 / "scores.csv", ["ligand", "name", "diffdock_confidence", "engine"], [
        {"ligand": "l0", "name": "A", "diffdock_confidence": "0.2", "engine": "diffdock"},
        {"ligand": "l1", "name": "B", "diffdock_confidence": "0.7", "engine": "diffdock"},
    ])
    _write(sd1 / "scores.csv", ["ligand", "name", "diffdock_confidence", "engine"], [
        {"ligand": "l0", "name": "A", "diffdock_confidence": "0.9", "engine": "diffdock"},
    ])

    out = tmp_path / "output"
    result = merge_shard_results([sd0, sd1], out, engine="diffdock")

    rows = list(csv.DictReader((out / "scores.csv").open()))
    assert [r["name"] for r in rows] == ["A", "B"]
    assert rows[0]["diffdock_confidence"] == "0.9"
    assert result["score_type"] == "diffdock_confidence"


def test_build_consensus_results_averages_normalized_ranks(tmp_path):
    u = tmp_path / "unidock"
    g = tmp_path / "gnina"
    u.mkdir()
    g.mkdir()
    _write(u / "scores.csv", ["ligand", "name", "smiles", "score"], [
        {"ligand": "u0", "name": "A", "smiles": "CCO", "score": "-9.0"},
        {"ligand": "u1", "name": "B", "smiles": "CCN", "score": "-8.0"},
        {"ligand": "u2", "name": "C", "smiles": "CCC", "score": "-7.0"},
    ])
    _write(g / "scores.csv", ["ligand", "name", "smiles", "score"], [
        {"ligand": "g0", "name": "A", "smiles": "CCO", "score": "-6.0"},
        {"ligand": "g1", "name": "B", "smiles": "CCN", "score": "-10.0"},
        {"ligand": "g2", "name": "C", "smiles": "CCC", "score": "-7.0"},
    ])
    (u / "results_meta.json").write_text(json.dumps({"engine": "unidock", "score_col": "score"}))
    (g / "results_meta.json").write_text(json.dumps({"engine": "gnina", "score_col": "score"}))

    out = tmp_path / "consensus"
    result = build_consensus_results([u, g], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    meta = json.loads((out / "results_meta.json").read_text())
    assert result["engine"] == "consensus"
    assert meta["score_type"] == "consensus_rank"
    assert [r["name"] for r in rows] == ["B", "A", "C"]
    assert rows[0]["consensus_rank"] == "1"
    assert rows[0]["unidock_rank"] == "2"
    assert rows[0]["gnina_rank"] == "1"
    assert rows[1]["unidock_rank"] == "1"
    assert rows[1]["gnina_rank"] == "3"


def test_consensus_keeps_distinct_compounds_with_the_same_smiles(tmp_path):
    u = tmp_path / "unidock"
    g = tmp_path / "gnina"
    u.mkdir()
    g.mkdir()
    for directory, engine in ((u, "unidock"), (g, "gnina")):
        _write(directory / "scores.csv", ["ligand", "name", "smiles", "score"], [
            {"ligand": f"{engine}-0", "name": "record-A", "smiles": "CCO", "score": "-9.0"},
            {"ligand": f"{engine}-1", "name": "record-B", "smiles": "CCO", "score": "-8.0"},
        ])
        (directory / "results_meta.json").write_text(
            json.dumps({"engine": engine, "score_col": "score"})
        )

    out = tmp_path / "consensus"
    build_consensus_results([u, g], out)

    rows = list(csv.DictReader((out / "scores.csv").open()))
    assert [row["name"] for row in rows] == ["record-A", "record-B"]


def test_consensus_uses_average_tie_ranks_and_penalizes_missing_engines(tmp_path):
    u = tmp_path / "unidock"
    g = tmp_path / "gnina"
    u.mkdir()
    g.mkdir()
    _write(u / "scores.csv", ["ligand", "name", "smiles", "score"], [
        {"ligand": "u0", "name": "A", "smiles": "CCO", "score": "-9.0"},
        {"ligand": "u1", "name": "B", "smiles": "CCN", "score": "-9.0"},
        {"ligand": "u2", "name": "C", "smiles": "CCC", "score": "-7.0"},
    ])
    _write(g / "scores.csv", ["ligand", "name", "smiles", "score"], [
        {"ligand": "g0", "name": "A", "smiles": "CCO", "score": "-10.0"},
        {"ligand": "g2", "name": "C", "smiles": "CCC", "score": "-8.0"},
    ])
    (u / "results_meta.json").write_text(json.dumps({"engine": "unidock", "score_col": "score"}))
    (g / "results_meta.json").write_text(json.dumps({"engine": "gnina", "score_col": "score"}))

    out = tmp_path / "consensus"
    build_consensus_results([u, g], out)
    rows = {row["name"]: row for row in csv.DictReader((out / "scores.csv").open())}

    assert rows["A"]["unidock_rank"] == "1.5"
    assert rows["B"]["unidock_rank"] == "1.5"
    assert rows["B"]["gnina_rank"] == ""
    assert rows["B"]["consensus_coverage"] == "1/2"
    assert float(rows["B"]["consensus_norm_rank"]) == 0.625
    assert float(rows["A"]["consensus_norm_rank"]) == 0.125


def test_results_report_includes_consensus_section(tmp_path):
    from ezscreen.results.report_html import write_results_report

    out = tmp_path / "output"
    out.mkdir()
    _write(out / "scores.csv", [
        "consensus_rank", "consensus_norm_rank", "consensus_engines",
        "ligand", "name", "smiles", "unidock_rank", "gnina_rank",
    ], [
        {
            "consensus_rank": "1",
            "consensus_norm_rank": "0.25",
            "consensus_engines": "unidock,gnina",
            "ligand": "l0",
            "name": "A",
            "smiles": "CCO",
            "unidock_rank": "1",
            "gnina_rank": "2",
        },
    ])
    (out / "results_meta.json").write_text(json.dumps({"score_type": "consensus_rank"}))

    html = write_results_report(out / "scores.csv", out / "report.html", cluster=False).read_text()
    assert "Consensus Ranking" in html
    assert "unidock rank" in html


def test_results_report_warns_for_legacy_floor_rankings(tmp_path):
    from ezscreen.results.report_html import write_results_report

    out = tmp_path / "legacy"
    out.mkdir()
    (out / "scores.csv").write_text(
        "ligand,score,name,smiles\nfailed,-20.0,failed,CC\nok,-8.0,ok,CCC\n"
    )

    html = write_results_report(
        out / "scores.csv", out / "report.html", cluster=False
    ).read_text()

    assert "Ranking integrity warning" in html
    assert "archived ranking as quarantined" in html


def test_results_report_escapes_imported_html_values(tmp_path):
    from ezscreen.results.report_html import write_results_report

    out = tmp_path / "output"
    out.mkdir()
    _write(out / "scores.csv", ["ligand", "name", "smiles", "score"], [{
        "ligand": "lig<script>",
        "name": "<img src=x onerror=alert(1)>",
        "smiles": "CCO",
        "score": "-7.5",
    }])
    (out / "results_meta.json").write_text(json.dumps({"score_type": "vina"}))

    html = write_results_report(
        out / "scores.csv",
        out / "report.html",
        run_id='run"><script>alert(1)</script>',
        metadata={"target": "ABL<1>", "owner": "<b>lab</b>"},
        cluster=False,
        annotations={"lig<script>": {"flag": "red", "note": "<script>alert(1)</script>"}},
    ).read_text(encoding="utf-8")

    assert '&lt;img src=x onerror=alert(1)&gt;' in html
    assert 'run&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert 'ABL&lt;1&gt;' in html
    assert '&lt;b&gt;lab&lt;/b&gt;' in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert '<img src=x onerror=alert(1)>' not in html

