from __future__ import annotations

from ezscreen.results.variants import (
    collapse_variants,
    has_variants,
    source_name,
)

# best-first, as scores.csv is written
_ROWS = [
    {"ligand": "nb0_lig_00001", "name": "Z999_v3", "score": "-7.8"},
    {"ligand": "nb0_lig_00002", "name": "Z999_v1", "score": "-7.1"},
    {"ligand": "nb0_lig_00003", "name": "Z111",    "score": "-7.0"},
    {"ligand": "nb0_lig_00004", "name": "Z999_v2", "score": "-6.5"},
]


def test_source_name_strips_variant_suffix():
    assert source_name("Z999_v3") == "Z999"
    assert source_name("Z111") == "Z111"
    assert source_name("") == ""


def test_has_variants_detects_suffix():
    assert has_variants(_ROWS) is True
    assert has_variants([{"name": "a"}, {"name": "b"}]) is False


def test_collapse_keeps_best_per_source_and_counts():
    out = collapse_variants(_ROWS)
    # Z999 (3 forms) + Z111 (1 form) → 2 rows
    assert len(out) == 2
    z999 = next(r for r in out if source_name(r["name"]) == "Z999")
    # best-first input → first form seen (the -7.8 one) is kept
    assert z999["name"] == "Z999_v3"
    assert z999["variant_count"] == 3
    # rank order preserved: Z999 before Z111
    assert [source_name(r["name"]) for r in out] == ["Z999", "Z111"]
