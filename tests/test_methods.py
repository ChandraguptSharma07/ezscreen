from __future__ import annotations

from ezscreen.results.methods import build_methods_text

_RUN_META = {
    "version": "1.11.0",
    "receptor": {
        "pdb_id": "1HSG",
        "is_alphafold": False,
        "chains": ["A"],
        "source": "rcsb",
    },
    "binding_site": {
        "method": "p2rank",
        "center": [12.3, -4.5, 30.0],
        "size": [20.0, 20.0, 20.0],
    },
    "ligands": {
        "total_input": 1000,
        "admet_applied": True,
        "protonation_ph": 7.4,
        "force_field": "MMFF94",
    },
    "docking": {
        "engine": "UniDock",
        "search_mode": "balance",
        "backend": "Kaggle GPU",
    },
}


def test_methods_contains_key_tokens():
    text = build_methods_text(_RUN_META)

    # receptor + box
    assert "1HSG" in text
    assert "12.3" in text  # box centre

    # engine
    assert "UniDock" in text

    # tool citations
    assert "Yu et al., 2023" in text          # UniDock
    assert "Krivak & Hoksza, 2018" in text    # P2Rank (pocket method was p2rank)
    assert "Salentin et al., 2015" in text    # PLIP
    assert "RDKit" in text
    assert "Meeko" in text

    assert "ezscreen v1.11.0" in text


def test_p2rank_citation_only_when_used():
    cocrystal = dict(_RUN_META)
    cocrystal["binding_site"] = {
        "method": "cocrystal",
        "reference_ligand": "MK1",
        "center": [1.0, 2.0, 3.0],
        "size": [18.0, 18.0, 18.0],
    }
    text = build_methods_text(cocrystal)
    assert "Krivak & Hoksza" not in text
    assert "co-crystallised ligand MK1" in text


def test_alphafold_phrasing():
    af = dict(_RUN_META)
    af["receptor"] = {
        "is_alphafold": True,
        "af_accession": "P0DTC2",
        "af_version": 4,
        "chains": [],
    }
    text = build_methods_text(af)
    assert "AlphaFold model for P0DTC2" in text
    assert "v4" in text
