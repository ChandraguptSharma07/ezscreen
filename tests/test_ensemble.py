from pathlib import Path

import numpy as np
import pytest

from ezscreen.errors import ReceptorPrepError
from ezscreen.prep import ensemble


class FakeRecPrep:
    calls: list[dict] = []

    @classmethod
    def prep_receptor(cls, **kwargs):
        cls.calls.append(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        pdbqt = out / "prepared.pdbqt"
        pdbqt.write_text(f"PREPARED {Path(kwargs['pdb_path']).name}\n")
        fixed = out / "fixed.pdb"
        fixed.write_text("ATOM\nEND\n")
        return {"pdbqt_path": pdbqt, "fixed_pdb_path": fixed}


def test_prepare_manual_ensemble_copies_pdbqt_and_preps_pdb(tmp_path):
    FakeRecPrep.calls = []
    pdbqt = tmp_path / "conf_a.pdbqt"
    pdbqt.write_text("PDBQT A\n")
    pdb = tmp_path / "conf_b.pdb"
    pdb.write_text("ATOM\nEND\n")
    conformers = ensemble.prepare_manual_ensemble(
        [pdbqt, pdb],
        output_dir=tmp_path / "ensemble",
        chains=["A"],
        ph=7.4,
        rec_prep_module=FakeRecPrep,
    )
    assert [c.source for c in conformers] == ["manual_pdbqt", "manual_pdb"]
    assert conformers[0].pdbqt_path.read_text() == "PDBQT A\n"
    assert conformers[1].pdbqt_path.read_text() == "PREPARED conf_b.pdb\n"
    assert FakeRecPrep.calls[0]["pdb_path"] == pdb
    assert FakeRecPrep.calls[0]["receptor_isomers"] == "off"


def test_prepare_manual_ensemble_dedupes_by_pdbqt_hash(tmp_path):
    a = tmp_path / "a.pdbqt"
    b = tmp_path / "b.pdbqt"
    a.write_text("SAME\n")
    b.write_text("SAME\n")
    conformers = ensemble.prepare_manual_ensemble(
        [a, b], output_dir=tmp_path / "ensemble"
    )
    assert len(conformers) == 1
    assert conformers[0].metadata["pdbqt_sha256"]


def test_generate_auto_ensemble_uses_pdb_candidates(tmp_path, monkeypatch):
    FakeRecPrep.calls = []
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    candidate = tmp_path / "candidate.pdb"
    candidate.write_text("ATOM CANDIDATE\nEND\n")
    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [candidate]
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NMA should not run")
        ),
    )
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=FakeRecPrep,
    )
    assert len(conformers) == 1
    assert conformers[0].source == "auto_pdb"
    assert conformers[0].metadata["method"] == "pdb_then_nma"
    assert conformers[0].metadata["fallback_used"] is False


def test_generate_auto_ensemble_falls_back_to_nma(tmp_path, monkeypatch):
    FakeRecPrep.calls = []
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    nma = tmp_path / "nma.pdb"
    nma.write_text("ATOM NMA\nEND\n")
    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        ensemble, "_nma_fallback_candidates", lambda *_args, **_kwargs: [nma]
    )
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=FakeRecPrep,
    )
    assert len(conformers) == 1
    assert conformers[0].source == "auto_nma"
    assert conformers[0].metadata["fallback_used"] is True


def test_generate_auto_ensemble_skips_bad_pdb_candidates(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad = tmp_path / "bad.pdb"
    bad.write_text("ATOM BAD\nEND\n")
    good = tmp_path / "good.pdb"
    good.write_text("ATOM GOOD\nEND\n")

    class FlakyRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            if Path(kwargs["pdb_path"]).name == "bad.pdb":
                raise ReceptorPrepError("bad conformer")
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text(f"PREPARED {Path(kwargs['pdb_path']).name}\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": kwargs["pdb_path"]}

    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [bad, good]
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NMA should not run")
        ),
    )
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=FlakyRecPrep,
    )
    assert len(conformers) == 1
    assert conformers[0].source == "auto_pdb"
    assert conformers[0].metadata["candidate_path"].endswith("good.pdb")


def test_generate_auto_ensemble_checks_extra_pdb_candidates_before_fallback(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad_a = tmp_path / "bad_a.pdb"
    bad_b = tmp_path / "bad_b.pdb"
    good_a = tmp_path / "good_a.pdb"
    good_b = tmp_path / "good_b.pdb"
    for path_obj in (bad_a, bad_b, good_a, good_b):
        path_obj.write_text(f"ATOM {path_obj.stem}\nEND\n")

    class CandidateNamedRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text(f"PREPARED {Path(kwargs['pdb_path']).name}\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": kwargs["pdb_path"]}

    monkeypatch.setattr(
        ensemble,
        "_pdb_derived_candidates",
        lambda *_args, **_kwargs: [bad_a, bad_b, good_a, good_b],
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NMA should not run")
        ),
    )

    def fake_qc(_ref, conf_pdb, _chains, _site):
        if Path(conf_pdb).stem.startswith("good_"):
            return True, {"qc_status": "accepted", "qc_reason": "pocket_ca_rmsd"}
        return False, {"qc_status": "rejected", "qc_reason": "near_duplicate"}

    monkeypatch.setattr(ensemble, "_passes_auto_qc", fake_qc)
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=2,
        output_dir=tmp_path / "auto",
        rec_prep_module=CandidateNamedRecPrep,
    )

    assert {Path(c.metadata["candidate_path"]).name for c in conformers} == {
        "good_a.pdb",
        "good_b.pdb",
    }
    assert len(conformers) == 2


def test_generate_auto_ensemble_prefers_qc_accepted_conformers_over_skipped(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    weak = tmp_path / "weak.pdb"
    good_a = tmp_path / "good_a.pdb"
    good_b = tmp_path / "good_b.pdb"
    for path_obj in (weak, good_a, good_b):
        path_obj.write_text(f"ATOM {path_obj.stem}\nEND\n")

    class CandidateNamedRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text(f"PREPARED {Path(kwargs['pdb_path']).name}\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": kwargs["pdb_path"]}

    monkeypatch.setattr(
        ensemble,
        "_pdb_derived_candidates",
        lambda *_args, **_kwargs: [weak, good_a, good_b],
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NMA should not run")
        ),
    )

    def fake_qc(_ref, conf_pdb, _chains, _site):
        name = Path(conf_pdb).name
        if name == "weak.pdb":
            return True, {
                "qc_status": "skipped",
                "qc_reason": "insufficient_pocket_atoms",
                "qc_matched_ca": 1,
            }
        return True, {
            "qc_status": "accepted",
            "qc_reason": "pocket_ca_rmsd",
            "qc_matched_ca": 25,
            "pocket_ca_rmsd": 1.7 if name == "good_a.pdb" else 1.9,
        }

    monkeypatch.setattr(ensemble, "_passes_auto_qc", fake_qc)
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=2,
        output_dir=tmp_path / "auto",
        rec_prep_module=CandidateNamedRecPrep,
    )

    assert {Path(c.metadata["candidate_path"]).name for c in conformers} == {
        "good_a.pdb",
        "good_b.pdb",
    }


def test_generate_auto_ensemble_falls_back_when_pdb_prep_fails(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad = tmp_path / "bad.pdb"
    bad.write_text("ATOM BAD\nEND\n")
    nma = tmp_path / "nma.pdb"
    nma.write_text("ATOM NMA\nEND\n")

    class PdbFailsRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            name = Path(kwargs["pdb_path"]).name
            if name == "bad.pdb":
                raise ReceptorPrepError("bad pdb candidate")
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text(f"PREPARED {name}\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": kwargs["pdb_path"]}

    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [bad]
    )
    monkeypatch.setattr(
        ensemble, "_nma_fallback_candidates", lambda *_args, **_kwargs: [nma]
    )
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=PdbFailsRecPrep,
    )
    assert len(conformers) == 1
    assert conformers[0].source == "auto_nma"
    assert conformers[0].metadata["fallback_used"] is True


def test_generate_auto_ensemble_returns_empty_when_all_candidates_fail(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad = tmp_path / "bad.pdb"
    bad.write_text("ATOM BAD\nEND\n")

    class AllFailRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            raise ReceptorPrepError("all bad")

    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [bad]
    )
    monkeypatch.setattr(
        ensemble, "_nma_fallback_candidates", lambda *_args, **_kwargs: [bad]
    )
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=AllFailRecPrep,
    )
    assert conformers == []


def test_pocket_ca_rmsd_falls_back_to_sequence_alignment_for_renumbered_residues(
    tmp_path,
):
    reference = tmp_path / "reference.pdb"
    reference.write_text(
        "\n".join(
            [
                "ATOM      1  CA  GLY A 101       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A 102       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  SER A 103       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  TYR A 104       3.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n"
    )
    conformer = tmp_path / "conformer.pdb"
    conformer.write_text(
        "\n".join(
            [
                "ATOM      1  CA  GLY A   1       0.100   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  ALA A   2       1.100   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  SER A   3       2.100   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  TYR A   4       3.100   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n"
    )

    rmsd, matched = ensemble._pocket_ca_rmsd(reference, conformer, (1.5, 0.0, 0.0))

    assert matched == 4
    assert rmsd == pytest.approx(0.1)


def test_generate_auto_ensemble_qc_checks_candidate_geometry_when_prep_renumbers(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    candidate = tmp_path / "aligned_good.pdb"
    candidate.write_text("ATOM GOOD\nEND\n")

    class RenumberingRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text("PREPARED\n")
            fixed = out / "receptor_fixed.pdb"
            fixed.write_text("ATOM RENNUMBERED\nEND\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": fixed}

    seen = {"conf_name": None}

    def fake_qc(_ref, conf_pdb, _chains, _site):
        seen["conf_name"] = Path(conf_pdb).name
        return True, {
            "qc_status": "accepted",
            "qc_reason": "pocket_ca_rmsd",
            "qc_matched_ca": 25,
            "pocket_ca_rmsd": 1.7,
        }

    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [candidate]
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NMA should not run")
        ),
    )
    monkeypatch.setattr(ensemble, "_passes_auto_qc", fake_qc)

    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=2,
        output_dir=tmp_path / "auto",
        rec_prep_module=RenumberingRecPrep,
    )

    assert len(conformers) == 1
    assert seen["conf_name"] == "aligned_good.pdb"


def test_generate_auto_ensemble_skips_qc_rejected_candidates(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad = tmp_path / "bad_qc.pdb"
    bad.write_text("ATOM BAD\nEND\n")
    good = tmp_path / "good_qc.pdb"
    good.write_text("ATOM GOOD\nEND\n")
    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [bad, good]
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NMA should not run")
        ),
    )

    def fake_qc(_ref, conf_pdb, _chains, _site):
        if Path(conf_pdb).name == "good_qc.pdb":
            return True, {"qc_status": "accepted", "qc_reason": "pocket_ca_rmsd"}
        return False, {"qc_status": "rejected", "qc_reason": "near_duplicate"}

    class CandidateNamedRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text(f"PREPARED {Path(kwargs['pdb_path']).name}\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": kwargs["pdb_path"]}

    monkeypatch.setattr(ensemble, "_passes_auto_qc", fake_qc)
    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=CandidateNamedRecPrep,
    )
    assert len(conformers) == 1
    assert conformers[0].metadata["qc_status"] == "accepted"
    assert conformers[0].metadata["candidate_path"].endswith("good_qc.pdb")


def test_pdb_candidates_extract_additional_models(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "MODEL        1\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C\n"
        "ENDMDL\n"
        "MODEL        3\n"
        "ATOM      1  CA  ALA A   1       2.000   0.000   0.000  1.00 20.00           C\n"
        "ENDMDL\n"
        "END\n"
    )
    paths = ensemble._pdb_derived_candidates(
        receptor, ["A"], {"type": "manual"}, tmp_path / "cands", 3
    )
    assert len(paths) == 2
    assert [p.name for p in paths] == ["input_model_002.pdb", "input_model_003.pdb"]


def test_search_rcsb_sequence_candidates_accepts_string_result_items(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result_set": ["1ABC", {"identifier": "2DEF"}, "2DEF", "BADLEN"]}

    class FakeRequests:
        @staticmethod
        def post(*args, **kwargs):
            return FakeResponse()

    monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)

    hits = ensemble._search_rcsb_sequence_candidates("A" * 50, exclude_ids={"1ABC"})

    assert hits == ["2DEF"]


def test_shared_residue_pairs_aligns_shifted_numbering_by_sequence():
    reference = [
        {"resname": "ALA", "resseq": 10},
        {"resname": "GLY", "resseq": 11},
        {"resname": "SER", "resseq": 12},
        {"resname": "TYR", "resseq": 13},
    ]
    candidate = [
        {"resname": "GLY", "resseq": 101},
        {"resname": "SER", "resseq": 102},
        {"resname": "TYR", "resseq": 103},
        {"resname": "LYS", "resseq": 104},
    ]

    pairs = ensemble._shared_residue_pairs(reference, candidate)

    assert [(a["resseq"], b["resseq"]) for a, b in pairs] == [
        (11, 101),
        (12, 102),
        (13, 103),
    ]


def test_infer_pocket_chains_prefers_site_local_chain_in_multichain_complex(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A   2       1.500   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  SER B   1      25.000  25.000  25.000  1.00 20.00           C\n"
        "END\n"
    )
    site = {
        "type": "cocrystal",
        "ligands": [{"chain": "A", "coords": [(0.5, 0.2, 0.1), (1.0, 0.0, 0.0)]}],
    }

    pocket_chains = ensemble._infer_pocket_chains(receptor, ["A", "B"], site)

    assert pocket_chains == ["A"]


def test_pdb_candidates_use_site_local_chain_even_when_multiple_chains_selected(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "1abc.pdb"
    receptor.write_text("ATOM\nEND\n")
    seen = {"query_chains": None, "align_chains": []}

    monkeypatch.setattr(
        ensemble, "_extract_additional_models", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        ensemble, "_infer_pocket_chains", lambda *_args, **_kwargs: ["A"]
    )

    def fake_infer_query_sequence(_pdb, chains):
        seen["query_chains"] = list(chains)
        return "A" * 50

    monkeypatch.setattr(ensemble, "_infer_query_sequence", fake_infer_query_sequence)
    monkeypatch.setattr(
        ensemble, "_search_rcsb_sequence_candidates", lambda *_args, **_kwargs: ["2DEF"]
    )
    monkeypatch.setattr(
        ensemble, "_site_center_from_details", lambda *_args, **_kwargs: (0.0, 0.0, 0.0)
    )

    def fake_download(entry_id, output_dir):
        path = Path(output_dir) / f"{entry_id}.pdb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("REMARK 2DEF\n")
        return path

    monkeypatch.setattr(ensemble, "_download_rcsb_pdb", fake_download)

    def fake_align(_reference_pdb, candidate_pdb, chains, output_dir, _site_details):
        seen["align_chains"].append(list(chains))
        aligned = Path(output_dir) / f"aligned_{candidate_pdb.stem}.pdb"
        aligned.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_text("REMARK aligned\n")
        return ensemble._AlignedCandidate(
            aligned, "remote_pdb", 1.6, {"alignment_residues": 20}
        )

    monkeypatch.setattr(ensemble, "_align_pdb_candidate", fake_align)
    monkeypatch.setattr(ensemble, "_pocket_ca_rmsd", lambda *_args, **_kwargs: (0.8, 8))

    paths = ensemble._pdb_derived_candidates(
        receptor,
        ["A", "B"],
        {"type": "cocrystal", "ligands": [{"chain": "A", "coords": [(0.0, 0.0, 0.0)]}]},
        tmp_path / "cands",
        2,
    )

    assert seen["query_chains"] == ["A"]
    assert seen["align_chains"] == [["A"]]
    assert [p.name for p in paths] == ["aligned_2DEF.pdb"]


def test_pdb_candidates_skip_remote_harvest_for_interface_spanning_sites(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "1abc.pdb"
    receptor.write_text("ATOM\nEND\n")
    monkeypatch.setattr(
        ensemble, "_extract_additional_models", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        ensemble, "_infer_pocket_chains", lambda *_args, **_kwargs: ["A", "B"]
    )
    called = {"search": 0}

    def fake_search(*args, **kwargs):
        called["search"] += 1
        return ["2DEF"]

    monkeypatch.setattr(ensemble, "_search_rcsb_sequence_candidates", fake_search)

    paths = ensemble._pdb_derived_candidates(
        receptor,
        ["A", "B"],
        {"type": "manual", "center": [0.0, 0.0, 0.0]},
        tmp_path / "cands",
        2,
    )

    assert paths == []
    assert called["search"] == 0


def test_pdb_candidates_search_related_entries_and_keep_diverse_subset(
    tmp_path, monkeypatch
):
    receptor = tmp_path / "1abc.pdb"
    receptor.write_text("ATOM\nEND\n")
    monkeypatch.setattr(
        ensemble, "_extract_additional_models", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        ensemble, "_infer_query_sequence", lambda *_args, **_kwargs: "A" * 50
    )
    monkeypatch.setattr(
        ensemble,
        "_search_rcsb_sequence_candidates",
        lambda *_args, **_kwargs: ["2DEF", "3GHI", "4JKL"],
    )
    downloaded: list[str] = []

    def fake_download(entry_id, output_dir):
        downloaded.append(entry_id)
        path = Path(output_dir) / f"{entry_id}.pdb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"REMARK {entry_id}\n")
        return path

    monkeypatch.setattr(ensemble, "_download_rcsb_pdb", fake_download)
    monkeypatch.setattr(
        ensemble, "_site_center_from_details", lambda *_args, **_kwargs: (0.0, 0.0, 0.0)
    )
    rmsd_map = {"2DEF": 1.4, "3GHI": 1.9, "4JKL": 2.4}

    def fake_align(_reference_pdb, candidate_pdb, _chains, output_dir, _site_details):
        entry_id = candidate_pdb.stem
        aligned = Path(output_dir) / f"aligned_{entry_id}.pdb"
        aligned.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_text(f"REMARK aligned {entry_id}\n")
        return ensemble._AlignedCandidate(
            aligned, "remote_pdb", rmsd_map[entry_id], {"alignment_residues": 25}
        )

    monkeypatch.setattr(ensemble, "_align_pdb_candidate", fake_align)
    pairwise = {
        frozenset({"aligned_2DEF.pdb", "aligned_3GHI.pdb"}): (0.20, 8),
        frozenset({"aligned_2DEF.pdb", "aligned_4JKL.pdb"}): (0.90, 8),
        frozenset({"aligned_3GHI.pdb", "aligned_4JKL.pdb"}): (0.85, 8),
    }

    def fake_pairwise(a, b, _center, radius=ensemble._AUTO_QC_POCKET_RADIUS):
        if a == receptor or b == receptor:
            return (1.5, 8)
        return pairwise[frozenset({Path(a).name, Path(b).name})]

    monkeypatch.setattr(ensemble, "_pocket_ca_rmsd", fake_pairwise)
    paths = ensemble._pdb_derived_candidates(
        receptor, ["A"], {"type": "manual"}, tmp_path / "cands", 2
    )
    assert downloaded == ["2DEF", "3GHI", "4JKL"]
    assert [p.name for p in paths] == ["aligned_4JKL.pdb", "aligned_3GHI.pdb"]


def test_generate_auto_ensemble_logs_candidate_flow(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    dup_a = tmp_path / "dup_a.pdb"
    dup_b = tmp_path / "dup_b.pdb"
    dup_a.write_text("ATOM A\nEND\n")
    dup_b.write_text("ATOM B\nEND\n")
    messages: list[str] = []

    class SamePrepRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            out = Path(kwargs["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            pdbqt = out / "prepared.pdbqt"
            pdbqt.write_text("SAME\n")
            return {"pdbqt_path": pdbqt, "fixed_pdb_path": kwargs["pdb_path"]}

    monkeypatch.setattr(
        ensemble, "_infer_pocket_chains", lambda *_args, **_kwargs: ["A"]
    )
    monkeypatch.setattr(
        ensemble, "_pdb_derived_candidates", lambda *_args, **_kwargs: [dup_a, dup_b]
    )
    monkeypatch.setattr(
        ensemble, "_nma_fallback_candidates", lambda *_args, **_kwargs: []
    )

    def fake_qc(_ref, conf_pdb, _chains, _site):
        if Path(conf_pdb).name == "dup_a.pdb":
            return False, {"qc_reason": "near_duplicate"}
        return True, {"qc_reason": "pocket_ca_rmsd", "qc_status": "accepted"}

    monkeypatch.setattr(ensemble, "_passes_auto_qc", fake_qc)

    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=SamePrepRecPrep,
        logger=messages.append,
    )

    assert len(conformers) == 1
    assert any("using prep_chains=A" in msg for msg in messages)
    assert any("source=pdb yielded 2 candidate(s)" in msg for msg in messages)
    assert any("QC rejected conformer_001_dup_a" in msg for msg in messages)
    assert any("QC accepted conformer_002_dup_b" in msg for msg in messages)
    assert any(
        "source=pdb kept 1 conformer(s) after prep/QC/dedupe" in msg for msg in messages
    )


def test_nma_fallback_accepts_coordsets_without_truthiness(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")

    class AmbiguousCoordsets(list):
        def __bool__(self):
            raise ValueError("ambiguous truth value")

    class FakeSelection:
        def __init__(self):
            self.coords = np.array([(0.0, 0.0, 0.0)], dtype=float)
            self.resindices = [0]

        def __len__(self):
            return 5

        def setCoords(self, coords):
            self.coords = coords

        def getCoords(self):
            return self.coords

        def getResindices(self):
            return self.resindices

    class FakeAtoms:
        def __init__(self):
            self.ca = FakeSelection()
            self.coords = np.array([(0.0, 0.0, 0.0)], dtype=float)
            self.resindices = [0]

        def select(self, query):
            return self.ca

        def copy(self):
            copied = FakeAtoms()
            copied.coords = list(self.coords)
            copied.resindices = list(self.resindices)
            return copied

        def getCoords(self):
            return self.coords

        def setCoords(self, coords):
            self.coords = np.array(coords, dtype=float)

        def getResindices(self):
            return self.resindices

    class FakeTraversal:
        def getCoordsets(self):
            return AmbiguousCoordsets(
                [
                    [(0.0, 0.0, 0.0)],
                    [(1.0, 1.0, 1.0)],
                ]
            )

    class FakeANM:
        def __init__(self, _name):
            self.modes = [object()]

        def buildHessian(self, _selection):
            return None

        def calcModes(self, n_modes=1):
            self.modes = [object()] * n_modes

        def __getitem__(self, item):
            return self.modes[item]

    class FakePrody:
        @staticmethod
        def parsePDB(_path):
            return FakeAtoms()

        @staticmethod
        def ANM(name):
            return FakeANM(name)

        @staticmethod
        def traverseMode(_mode, _selection, rmsd=1.0, n_steps=3):
            return FakeTraversal()

        @staticmethod
        def writePDB(path, _moved):
            Path(path).write_text("ATOM\nEND\n")

    monkeypatch.setitem(__import__("sys").modules, "prody", FakePrody)

    out_dir = tmp_path / "nma"
    paths = ensemble._nma_fallback_candidates(
        receptor,
        ["A"],
        {"type": "manual"},
        out_dir,
        1,
    )

    assert len(paths) == 1
    assert paths[0].name == "nma_mode_001_neg.pdb"
    assert "MARKED-COPY" not in paths[0].read_text().upper()
