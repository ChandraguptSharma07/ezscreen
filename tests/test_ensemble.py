from pathlib import Path

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
        [a, b],
        output_dir=tmp_path / "ensemble",
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
        ensemble,
        "_pdb_derived_candidates",
        lambda *_args: [candidate],
    )
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args: (_ for _ in ()).throw(AssertionError("NMA should not run")),
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

    monkeypatch.setattr(ensemble, "_pdb_derived_candidates", lambda *_args: [])
    monkeypatch.setattr(ensemble, "_nma_fallback_candidates", lambda *_args: [nma])

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

    monkeypatch.setattr(ensemble, "_pdb_derived_candidates", lambda *_args: [bad, good])
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args: (_ for _ in ()).throw(AssertionError("NMA should not run")),
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

    monkeypatch.setattr(ensemble, "_pdb_derived_candidates", lambda *_args: [bad])
    monkeypatch.setattr(ensemble, "_nma_fallback_candidates", lambda *_args: [nma])

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


def test_generate_auto_ensemble_returns_empty_when_all_candidates_fail(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad = tmp_path / "bad.pdb"
    bad.write_text("ATOM BAD\nEND\n")

    class AllFailRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            raise ReceptorPrepError("all bad")

    monkeypatch.setattr(ensemble, "_pdb_derived_candidates", lambda *_args: [bad])
    monkeypatch.setattr(ensemble, "_nma_fallback_candidates", lambda *_args: [bad])

    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=AllFailRecPrep,
    )

    assert conformers == []


def test_generate_auto_ensemble_skips_qc_rejected_candidates(tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")
    bad = tmp_path / "bad_qc.pdb"
    bad.write_text("ATOM BAD\nEND\n")
    good = tmp_path / "good_qc.pdb"
    good.write_text("ATOM GOOD\nEND\n")

    monkeypatch.setattr(ensemble, "_pdb_derived_candidates", lambda *_args: [bad, good])
    monkeypatch.setattr(
        ensemble,
        "_nma_fallback_candidates",
        lambda *_args: (_ for _ in ()).throw(AssertionError("NMA should not run")),
    )

    def fake_qc(_ref, conf_pdb, _chains, _site):
        if "prep_002" in str(conf_pdb):
            return True, {"qc_status": "accepted", "qc_reason": "pocket_ca_rmsd"}
        return False, {"qc_status": "rejected", "qc_reason": "near_duplicate"}

    monkeypatch.setattr(ensemble, "_passes_auto_qc", fake_qc)

    conformers = ensemble.generate_auto_ensemble(
        receptor,
        ["A"],
        {"type": "manual"},
        max_conformers=3,
        output_dir=tmp_path / "auto",
        rec_prep_module=FakeRecPrep,
    )

    assert len(conformers) == 1
    assert conformers[0].metadata["qc_status"] == "accepted"
    assert conformers[0].metadata["candidate_path"].endswith("good_qc.pdb")
