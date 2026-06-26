from __future__ import annotations

from rdkit import Chem

from ezscreen.prep import enumerate as enum

ACETIC_ACID = "CC(=O)O"


def test_returns_input_when_gypsum_unavailable(monkeypatch):
    # Gypsum-DL not locatable → graceful degradation to the single input form.
    monkeypatch.setattr(enum, "_gypsum_command", lambda: None)
    assert enum.enumerate_variants(ACETIC_ACID) == [ACETIC_ACID]
    assert enum.gypsum_available() is False


def test_returns_input_when_subprocess_fails(monkeypatch):
    # Gypsum-DL "present" but the run errors → still falls back to [smiles].
    monkeypatch.setattr(enum, "_gypsum_command", lambda: ["gypsum-dl"])

    def _boom(*a, **k):
        raise RuntimeError("gypsum exploded")

    monkeypatch.setattr(enum.subprocess, "run", _boom)
    assert enum.enumerate_variants(ACETIC_ACID) == [ACETIC_ACID]


def test_collect_variants_dedupes_from_sdf(tmp_path):
    # Two distinct protomers in the output → two unique SMILES collected.
    out = tmp_path / "out"
    out.mkdir()
    writer = Chem.SDWriter(str(out / "src0.sdf"))
    for smi in ("CC(=O)O", "CC(=O)[O-]", "CC(=O)O"):  # last is a duplicate
        writer.write(Chem.MolFromSmiles(smi))
    writer.close()

    variants = enum._collect_variants(out, fallback=ACETIC_ACID)
    assert len(variants) == 2
    assert "CC(=O)[O-]" in variants


def test_build_args_honours_skip_flags():
    cmd = ["gypsum-dl"]
    args = enum._build_args(
        cmd, "in.smi", "out", 7.4, 4,
        protonation=True, tautomers=False, stereo=False, ring=True,
    )
    assert "--2d_output_only" in args
    assert "--skip_making_tautomers" in args
    assert "--skip_enumerate_chiral_mol" in args
    assert "--skip_alternate_ring_conformations" not in args  # ring enabled
    assert "--max_variants_per_compound" in args
    # protonation on → a pH window straddling 7.4
    assert "--min_ph" in args and "--max_ph" in args
