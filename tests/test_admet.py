from __future__ import annotations

from rdkit.Chem import MolFromSmiles

from ezscreen.admet.filter import FilterConfig, filter_mol

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def mol(smiles: str):
    m = MolFromSmiles(smiles)
    assert m is not None, f"RDKit could not parse SMILES: {smiles}"
    return m


# ---------------------------------------------------------------------------
# Lipinski Rule of Five
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"           # MW ~180, passes Ro5
CYCLOSPORIN = "CCC1NC(=O)C(CC2=CC=CC=C2)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(C(C)CC)N(C)C(=O)C(CC(C)C)N(C)C(=O)CN(C)C(=O)C(NC(=O)C(CC(C)C)N(C)C(=O)CN1C)C(C)O"  # MW >1000


def test_lipinski_passes_aspirin():
    cfg = FilterConfig(lipinski=True, pains=False, toxicophores=False, veber=False)
    result = filter_mol(mol(ASPIRIN), cfg)
    assert result.passed
    assert result.failures == []


def test_lipinski_fails_cyclosporin():
    cfg = FilterConfig(lipinski=True, pains=False, toxicophores=False, veber=False)
    result = filter_mol(mol(CYCLOSPORIN), cfg)
    assert not result.passed
    # should flag MW and/or HBD/HBA
    assert any("MW" in f or "HBD" in f or "HBA" in f for f in result.failures)


def test_lipinski_off_skips_check():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=False, veber=False)
    result = filter_mol(mol(CYCLOSPORIN), cfg)
    assert result.passed


# ---------------------------------------------------------------------------
# PAINS
# ---------------------------------------------------------------------------

# Rhodanine scaffold — well-known PAINS hit
PAINS_SCAFFOLD = "O=C1CSC(=S)N1c1ccc(O)cc1"

def test_pains_fail_on_known_scaffold():
    cfg = FilterConfig(lipinski=False, pains=True, toxicophores=False, veber=False)
    result = filter_mol(mol(PAINS_SCAFFOLD), cfg)
    assert not result.passed
    assert any("PAINS" in f for f in result.failures)


def test_pains_passes_aspirin():
    cfg = FilterConfig(lipinski=False, pains=True, toxicophores=False, veber=False)
    result = filter_mol(mol(ASPIRIN), cfg)
    assert result.passed


def test_pains_off_skips_check():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=False, veber=False)
    result = filter_mol(mol(PAINS_SCAFFOLD), cfg)
    assert result.passed


# ---------------------------------------------------------------------------
# Brenk toxicophores
# ---------------------------------------------------------------------------

# Nitro-aromatic — flagged by Brenk
NITRO_COMPOUND = "O=[N+]([O-])c1ccccc1"

def test_brenk_fail_on_nitro():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=True, veber=False)
    result = filter_mol(mol(NITRO_COMPOUND), cfg)
    assert not result.passed
    assert any("Toxicophore" in f or "Brenk" in f or "toxicophore" in f.lower() for f in result.failures)


def test_brenk_passes_ibuprofen():
    ibuprofen = "CC(C)Cc1ccc(CC(C)C(=O)O)cc1"
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=True, veber=False)
    result = filter_mol(mol(ibuprofen), cfg)
    assert result.passed


# ---------------------------------------------------------------------------
# Veber
# ---------------------------------------------------------------------------

IBUPROFEN = "CC(C)Cc1ccc(CC(C)C(=O)O)cc1"   # TPSA ~37, RotBonds 5 — passes

# High-rotatable-bond compound: linear PEG-like
HIGH_ROTB = "OCCOCCOCCOCCOCCOCCOCCOCCO"       # 8+ rotatable bonds, TPSA > 140


def test_veber_passes_ibuprofen():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=False, veber=True)
    result = filter_mol(mol(IBUPROFEN), cfg)
    assert result.passed


def test_veber_off_skips_check():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=False, veber=False)
    result = filter_mol(mol(HIGH_ROTB), cfg)
    assert result.passed


# ---------------------------------------------------------------------------
# Egan BBB
# ---------------------------------------------------------------------------

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"     # LogP ~-1.0, TPSA 58 — passes Egan (lower bound -1.5)
HIGH_LOGP = "c1ccc(cc1)c1ccc(cc1)c1ccc(cc1)c1ccc(cc1)C"  # very high LogP


def test_egan_passes_caffeine():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=False, veber=False, egan_bbb=True)
    result = filter_mol(mol(CAFFEINE), cfg)
    assert result.passed


def test_egan_fail_on_high_logp():
    cfg = FilterConfig(lipinski=False, pains=False, toxicophores=False, veber=False, egan_bbb=True)
    result = filter_mol(mol(HIGH_LOGP), cfg)
    assert not result.passed
    assert any("Egan" in f for f in result.failures)


def test_egan_off_by_default():
    default_cfg = FilterConfig()
    assert default_cfg.egan_bbb is False


# ---------------------------------------------------------------------------
# All filters combined
# ---------------------------------------------------------------------------

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # MW 206, LogP 3.5, no PAINS/Brenk flags

def test_all_filters_pass_ibuprofen():
    # Aspirin's acetyl ester is correctly flagged by Brenk — use ibuprofen as a clean reference
    cfg = FilterConfig(lipinski=True, pains=True, toxicophores=True, veber=True, egan_bbb=False)
    result = filter_mol(mol(IBUPROFEN), cfg)
    assert result.passed


def test_filter_result_accumulates_failures():
    cfg = FilterConfig(lipinski=True, pains=True, toxicophores=True, veber=True)
    result = filter_mol(mol(CYCLOSPORIN), cfg)
    assert not result.passed
    assert len(result.failures) >= 1
