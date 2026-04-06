from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

V1_DISCLAIMER = (
    "v1 ADMET is rule-based only — not predictive. "
    "Results reflect simple physicochemical filters, not biological activity."
)

# ---------------------------------------------------------------------------
# Filter definitions
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    lipinski:     bool = True   # Lipinski Rule of Five
    pains:        bool = True   # PAINS alerts
    toxicophores: bool = True   # basic toxicophore patterns
    veber:        bool = True   # Veber oral bioavailability
    egan_bbb:     bool = False  # Egan BBB (off by default — most VS targets aren't CNS)


@dataclass
class FilterResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------

def _check_lipinski(mol) -> list[str]:
    from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors
    from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA
    failures = []
    mw  = MolWt(mol)
    lp  = MolLogP(mol)
    hbd = CalcNumHBD(mol)
    hba = CalcNumHBA(mol)
    if mw  > 500: failures.append(f"MW {mw:.1f} > 500")
    if lp  > 5:   failures.append(f"LogP {lp:.2f} > 5")
    if hbd > 5:   failures.append(f"HBD {hbd} > 5")
    if hba > 10:  failures.append(f"HBA {hba} > 10")
    return failures


def _check_pains(mol) -> list[str]:
    from rdkit.Chem import FilterCatalog
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    entry = catalog.GetFirstMatch(mol)
    if entry:
        return [f"PAINS alert: {entry.GetDescription()}"]
    return []


def _check_toxicophores(mol) -> list[str]:
    from rdkit.Chem import FilterCatalog
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)
    entry = catalog.GetFirstMatch(mol)
    if entry:
        return [f"Toxicophore: {entry.GetDescription()}"]
    return []


def _check_veber(mol) -> list[str]:
    from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
    failures = []
    tpsa = CalcTPSA(mol)
    rotb = CalcNumRotatableBonds(mol)
    if tpsa > 140:  failures.append(f"TPSA {tpsa:.1f} > 140 Å²")
    if rotb > 10:   failures.append(f"RotBonds {rotb} > 10")
    return failures


def _check_egan_bbb(mol) -> list[str]:
    from rdkit.Chem.Descriptors import MolLogP
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    failures = []
    lp   = MolLogP(mol)
    tpsa = CalcTPSA(mol)
    if not (-1 <= lp <= 6):   failures.append(f"Egan BBB: LogP {lp:.2f} out of [-1, 6]")
    if not (0 <= tpsa <= 131): failures.append(f"Egan BBB: TPSA {tpsa:.1f} out of [0, 131]")
    return failures


# ---------------------------------------------------------------------------
# Main filter function
# ---------------------------------------------------------------------------

def filter_mol(mol, cfg: FilterConfig) -> FilterResult:
    all_failures: list[str] = []

    if cfg.lipinski:
        all_failures.extend(_check_lipinski(mol))
    if cfg.pains:
        all_failures.extend(_check_pains(mol))
    if cfg.toxicophores:
        all_failures.extend(_check_toxicophores(mol))
    if cfg.veber:
        all_failures.extend(_check_veber(mol))
    if cfg.egan_bbb:
        all_failures.extend(_check_egan_bbb(mol))

    return FilterResult(passed=len(all_failures) == 0, failures=all_failures)


def filter_library(
    input_path: str,
    output_path: str,
    cfg: FilterConfig | None = None,
) -> dict[str, Any]:
    """
    Filter an SDF file. Returns a summary dict for the prep report.
    Molecules that pass are written to output_path.
    Molecules that fail are counted by rule.
    """
    from rdkit.Chem import SDMolSupplier, SDWriter

    if cfg is None:
        cfg = FilterConfig()

    supplier = SDMolSupplier(str(input_path), removeHs=False, sanitize=True)
    writer   = SDWriter(str(output_path))

    total = passed = 0
    breakdown: dict[str, int] = {
        "ro5_violations":   0,
        "pains_alerts":     0,
        "toxicophores":     0,
        "veber_violations": 0,
        "egan_bbb":         0,
    }

    for mol in supplier:
        if mol is None:
            continue
        total += 1
        result = filter_mol(mol, cfg)
        if result.passed:
            writer.write(mol)
            passed += 1
        else:
            for f in result.failures:
                fl = f.lower()
                if "mw"  in fl or "logp" in fl or "hbd" in fl or "hba" in fl:
                    breakdown["ro5_violations"]   += 1
                elif "pains" in fl:
                    breakdown["pains_alerts"]      += 1
                elif "toxicophore" in fl or "brenk" in fl:
                    breakdown["toxicophores"]      += 1
                elif "tpsa" in fl or "rotbond" in fl:
                    breakdown["veber_violations"]  += 1
                elif "egan" in fl:
                    breakdown["egan_bbb"]          += 1

    writer.close()
    removed = total - passed

    return {
        "total_input":    total,
        "admet_removed":  removed,
        "admet_breakdown": breakdown,
        "disclaimer":     V1_DISCLAIMER,
    }
