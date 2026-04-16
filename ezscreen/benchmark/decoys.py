from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import requests

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

_MW_WIN    = 25.0
_LOGP_WIN  = 1.5
_HBD_WIN   = 1
_HBA_WIN   = 2
_ROTB_WIN  = 2
_TC_CUTOFF = 0.35


@dataclass
class _Props:
    smiles: str
    mw: float
    logp: float
    hbd: int
    hba: int
    rotb: int
    charge: int


def _props(smiles: str) -> _Props | None:
    from rdkit import Chem
    from rdkit.Chem.Descriptors import MolLogP, MolWt
    from rdkit.Chem.rdMolDescriptors import (
        CalcNumHBA,
        CalcNumHBD,
        CalcNumRotatableBonds,
    )
    from rdkit.Chem.rdmolops import GetFormalCharge

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return _Props(
            smiles=smiles,
            mw=MolWt(mol),
            logp=MolLogP(mol),
            hbd=CalcNumHBD(mol),
            hba=CalcNumHBA(mol),
            rotb=CalcNumRotatableBonds(mol),
            charge=GetFormalCharge(mol),
        )
    except Exception:
        return None


def _fetch_from_chembl(p: _Props, n: int) -> list[str]:
    params = {
        "molecular_weight__gte": p.mw - _MW_WIN,
        "molecular_weight__lte": p.mw + _MW_WIN,
        "alogp__gte": p.logp - _LOGP_WIN,
        "alogp__lte": p.logp + _LOGP_WIN,
        "hbd__lte": p.hbd + _HBD_WIN,
        "hba__lte": p.hba + _HBA_WIN,
        "rtb__lte": p.rotb + _ROTB_WIN,
        "molecule_type": "Small molecule",
        "limit": min(n, 100),
        "offset": random.randint(0, 500),
        "format": "json",
    }
    try:
        resp = requests.get(_CHEMBL_API, params=params, timeout=20)
        resp.raise_for_status()
        out = []
        for entry in resp.json().get("molecules", []):
            structs = entry.get("molecule_structures") or {}
            smi = structs.get("canonical_smiles")
            if smi:
                out.append(smi)
        return out
    except Exception:
        return []


def _filter_by_tanimoto(active_smi: str, candidates: list[str], n: int) -> list[str]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(active_smi)
    if mol is None:
        return []
    active_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    kept = []
    random.shuffle(candidates)
    for smi in candidates:
        if len(kept) >= n:
            break
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        if DataStructs.TanimotoSimilarity(active_fp, fp) < _TC_CUTOFF:
            kept.append(smi)
    return kept


def generate_decoys(
    actives: Sequence[str],
    n_per_active: int = 50,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for smi in actives:
        p = _props(smi)
        if p is None:
            result[smi] = []
            continue
        candidates = _fetch_from_chembl(p, n=n_per_active * 6)
        result[smi] = _filter_by_tanimoto(smi, candidates, n_per_active)
    return result
