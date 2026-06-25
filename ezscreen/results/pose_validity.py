from __future__ import annotations

from pathlib import Path


def check_poses(poses_sdf: Path, receptor_pdb: Path) -> dict[str, dict] | None:
    """Run PoseBusters validity checks on every docked pose.

    Returns {lig_id: {"passed": bool, "failed_checks": [str, ...]}} or None when
    posebusters isn't installed or the inputs are missing. failed_checks holds the
    names of any PoseBusters checks that came back False for that pose.
    """
    try:
        import pandas as pd
        from posebusters import PoseBusters
        from rdkit import Chem
    except ImportError:
        return None

    if not poses_sdf.exists() or not receptor_pdb.exists():
        return None

    supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False, sanitize=True)
    buster = PoseBusters(config="dock")

    results: dict[str, dict] = {}
    for i, mol in enumerate(supplier):
        if mol is None:
            results[f"_unparsed_{i}"] = {"passed": False, "failed_checks": ["parse_error"]}
            continue

        if mol.HasProp("lig_id"):
            lig_id = mol.GetProp("lig_id")
        elif mol.HasProp("_Name") and mol.GetProp("_Name"):
            lig_id = mol.GetProp("_Name")
        else:
            lig_id = f"pose_{i}"

        try:
            df = buster.bust(mol, None, str(receptor_pdb))
            row = df.iloc[0]
            failed = [str(col) for col, val in row.items() if pd.notna(val) and not bool(val)]
            results[lig_id] = {"passed": not failed, "failed_checks": failed}
        except Exception:
            results[lig_id] = {"passed": False, "failed_checks": ["parse_error"]}

    return results or None
