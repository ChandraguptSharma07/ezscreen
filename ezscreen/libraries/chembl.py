from __future__ import annotations

from pathlib import Path

import requests

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
_PAGE_SIZE = 1000


def _uniprot_to_chembl_target(uniprot_id: str) -> str | None:
    url = f"{_CHEMBL_API}/target.json"
    params = {"target_components__accession": uniprot_id, "limit": 1}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        targets = resp.json().get("targets", [])
        if not targets:
            return None
        return targets[0]["target_chembl_id"]
    except Exception:
        return None


def _fetch_activities_page(
    session: requests.Session,
    target_chembl_id: str,
    ic50_um: float,
    offset: int,
) -> list[dict]:
    url = f"{_CHEMBL_API}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": "IC50",
        "standard_relation__in": "=,<",
        "standard_units": "nM",
        "standard_value__lte": ic50_um * 1000,  # µM → nM
        "assay_type": "B",  # binding assays only
        "limit": _PAGE_SIZE,
        "offset": offset,
    }
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("activities", [])
    except Exception:
        return []


def _fetch_smiles(session: requests.Session, chembl_id: str) -> str | None:
    url = f"{_CHEMBL_API}/molecule/{chembl_id}.json"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        structs = resp.json().get("molecule_structures") or {}
        return structs.get("canonical_smiles")
    except Exception:
        return None


def fetch_chembl_actives(
    uniprot_id: str,
    output_path: Path,
    ic50_um: float = 1.0,
    max_compounds: int | None = None,
) -> int:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    console.print(f"  Resolving UniProt {uniprot_id} to ChEMBL target...")
    target_id = _uniprot_to_chembl_target(uniprot_id)
    if target_id is None:
        console.print(f"  [red]No ChEMBL target found for {uniprot_id}[/red]")
        return 0

    console.print(f"  Target: {target_id}  (IC50 <= {ic50_um} µM, binding assays)")

    seen_smiles: set[str] = set()
    rows: list[tuple[str, str]] = []  # (smiles, chembl_id)

    with requests.Session() as session:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Fetching ChEMBL actives for {target_id}...", total=None
            )
            offset = 0
            while True:
                batch = _fetch_activities_page(session, target_id, ic50_um, offset)
                if not batch:
                    break
                for act in batch:
                    mol_id = act.get("molecule_chembl_id")
                    if not mol_id:
                        continue
                    smi = _fetch_smiles(session, mol_id)
                    if smi and smi not in seen_smiles:
                        seen_smiles.add(smi)
                        rows.append((smi, mol_id))
                    if max_compounds and len(rows) >= max_compounds:
                        break
                progress.update(
                    task,
                    description=f"Fetching ChEMBL actives for {target_id}... {len(rows)} so far",
                )
                if max_compounds and len(rows) >= max_compounds:
                    break
                if len(batch) < _PAGE_SIZE:
                    break
                offset += _PAGE_SIZE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{smi} {cid}" for smi, cid in rows]
    output_path.write_text("\n".join(lines) + "\n")
    console.print(f"  Wrote {len(rows)} actives to {output_path}")
    return len(rows)
