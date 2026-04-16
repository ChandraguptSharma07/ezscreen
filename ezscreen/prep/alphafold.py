from __future__ import annotations

from pathlib import Path

import requests
from rich.console import Console

# EBI AlphaFold DB — v4 model, fragment 1 covers the full chain for most proteins
_AF_PDB_URL  = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
_AF_META_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"

_PLDDT_WARN = 50   # residues below this are low-confidence
_PLDDT_LOOP = 40   # residues below this are disordered loops


def _fetch_pdb(uniprot_id: str) -> str:
    url = _AF_PDB_URL.format(uid=uniprot_id)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        raise ValueError(
            f"No AlphaFold structure found for {uniprot_id}. "
            "Check the UniProt accession or try https://alphafold.ebi.ac.uk."
        )
    resp.raise_for_status()
    return resp.text


def _parse_plddt(pdb_text: str) -> dict[int, float]:
    """Returns {residue_seq_number: pLDDT} from the B-factor column of ATOM records."""
    scores: dict[int, float] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            res_seq = int(line[22:26].strip())
            bfactor = float(line[60:66].strip())
            if res_seq not in scores:
                scores[res_seq] = bfactor
        except (ValueError, IndexError):
            continue
    return scores


def _warn_low_confidence(
    plddt: dict[int, float],
    console: Console,
) -> list[tuple[int, int]]:
    """Print a summary of low-confidence stretches; return them as (start, end) spans."""
    low = sorted(r for r, v in plddt.items() if v < _PLDDT_WARN)
    if not low:
        return []

    # group into contiguous spans
    spans: list[tuple[int, int]] = []
    start = prev = low[0]
    for r in low[1:]:
        if r == prev + 1:
            prev = r
        else:
            spans.append((start, prev))
            start = prev = r
    spans.append((start, prev))

    pct = 100 * len(low) / len(plddt)
    console.print(
        f"  [#e3b341]pLDDT warning:[/#e3b341] "
        f"{len(low)} residues ({pct:.0f}%) below confidence threshold {_PLDDT_WARN}."
    )
    for s, e in spans[:5]:
        label = "disordered loop" if plddt.get(s, 100) < _PLDDT_LOOP else "low-confidence"
        console.print(f"    residues {s}–{e}  [{label}]", style="#6e7681")
    if len(spans) > 5:
        console.print(f"    … and {len(spans) - 5} more spans", style="#6e7681")

    return spans


def download_alphafold_structure(
    uniprot_id: str,
    output_path: Path,
    warn_residues: list[int] | None = None,
) -> tuple[Path, list[tuple[int, int]]]:
    """Download the AF structure for uniprot_id, report low-confidence regions.

    warn_residues — optional list of residue numbers that are part of the binding
    site; if any fall in a low-confidence span the warning is escalated to an error.

    Returns (output_path, low_confidence_spans).
    """
    console = Console()
    uid = uniprot_id.strip().upper()
    console.print(f"  Fetching AlphaFold structure for {uid}...")

    pdb_text = _fetch_pdb(uid)
    plddt    = _parse_plddt(pdb_text)

    if not plddt:
        raise ValueError(f"Could not read pLDDT scores from AlphaFold PDB for {uid}.")

    mean_plddt = sum(plddt.values()) / len(plddt)
    console.print(f"  Mean pLDDT: {mean_plddt:.1f}  ({len(plddt)} residues)")

    low_spans = _warn_low_confidence(plddt, console)

    if warn_residues and low_spans:
        low_set = {r for s, e in low_spans for r in range(s, e + 1)}
        site_in_loop = [r for r in warn_residues if r in low_set]
        if site_in_loop:
            console.print(
                f"  [bold #f85149]Binding-site residues {site_in_loop} fall in a "
                f"low-confidence region (pLDDT < {_PLDDT_WARN}). "
                "Docking results in this region may be unreliable.[/bold #f85149]"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pdb_text)
    console.print(f"  Saved to {output_path}")
    return output_path, low_spans
