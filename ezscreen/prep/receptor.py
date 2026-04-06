from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import questionary
import requests
from rich.console import Console

from ezscreen.errors import InvalidReceptorError, ReceptorPrepError
from ezscreen.state import BACK

console = Console()
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


# ---------------------------------------------------------------------------
# AlphaFold detection (4-tier)
# ---------------------------------------------------------------------------

def detect_alphafold(pdb_path: Path) -> tuple[bool, str | None]:
    """Returns (is_alphafold, version_hint). version_hint is 'af2', 'af3', or None."""
    if pdb_path.suffix.lower() == ".cif":
        return True, "af3"

    if (pdb_path.parent / "summary_confidences.json").exists():
        return True, "af3"

    try:
        text = pdb_path.read_text(errors="ignore").upper()
    except OSError:
        return False, None

    if "ALPHAFOLD" in text:
        remarks = " ".join(
            ln for ln in text.splitlines()
            if ln.startswith("REMARK") and "ALPHAFOLD" in ln
        )
        if "ALPHAFOLD3" in remarks or ("V3" in remarks and "ALPHAFOLD" in remarks):
            return True, "af3"
        if any(k in remarks for k in ("MONOMER", "MULTIMER", "ALPHAFOLD2", "V2")):
            return True, "af2"
        return True, None

    bfactors: list[float] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                bfactors.append(float(line[60:66]))
            except (ValueError, IndexError):
                pass
    if len(bfactors) > 50 and all(0 <= b <= 100 for b in bfactors):
        if sum(1 for b in bfactors if b > 49) / len(bfactors) >= 0.9:
            return True, None

    return False, None


# ---------------------------------------------------------------------------
# Chain utilities
# ---------------------------------------------------------------------------

def get_chains(pdb_path: Path) -> list[str]:
    chains: list[str] = []
    seen: set[str] = set()
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line.startswith("ATOM  "):
            ch = line[21:22].strip()
            if ch and ch not in seen:
                seen.add(ch)
                chains.append(ch)
    return chains


def _strip_alt_conformations(src: Path, dst: Path) -> int:
    kept, stripped = [], 0
    for line in src.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            if line[16:17] in (" ", "A"):
                kept.append(line[:16] + " " + line[17:])
            else:
                stripped += 1
        else:
            kept.append(line)
    dst.write_text("\n".join(kept) + "\n")
    return stripped


def _filter_chains(src: Path, chains: list[str], dst: Path) -> None:
    chain_set = set(chains)
    keep_records = {"TER", "MODEL", "ENDMDL", "END", "CRYST1", "REMARK", "HEADER", "TITLE", "SEQRES"}
    lines = []
    for line in src.read_text(errors="ignore").splitlines():
        record = line[:6].rstrip()
        if record in ("ATOM", "HETATM"):
            if line[21:22] in chain_set:
                lines.append(line)
        elif record in keep_records:
            lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# RCSB download
# ---------------------------------------------------------------------------

def fetch_pdb(pdb_id: str, output_dir: Path) -> Path:
    pdb_id = pdb_id.strip().upper()
    if not (len(pdb_id) == 4 and pdb_id.isalnum()):
        raise InvalidReceptorError(f"Not a valid PDB ID: '{pdb_id}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    try:
        r = requests.get(RCSB_URL.format(pdb_id=pdb_id), timeout=30)
    except requests.Timeout as exc:
        raise InvalidReceptorError("RCSB request timed out") from exc
    except requests.ConnectionError as exc:
        raise InvalidReceptorError(f"Could not reach RCSB: {exc}") from exc
    if r.status_code == 404:
        raise InvalidReceptorError(f"PDB ID '{pdb_id}' not found on RCSB")
    if not r.ok:
        raise InvalidReceptorError(f"RCSB returned HTTP {r.status_code}")
    out.write_bytes(r.content)
    return out


# ---------------------------------------------------------------------------
# Chain selection prompt
# ---------------------------------------------------------------------------

def prompt_chain_selection(chains: list[str]) -> list[str] | object:
    """Returns selected chain list, or BACK sentinel."""
    if len(chains) == 1:
        return chains

    choice = questionary.select(
        "Chain selection:",
        choices=[f"Auto — use chain {chains[0]}", "Choose chains", "← Back"],
    ).ask()

    if choice is None or choice == "← Back":
        return BACK
    if choice.startswith("Auto"):
        return [chains[0]]

    selected = questionary.checkbox(
        "Select chains (Space to toggle, Enter to confirm):",
        choices=chains,
    ).ask()
    return selected if selected else BACK


# ---------------------------------------------------------------------------
# pdbfixer + Meeko
# ---------------------------------------------------------------------------

def _run_pdbfixer(src: Path, dst: Path, ph: float, keep_waters: bool) -> dict[str, Any]:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise ReceptorPrepError("pdbfixer/openmm not installed") from exc

    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues()
    n_missing = sum(len(v) for v in fixer.missingResidues.values())
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=keep_waters)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    residue_count = fixer.topology.getNumResidues()
    with dst.open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    try:
        from pdbfixer import __version__ as v
    except ImportError:
        v = "unknown"
    return {"missing_residues": n_missing, "residue_count": residue_count, "pdbfixer_version": v}


def _run_meeko_receptor(src: Path, output_dir: Path) -> tuple[Path, str]:
    exe = shutil.which("mk_prepare_receptor")
    if exe is None:
        raise ReceptorPrepError("mk_prepare_receptor not found — pip install meeko")
    pdbqt = output_dir / (src.stem + ".pdbqt")
    result = subprocess.run([exe, "-i", str(src), "-o", str(pdbqt)], capture_output=True, text=True)
    if result.returncode != 0:
        raise ReceptorPrepError(f"mk_prepare_receptor failed:\n{result.stderr.strip()}")
    try:
        import meeko
        mv = meeko.__version__
    except (ImportError, AttributeError):
        mv = "unknown"
    return pdbqt, mv


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prep_receptor(
    pdb_path: Path,
    chains: list[str],
    output_dir: Path,
    ph: float = 7.4,
    keep_waters: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[dict] = []

    noalt = output_dir / "receptor_noalt.pdb"
    n_alt = _strip_alt_conformations(pdb_path, noalt)
    if n_alt:
        warnings.append({
            "severity": "low", "category": "alternates_resolved",
            "affected_count": n_alt,
            "message": f"{n_alt} alternate conformation atoms removed (kept A)",
            "action": "none required",
        })

    chain_filtered = output_dir / "receptor_chains.pdb"
    _filter_chains(noalt, chains, chain_filtered)

    fixed = output_dir / "receptor_fixed.pdb"
    try:
        fx = _run_pdbfixer(chain_filtered, fixed, ph=ph, keep_waters=keep_waters)
    except ReceptorPrepError:
        raise
    except Exception as exc:
        raise ReceptorPrepError(f"pdbfixer error: {exc}") from exc

    if fx["missing_residues"]:
        warnings.append({
            "severity": "medium", "category": "missing_residues_modelled",
            "affected_count": fx["missing_residues"],
            "message": f"{fx['missing_residues']} missing residues modelled by pdbfixer",
            "action": "verify modelled regions are not in/near binding site",
        })
    warnings.append({
        "severity": "low", "category": "histidine_protonation", "affected_count": 0,
        "message": f"Histidine states assigned at pH {ph} — guesses only",
        "action": "verify HIE/HID if histidines are in binding site",
    })

    try:
        pdbqt, mv = _run_meeko_receptor(fixed, output_dir)
    except ReceptorPrepError:
        raise
    except Exception as exc:
        raise ReceptorPrepError(f"Meeko error: {exc}") from exc

    return {
        "pdbqt_path": pdbqt,
        "fixed_pdb_path": fixed,
        "report": {
            "chains_selected": chains,
            "residue_count": fx["residue_count"],
            "missing_residues": fx["missing_residues"],
            "alternates_resolved": n_alt,
            "tools": {"pdbfixer": fx["pdbfixer_version"], "meeko": mv},
            "warnings": warnings,
        },
    }
