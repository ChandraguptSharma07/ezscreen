from __future__ import annotations

import base64
import io
import json
import math
import shutil
import subprocess
import sys
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
            ln
            for ln in text.splitlines()
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
    has_atoms = False
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line.startswith("ATOM  "):
            has_atoms = True
            ch = line[21:22].strip()
            if ch and ch not in seen:
                seen.add(ch)
                chains.append(ch)
    # PDB with no chain IDs (blank column) — treat entire file as one chain
    if not chains and has_atoms:
        chains = [" "]
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
    keep_records = {
        "MODEL",
        "ENDMDL",
        "END",
        "CRYST1",
        "REMARK",
        "HEADER",
        "TITLE",
    }  # noqa: E501
    lines = []
    retained_segment = False
    for line in src.read_text(errors="ignore").splitlines():
        record = line[:6].rstrip()
        if record in ("ATOM", "HETATM"):
            retained_segment = line[21:22] in chain_set
            if retained_segment:
                lines.append(line)
        elif record == "TER":
            if retained_segment:
                lines.append(line)
            retained_segment = False
        elif record == "SEQRES":
            if line[11:12] in chain_set:
                lines.append(line)
        elif record in keep_records:
            lines.append(line)
            if record in {"MODEL", "ENDMDL", "END"}:
                retained_segment = False
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
# Heteroatom inventory + selection
# ---------------------------------------------------------------------------

_WATERS: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "D2O"})

_METALS: frozenset[str] = frozenset(
    {
        "ZN",
        "MG",
        "MN",
        "FE",
        "FE2",
        "CA",
        "NA",
        "K",
        "CU",
        "CU1",
        "CO",
        "NI",
        "CD",
        "HG",
        "MO",
        "W",
    }
)

# Organic prosthetic groups / cofactors worth keeping in the receptor by default.
_COFACTORS: frozenset[str] = frozenset(
    {
        "HEM",
        "HEC",
        "HEA",
        "HEB",
        "SRM",
        "VER",
        "NAD",
        "NAP",
        "NAI",
        "NDP",
        "NAJ",
        "FAD",
        "FMN",
        "FDA",
        "FADH",
        "ATP",
        "ADP",
        "AMP",
        "ANP",
        "AGS",
        "GTP",
        "GDP",
        "GNP",
        "GSP",
        "SAM",
        "SAH",
        "PLP",
        "PMP",
        "TPP",
        "TDP",
        "COA",
        "ACO",
        "BTN",
        "CLA",
        "BCL",
        "B12",
        "COB",
        "PQQ",
        "MGD",
        "F43",
        "H4B",
        "BH4",
    }
)

# Crystallographic additives that are never biologically relevant.
_ADDITIVE_HET: frozenset[str] = frozenset(
    {
        "GOL",
        "EDO",
        "MPD",
        "PGO",
        "PG4",
        "1PE",
        "PE8",
        "PE3",
        "PE4",
        "PE5",
        "PE6",
        "SO4",
        "PO4",
        "NO3",
        "CL",
        "BR",
        "IOD",
        "ACT",
        "ACY",
        "ACE",
        "DMS",
        "DMF",
        "EOH",
        "IPA",
        "TRS",
        "HEP",
        "MES",
        "MOH",
        "TAR",
        "CIT",
        "FMT",
        "IMD",
        "DIO",
        "TLA",
        "BOG",
        "OES",
        "FLC",
        "BME",
        "PEG",
    }
)


def _classify_hetero(resname: str, n_atoms: int) -> tuple[str, bool]:
    """Return (category, keep_by_default) for a HETATM residue."""
    if resname in _WATERS:
        return "water", False
    if resname in _COFACTORS:
        return "cofactor", True
    if resname in _METALS:
        return "metal", False
    if resname in _ADDITIVE_HET:
        return "additive", False
    if n_atoms >= 5:
        return "ligand", False  # a co-crystal ligand — usually the thing you displace
    return "other", False


def heteroatom_inventory(pdb_path: Path) -> list[dict[str, Any]]:
    """List HETATM residues grouped by (resname, chain, resseq) with a keep suggestion."""
    groups: dict[tuple, int] = {}
    order: list[tuple] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith("HETATM"):
            continue
        try:
            key = (line[17:20].strip(), line[21:22].strip(), int(line[22:26]))
        except (ValueError, IndexError):
            continue
        if key not in groups:
            groups[key] = 0
            order.append(key)
        groups[key] += 1

    inventory = []
    for resname, chain, resseq in order:
        n = groups[(resname, chain, resseq)]
        category, keep = _classify_hetero(resname, n)
        inventory.append(
            {
                "resname": resname,
                "chain": chain,
                "resseq": resseq,
                "n_atoms": n,
                "category": category,
                "keep_default": keep,
            }
        )
    return inventory


def _apply_hetero_selection(src: Path, dst: Path, keep_keys: set[tuple]) -> int:
    """Write src → dst keeping every ATOM record but only the selected HETATM residues."""
    removed = 0
    lines = []
    for line in src.read_text(errors="ignore").splitlines():
        if line.startswith("HETATM"):
            try:
                key = (line[17:20].strip(), line[21:22].strip(), int(line[22:26]))
            except (ValueError, IndexError):
                key = None
            if key not in keep_keys:
                removed += 1
                continue
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")
    return removed


def _append_selected_heterogens(
    protein_pdb: Path, selected_pdb: Path, dst: Path,
    protected_keys: set[tuple] | None = None,
) -> None:
    """Append selected HETATM records after PDBFixer without losing their IDs."""
    protein_lines = [
        line
        for line in protein_pdb.read_text(errors="ignore").splitlines()
        if not line.startswith("END")
        and (not line.startswith("HETATM") or _hetero_key_from_line(line) in (protected_keys or set()))
    ]
    hetero_lines = [
        line
        for line in selected_pdb.read_text(errors="ignore").splitlines()
        if line.startswith("HETATM") and _hetero_key_from_line(line) not in (protected_keys or set())
    ]
    dst.write_text("\n".join([*protein_lines, *hetero_lines, "END"]) + "\n")


def _hetero_key_from_line(line: str) -> tuple[str, str, int] | None:
    # Meeko serializes retained waters and other rigid receptor residues as ATOM.
    if not line.startswith(("ATOM  ", "HETATM")):
        return None
    try:
        return (line[17:20].strip(), line[21:22].strip(), int(line[22:26]))
    except (IndexError, ValueError):
        return None


def _hetero_retention_report(
    pdbqt_path: Path, requested: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Report whether explicitly selected heterogen groups survived PDBQT prep."""
    requested_entries = [
        {
            "resname": str(item["resname"]).strip().upper(),
            "chain": str(item.get("chain", "")).strip(),
            "resseq": int(item["resseq"]),
        }
        for item in (requested or [])
    ]
    present = {
        key
        for line in pdbqt_path.read_text(errors="ignore").splitlines()
        if (key := _hetero_key_from_line(line)) is not None
    }
    retained = [
        item
        for item in requested_entries
        if (item["resname"], item["chain"], item["resseq"]) in present
    ]
    missing = [item for item in requested_entries if item not in retained]
    return {
        "requested": requested_entries,
        "retained": retained,
        "missing": missing,
        "complete": not missing,
    }


_RESIDUE_STATE_FAMILIES: dict[str, frozenset[str]] = {
    "ASP": frozenset({"ASP", "ASH"}),
    "ASH": frozenset({"ASP", "ASH"}),
    "GLU": frozenset({"GLU", "GLH"}),
    "GLH": frozenset({"GLU", "GLH"}),
    "HIS": frozenset({"HIS", "HID", "HIE", "HIP"}),
    "HID": frozenset({"HIS", "HID", "HIE", "HIP"}),
    "HIE": frozenset({"HIS", "HID", "HIE", "HIP"}),
    "HIP": frozenset({"HIS", "HID", "HIE", "HIP"}),
    "LYS": frozenset({"LYS", "LYN"}),
    "LYN": frozenset({"LYS", "LYN"}),
    "CYS": frozenset({"CYS", "CYM"}),
    "CYM": frozenset({"CYS", "CYM"}),
    "TYR": frozenset({"TYR", "TYM"}),
    "TYM": frozenset({"TYR", "TYM"}),
}


def _apply_residue_state_overrides(
    src: Path, dst: Path, overrides: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply exact user-selected titratable residue names, failing on ambiguity."""
    normalized: dict[tuple[str, int, str], str] = {}
    for item in overrides:
        try:
            key = (
                str(item.get("chain", "")).strip(),
                int(item["resseq"]),
                str(item.get("icode", "")).strip(),
            )
            state = str(item["state"]).strip().upper()
        except (KeyError, TypeError, ValueError) as exc:
            raise ReceptorPrepError(f"Invalid residue-state override: {item!r}") from exc
        previous = normalized.get(key)
        if previous is not None and previous != state:
            raise ReceptorPrepError(
                f"Conflicting residue-state overrides for {key[0]}:{key[1]}{key[2]}"
            )
        normalized[key] = state

    found: dict[tuple[str, int, str], str] = {}
    output: list[str] = []
    for line in src.read_text(errors="ignore").splitlines():
        if line.startswith("ATOM  "):
            try:
                key = (
                    line[21:22].strip(),
                    int(line[22:26]),
                    line[26:27].strip(),
                )
            except (IndexError, ValueError):
                output.append(line)
                continue
            state = normalized.get(key)
            if state is not None:
                current = line[17:20].strip().upper()
                if state not in _RESIDUE_STATE_FAMILIES.get(current, frozenset()):
                    raise ReceptorPrepError(
                        f"Residue-state override {state} is incompatible with "
                        f"{key[0]}:{current}{key[1]}{key[2]}"
                    )
                line = line[:17] + f"{state:>3}" + line[20:]
                found[key] = state
        output.append(line)

    missing = [key for key in normalized if key not in found]
    if missing:
        rendered = ", ".join(f"{chain}:{resseq}{icode}" for chain, resseq, icode in missing)
        raise ReceptorPrepError(f"Residue-state override target(s) not found: {rendered}")

    dst.write_text("\n".join(output) + "\n")
    return {
        "assignment_count": len(found),
        "assignments": [
            {
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "state": found[(chain, resseq, icode)],
            }
            for chain, resseq, icode in sorted(found)
        ],
    }


# ---------------------------------------------------------------------------
# pdbfixer + Meeko
# ---------------------------------------------------------------------------


def _run_pdbfixer(
    src: Path, dst: Path, keep_waters: bool, remove_heterogens: bool = True
) -> dict[str, Any]:
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ReceptorPrepError("pdbfixer/openmm not installed") from exc

    fixer = PDBFixer(filename=str(src))
    fixer.findMissingResidues()
    n_missing = sum(len(v) for v in fixer.missingResidues.values())
    fixer.missingResidues = {}

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    # When the user has chosen specific heterogens to keep, the de-selected ones
    # are already stripped upstream — don't let pdbfixer remove the kept ones.
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_waters)
    # Do NOT call addMissingAtoms or addMissingHydrogens — PDBFixer adds
    # N/C-terminal H and OXT atoms that cause ProDy to infer valence-4 O
    # or valence-5 C, crashing Meeko's RDKit sanitization. Meeko assigns
    # its own H and atom types from residue templates.

    residue_count = fixer.topology.getNumResidues()
    with dst.open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    try:
        from pdbfixer import __version__ as v
    except ImportError:
        v = "unknown"
    return {
        "missing_residues": n_missing,
        "residue_count": residue_count,
        "pdbfixer_version": v,
    }


def _infer_element(atom_name: str, resname: str = "") -> str:
    """Guess element symbol from PDB atom name."""
    name = atom_name.strip().lstrip("0123456789")
    if not name:
        return ""
    if name.upper() == "CA" and resname.strip().upper() != "CA":
        return "C"
    two = name[:2].upper()
    if two in {"FE", "ZN", "MG", "MN", "CA", "NA", "CL", "BR", "CU", "CO", "NI"}:
        return two
    return name[0].upper()


def _ensure_element_column(lines: list[str]) -> list[str]:
    """Pad ATOM/HETATM lines to 80 chars and fill blank element column (76-78)."""
    out = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            line = line.rstrip("\n").ljust(80)
            if not line[76:78].strip():
                elem = _infer_element(line[12:16], line[17:20])
                line = line[:76] + elem.rjust(2) + line[78:]
        out.append(line)
    return out


def _strip_conect(src: Path, dst: Path) -> None:
    """Remove CONECT/LINK/SSBOND records and explicit hydrogen atoms.

    PDBFixer adds N-terminal H2/H3 via addMissingAtoms even without
    addMissingHydrogens. ProDy then infers extra bonds by distance and
    pushes CA valence to 5, crashing RDKit. Meeko rebuilds all Hs from
    its own residue templates, so stripping them here is safe.
    """
    skip_records = {"CONECT", "LINK  ", "SSBOND"}
    lines = []
    for line in src.read_text(errors="ignore").splitlines():
        if line[:6] in skip_records:
            continue
        if line.startswith(("ATOM  ", "HETATM")):
            element = line[76:78].strip() if len(line) >= 78 else ""
            atom_name = line[12:16].strip()
            if element == "H" or (not element and atom_name.startswith("H")):
                continue
            if atom_name in {"OXT", "OT1", "OT2"}:
                continue
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def _run_meeko_receptor(src: Path, output_dir: Path) -> tuple[Path, str]:
    exe = _meeko_receptor_command()
    if exe is None:
        raise ReceptorPrepError(
            "mk_prepare_receptor(.py) not found — pip install meeko"
        )

    # Strip CONECT/LINK/SSBOND records before Meeko sees the file — explicit
    # bond records from DUD-E/PDBFixer can create carbon with 5 bonds which
    # RDKit's sanitizer rejects with AtomValenceException.
    clean = output_dir / (src.stem + "_clean.pdb")
    _strip_conect(src, clean)

    pdbqt = output_dir / (src.stem + ".pdbqt")

    # OpenBLAS (used by meeko internally) tries to spawn many threads and can
    # exhaust memory on Windows if thread count is not capped. Limit to 1 thread
    # via env vars before launching the subprocess.
    import os

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    console.print("  [dim]meeko: converting receptor to PDBQT...[/dim]")
    try:
        result = subprocess.run(
            [exe, "--read_pdb", str(clean), "-p", str(pdbqt), "-a"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReceptorPrepError("mk_prepare_receptor timed out after 180 s") from exc

    if result.returncode != 0:
        stderr = (
            result.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(result.stderr, bytes)
            else result.stderr.strip()
        )
        meeko_err = stderr
        console.print("  [yellow]meeko failed — trying obabel fallback...[/yellow]")
        try:
            return _run_obabel_receptor(clean, output_dir)
        except ReceptorPrepError as ob_exc:
            raise ReceptorPrepError(
                f"mk_prepare_receptor failed:\n{meeko_err}"
            ) from ob_exc
    try:
        import meeko

        mv = meeko.__version__
    except (ImportError, AttributeError):
        mv = "unknown"
    return pdbqt, mv


def _meeko_receptor_command() -> str | None:
    """Return either supported Meeko receptor-preparation entry point."""
    for name in ("mk_prepare_receptor", "mk_prepare_receptor.py"):
        exe = shutil.which(name)
        if exe:
            return exe
        environment_exe = Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin") / name
        if environment_exe.is_file():
            return str(environment_exe)
    return None


def _run_obabel_receptor(src: Path, output_dir: Path) -> tuple[Path, str]:
    exe = shutil.which("obabel")
    if exe is None:
        raise ReceptorPrepError(
            "obabel not found — install Open Babel: https://openbabel.org"
        )
    pdbqt = output_dir / (src.stem + ".pdbqt")
    console.print("  [dim]obabel: converting receptor to PDBQT...[/dim]")
    result = subprocess.run(
        [exe, str(src), "-xr", "-O", str(pdbqt)],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=180,
    )
    if result.returncode != 0 or not pdbqt.exists() or pdbqt.stat().st_size == 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReceptorPrepError(f"obabel failed:\n{stderr}")
    return pdbqt, "obabel"


_TITRATABLE_RESIDUES: frozenset[str] = frozenset(
    {
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "ASP",
        "ASH",
        "GLU",
        "GLH",
        "LYS",
        "LYN",
        "CYS",
        "CYM",
        "TYR",
        "TYM",
    }
)


def _pdb2pqr_command() -> str | None:
    for name in ("pdb2pqr", "pdb2pqr30"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _pqr_atom_to_pdb_line(line: str, serial: int) -> str | None:
    parts = line.split()
    if len(parts) < 10 or parts[0] not in {"ATOM", "HETATM"}:
        return None
    record = parts[0]
    atom = parts[2]
    resname = parts[3]
    chain = ""
    idx = 4
    if len(parts[idx]) == 1 and not parts[idx].lstrip("-").isdigit():
        chain = parts[idx]
        idx += 1
    try:
        resseq = int(float(parts[idx]))
        x = float(parts[idx + 1])
        y = float(parts[idx + 2])
        z = float(parts[idx + 3])
    except (IndexError, ValueError):
        return None
    element = _infer_element(atom, resname)
    return (
        f"{record:<6}{serial:>5} {atom:<4} {resname:>3} {chain:1}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}          {element:>2}"
    )


def _pqr_to_pdb(pqr_path: Path, pdb_path: Path) -> dict[str, Any]:
    lines: list[str] = []
    assignments: dict[tuple[str, int, str], str] = {}
    serial = 1
    for line in pqr_path.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            pdb_line = _pqr_atom_to_pdb_line(line, serial)
            if pdb_line is None:
                continue
            serial += 1
            lines.append(pdb_line)
            resname = pdb_line[17:20].strip()
            if resname in _TITRATABLE_RESIDUES:
                key = (
                    pdb_line[21:22].strip(),
                    int(pdb_line[22:26]),
                    pdb_line[26:27].strip(),
                )
                assignments.setdefault(key, resname)
        elif line.startswith(("TER", "END")):
            lines.append(line)
    if not any(ln.startswith("ATOM") for ln in lines):
        raise ReceptorPrepError("PDB2PQR produced no protein atoms")
    if not lines or not lines[-1].startswith("END"):
        lines.append("END")
    pdb_path.write_text("\n".join(lines) + "\n")
    return {
        "assignment_count": len(assignments),
        "assignments": [
            {"chain": chain, "resseq": resseq, "icode": icode, "state": state}
            for (chain, resseq, icode), state in sorted(assignments.items())
        ],
    }


def _pqr_residue_assignments(pqr_path: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    fallback_by_resseq: dict[int, str] = {}
    for line in pqr_path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        pdb_line = _pqr_atom_to_pdb_line(line, 1)
        if pdb_line is None:
            continue
        resname = pdb_line[17:20].strip()
        if resname not in _TITRATABLE_RESIDUES:
            continue
        try:
            resseq = int(pdb_line[22:26])
        except ValueError:
            continue
        chain = pdb_line[21:22].strip()
        icode = pdb_line[26:27].strip()
        entry = grouped.setdefault(
            (chain, resseq, icode), {"resname": resname, "atom_names": set()}
        )
        entry["atom_names"].add(pdb_line[12:16].strip().upper())

    assignments: dict[tuple[str, int, str], str] = {}
    for key, entry in grouped.items():
        resname = entry["resname"]
        atom_names = entry["atom_names"]
        state = resname
        if resname == "HIS":
            has_hd1 = "HD1" in atom_names
            has_he2 = "HE2" in atom_names
            if has_hd1 and has_he2:
                state = "HIP"
            elif has_hd1:
                state = "HID"
            elif has_he2:
                state = "HIE"
        assignments[key] = state
        fallback_by_resseq.setdefault(key[1], state)
    return {"assignments": assignments, "fallback_by_resseq": fallback_by_resseq}


def _apply_pqr_residue_names_to_pdb(
    src: Path, pqr_path: Path, dst: Path
) -> dict[str, Any]:
    parsed = _pqr_residue_assignments(pqr_path)
    assignments: dict[tuple[str, int, str], str] = parsed["assignments"]
    fallback_by_resseq: dict[int, str] = parsed["fallback_by_resseq"]
    lines: list[str] = []
    applied: dict[tuple[str, int, str], str] = {}
    for line in src.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                icode = line[26:27].strip()
            except (IndexError, ValueError):
                lines.append(line)
                continue
            new_resname = assignments.get((chain, resseq, icode))
            if new_resname is None and not any(k[0] for k in assignments):
                new_resname = fallback_by_resseq.get(resseq)
            if new_resname:
                line = line[:17] + f"{new_resname:>3}" + line[20:]
                applied[(chain, resseq, icode)] = new_resname
        lines.append(line)
    if not any(ln.startswith("ATOM") for ln in lines):
        return _pqr_to_pdb(pqr_path, dst)
    lines = _ensure_element_column(lines)
    dst.write_text("\n".join(lines) + "\n")
    return {
        "assignment_count": len(applied),
        "assignments": [
            {"chain": chain, "resseq": resseq, "icode": icode, "state": state}
            for (chain, resseq, icode), state in sorted(applied.items())
        ],
    }


def _run_pdb2pqr_protonation(src: Path, dst: Path, ph: float) -> dict[str, Any]:
    exe = _pdb2pqr_command()
    if exe is None:
        return {
            "applied": False,
            "tool": None,
            "assignment_count": 0,
            "assignments": [],
            "error": "pdb2pqr executable not found",
        }

    pqr = dst.with_suffix(".pqr")
    log_path = pqr.with_suffix(".log")
    for stale in (dst, pqr, log_path):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    cmd = [
        exe,
        "--ff=AMBER",
        f"--with-ph={ph}",
        "--titration-state-method=propka",
        "--keep-chain",
        str(src),
        str(pqr),
    ]
    console.print(
        f"  [dim]pdb2pqr/propka: assigning receptor protonation at pH {ph}...[/dim]"
    )

    def _trim_error_text(text: str, limit: int = 4000) -> str:
        text = text.strip()
        if not text:
            return ""
        marker = "Traceback (most recent call last):"
        if marker in text:
            text = text[text.rfind(marker) :]
        if len(text) <= limit:
            return text
        text = text[-limit:]
        if "\n" in text:
            text = text.split("\n", 1)[1]
        return text.strip()

    def _error_detail(prefix: str = "") -> str:
        parts = [_trim_error_text(prefix)] if prefix.strip() else []
        if log_path.exists():
            log_tail = _trim_error_text(
                log_path.read_text(errors="replace"), limit=2000
            )
            if log_tail:
                parts.append(log_tail)
        return "\n".join(p for p in parts if p).strip()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "applied": False,
            "tool": "pdb2pqr",
            "assignment_count": 0,
            "assignments": [],
            "error": _error_detail(f"pdb2pqr timed out after {exc.timeout} s"),
        }
    except OSError as exc:
        return {
            "applied": False,
            "tool": "pdb2pqr",
            "assignment_count": 0,
            "assignments": [],
            "error": str(exc),
        }

    if result.returncode != 0 or not pqr.exists() or pqr.stat().st_size == 0:
        detail = _trim_error_text(result.stderr or result.stdout or "")
        if not detail and result.returncode == 0:
            detail = "pdb2pqr exited successfully but produced no PQR output"
        else:
            detail = detail or f"pdb2pqr exited with code {result.returncode}"
        return {
            "applied": False,
            "tool": "pdb2pqr",
            "assignment_count": 0,
            "assignments": [],
            "error": _error_detail(detail),
        }

    try:
        converted = _apply_pqr_residue_names_to_pdb(src, pqr, dst)
    except ReceptorPrepError as exc:
        return {
            "applied": False,
            "tool": "pdb2pqr",
            "assignment_count": 0,
            "assignments": [],
            "error": str(exc),
        }
    converted.update({"applied": True, "tool": "pdb2pqr", "error": None})
    return converted


# ---------------------------------------------------------------------------
# Receptor QC
# ---------------------------------------------------------------------------

_STANDARD_AA: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

_COVALENT_RADII: dict[str, float] = {
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
}

_MAX_VALENCE: dict[str, int] = {
    "C": 4,
    "N": 4,
    "O": 2,
    "S": 6,
    "P": 5,
}

_METAL_ELEMENTS: frozenset[str] = frozenset(
    {"ZN", "MG", "MN", "FE", "CA", "NA", "K", "CU", "CO", "NI", "CD", "HG", "MO", "W"}
)
_METAL_DONOR_ELEMENTS: frozenset[str] = frozenset({"N", "O", "S"})
_METAL_DONOR_DISTANCE_ANGSTROM = 3.0


def _pdb_float(line: str, start: int, end: int) -> float:
    return float(line[start:end])


def _atom_key(line: str) -> tuple[str, int, str, str]:
    return (
        line[21:22].strip(),
        int(line[22:26]),
        line[26:27].strip(),
        line[17:20].strip(),
    )


def _iter_pdb_atoms(pdb_path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            element = line[76:78].strip() if len(line) >= 78 else ""
            atoms.append(
                {
                    "record": line[:6].strip(),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21:22].strip(),
                    "resseq": int(line[22:26]),
                    "icode": line[26:27].strip(),
                    "element": (element or _infer_element(line[12:16])).upper(),
                    "xyz": (
                        _pdb_float(line, 30, 38),
                        _pdb_float(line, 38, 46),
                        _pdb_float(line, 46, 54),
                    ),
                }
            )
        except (IndexError, ValueError):
            continue
    return atoms


def _sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(a, a))
    if length == 0:
        raise ValueError("zero-length vector in dihedral calculation")
    return (a[0] / length, a[1] / length, a[2] / length)


def _project_out(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    scale = _dot(a, b)
    return (a[0] - scale * b[0], a[1] - scale * b[1], a[2] - scale * b[2])


def _dihedral_degrees(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    b0 = _sub(p0, p1)
    b1 = _norm(_sub(p2, p1))
    b2 = _sub(p3, p2)
    v = _project_out(b0, b1)
    w = _project_out(b2, b1)
    x = _dot(v, w)
    y = _dot(_cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def ramachandran_angles(pdb_path: Path) -> list[dict[str, Any]]:
    """Compute phi/psi backbone dihedrals for residues with complete neighbours."""
    residues: dict[tuple[str, int, str, str], dict[str, tuple[float, float, float]]] = (
        {}
    )
    order: list[tuple[str, int, str, str]] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in {"N", "CA", "C"}:
            continue
        try:
            key = _atom_key(line)
            xyz = (
                _pdb_float(line, 30, 38),
                _pdb_float(line, 38, 46),
                _pdb_float(line, 46, 54),
            )
        except (IndexError, ValueError):
            continue
        if key not in residues:
            residues[key] = {}
            order.append(key)
        residues[key][atom_name] = xyz

    angles: list[dict[str, Any]] = []
    for i, key in enumerate(order):
        if i == 0 or i == len(order) - 1:
            continue
        prev_atoms = residues[order[i - 1]]
        atoms = residues[key]
        next_atoms = residues[order[i + 1]]
        required = (
            "C" in prev_atoms and {"N", "CA", "C"}.issubset(atoms) and "N" in next_atoms
        )
        if not required:
            continue
        try:
            phi = _dihedral_degrees(
                prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"]
            )
            psi = _dihedral_degrees(
                atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"]
            )
        except ValueError:
            continue
        chain, resseq, icode, resname = key
        angles.append(
            {
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "resname": resname,
                "phi": round(phi, 2),
                "psi": round(psi, 2),
            }
        )
    return angles


def _render_ramachandran_plot(angles: list[dict[str, Any]]) -> str | None:
    if not angles:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.scatter([a["phi"] for a in angles], [a["psi"] for a in angles], s=10, alpha=0.65)
    ax.axhline(0, color="#d0d7de", linewidth=0.7)
    ax.axvline(0, color="#d0d7de", linewidth=0.7)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel("psi (degrees)")
    ax.set_title("Ramachandran plot")
    ax.grid(True, color="#eaeef2", linewidth=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def metal_site_qc(pdb_path: Path) -> dict[str, Any]:
    """Report retained receptor metals and nearby protein donor atoms."""
    atoms = _iter_pdb_atoms(pdb_path)
    metals = [atom for atom in atoms if atom["element"] in _METAL_ELEMENTS]
    metal_rows: list[dict[str, Any]] = []
    for metal in metals:
        donors = []
        for atom in atoms:
            if (
                atom["record"] != "ATOM"
                or atom["resname"] not in _STANDARD_AA
                or atom["element"] not in _METAL_DONOR_ELEMENTS
            ):
                continue
            distance = _distance(metal["xyz"], atom["xyz"])
            if distance <= _METAL_DONOR_DISTANCE_ANGSTROM:
                donors.append(
                    {
                        "chain": atom["chain"],
                        "resseq": atom["resseq"],
                        "icode": atom["icode"],
                        "resname": atom["resname"],
                        "atom": atom["name"],
                        "element": atom["element"],
                        "distance_angstrom": round(distance, 3),
                    }
                )
        metal_rows.append(
            {
                "record": metal["record"],
                "chain": metal["chain"],
                "resseq": metal["resseq"],
                "icode": metal["icode"],
                "resname": metal["resname"],
                "atom": metal["name"],
                "element": metal["element"],
                "nearby_protein_donors": sorted(
                    donors,
                    key=lambda donor: (
                        donor["distance_angstrom"],
                        donor["chain"],
                        donor["resseq"],
                        donor["atom"],
                    ),
                ),
            }
        )

    elements = sorted({metal["element"] for metal in metal_rows})
    if not elements:
        status = "not_applicable"
    elif elements == ["ZN"]:
        status = "zinc_specialized_available"
    else:
        status = "element_specific_validation_needed"
    return {
        "metal_count": len(metal_rows),
        "metal_elements": elements,
        "metals": metal_rows,
        "donor_cutoff_angstrom": _METAL_DONOR_DISTANCE_ANGSTROM,
        "score_validation_status": status,
    }


def _prep_protocol_audit(source_pdb: Path, prepared_pdb: Path) -> dict[str, Any]:
    source_metal_qc = metal_site_qc(source_pdb)
    prepared_metal_qc = metal_site_qc(prepared_pdb)
    source_elements = set(source_metal_qc["metal_elements"])
    prepared_elements = set(prepared_metal_qc["metal_elements"])
    removed_elements = sorted(source_elements - prepared_elements)
    suggestions: list[str] = []
    if removed_elements:
        suggestions.append("metal_keep_remove")
    if "ZN" in prepared_elements:
        suggestions.extend(["zn_coordination_qc", "ad4zn_scoring"])
    if prepared_elements - {"ZN"}:
        suggestions.append("custom_metal_scoring_validation")
    return {
        "risk_level": "high" if source_elements or prepared_elements else "low",
        "detected": {
            "source_metal_elements": sorted(source_elements),
            "retained_metal_elements": sorted(prepared_elements),
            "removed_metal_elements": removed_elements,
        },
        "benchmark_ablation_suggestions": suggestions,
    }


def receptor_sanity_checks(pdb_path: Path) -> dict[str, Any]:
    atoms = _iter_pdb_atoms(pdb_path)
    residues: dict[tuple[str, int, str, str], set[str]] = {}
    for atom in atoms:
        if atom["record"] != "ATOM" or atom["resname"] not in _STANDARD_AA:
            continue
        key = (atom["chain"], atom["resseq"], atom["icode"], atom["resname"])
        residues.setdefault(key, set()).add(atom["name"])

    missing_backbone = []
    for chain, resseq, icode, resname in residues:
        missing = sorted(
            {"N", "CA", "C", "O"} - residues[(chain, resseq, icode, resname)]
        )
        if missing:
            missing_backbone.append(
                {
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "missing": missing,
                }
            )

    heavy = [a for a in atoms if a["element"] != "H"]
    clashes = []
    valence_counts = [0] * len(heavy)
    for i, atom_i in enumerate(heavy):
        for j in range(i + 1, len(heavy)):
            atom_j = heavy[j]
            same_residue = (atom_i["chain"], atom_i["resseq"], atom_i["icode"]) == (
                atom_j["chain"],
                atom_j["resseq"],
                atom_j["icode"],
            )
            dist = _distance(atom_i["xyz"], atom_j["xyz"])
            if not same_residue and dist < 1.0:
                clashes.append(
                    {
                        "a": f"{atom_i['chain']}:{atom_i['resname']}{atom_i['resseq']}:{atom_i['name']}",
                        "b": f"{atom_j['chain']}:{atom_j['resname']}{atom_j['resseq']}:{atom_j['name']}",
                        "distance": round(dist, 2),
                    }
                )
            radius_i = _COVALENT_RADII.get(atom_i["element"])
            radius_j = _COVALENT_RADII.get(atom_j["element"])
            if radius_i is None or radius_j is None:
                continue
            if dist <= radius_i + radius_j + 0.45:
                valence_counts[i] += 1
                valence_counts[j] += 1

    odd_valences = []
    for atom, valence in zip(heavy, valence_counts):
        max_valence = _MAX_VALENCE.get(atom["element"])
        if max_valence is not None and valence > max_valence:
            odd_valences.append(
                {
                    "atom": f"{atom['chain']}:{atom['resname']}{atom['resseq']}:{atom['name']}",
                    "element": atom["element"],
                    "inferred_valence": valence,
                    "max_expected": max_valence,
                }
            )

    return {
        "atom_count": len(atoms),
        "residue_count_checked": len(residues),
        "missing_backbone": missing_backbone,
        "steric_clashes": clashes[:50],
        "steric_clash_count": len(clashes),
        "odd_valences": odd_valences[:50],
        "odd_valence_count": len(odd_valences),
    }


def _qc_warnings(qc: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    if qc["missing_backbone"]:
        warnings.append(
            {
                "severity": "medium",
                "category": "missing_backbone_atoms",
                "affected_count": len(qc["missing_backbone"]),
                "message": f"{len(qc['missing_backbone'])} residue(s) are missing backbone atoms after prep",
                "action": "inspect the prepared receptor before docking",
            }
        )
    if qc["steric_clash_count"]:
        warnings.append(
            {
                "severity": "high" if qc["steric_clash_count"] > 10 else "medium",
                "category": "steric_clashes",
                "affected_count": qc["steric_clash_count"],
                "message": f"{qc['steric_clash_count']} close heavy-atom contact(s) found in the prepared receptor",
                "action": "inspect clashes near the binding site before docking",
            }
        )
    if qc["odd_valence_count"]:
        warnings.append(
            {
                "severity": "medium",
                "category": "possible_odd_valences",
                "affected_count": qc["odd_valence_count"],
                "message": f"{qc['odd_valence_count']} atom(s) have unusual inferred valence in the prepared receptor",
                "action": "check residue chemistry and retained cofactors/metals",
            }
        )
    return warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def prep_receptor(
    pdb_path: Path,
    chains: list[str],
    output_dir: Path,
    ph: float = 7.4,
    keep_waters: bool = False,
    keep_hetero: list[dict[str, Any]] | None = None,
    receptor_isomers: str = "off",
    residue_state_overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if residue_state_overrides and receptor_isomers != "off":
        raise ReceptorPrepError(
            "Choose either automatic receptor protonation variants or manual "
            "residue-state overrides, not both"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[dict] = []

    noalt = output_dir / "receptor_noalt.pdb"
    n_alt = _strip_alt_conformations(pdb_path, noalt)
    if n_alt:
        warnings.append(
            {
                "severity": "low",
                "category": "alternates_resolved",
                "affected_count": n_alt,
                "message": f"{n_alt} alternate conformation atoms removed (kept A)",
                "action": "none required",
            }
        )

    chain_filtered = output_dir / "receptor_chains.pdb"
    _filter_chains(noalt, chains, chain_filtered)

    # Honour explicit heteroatom keep/remove choices from the wizard. When any are
    # kept, pre-strip the rest here and tell pdbfixer not to remove heterogens.
    prefixer_src = chain_filtered
    n_hetero_removed = 0
    if keep_hetero is not None:
        keep_keys = {
            (d["resname"], d.get("chain", ""), int(d["resseq"])) for d in keep_hetero
        }
        hetero_filtered = output_dir / "receptor_hetero.pdb"
        n_hetero_removed = _apply_hetero_selection(
            chain_filtered, hetero_filtered, keep_keys
        )
        prefixer_src = hetero_filtered
        warnings.append(
            {
                "severity": "low",
                "category": "heteroatoms_selected",
                "affected_count": len(keep_keys),
                "message": f"{len(keep_keys)} heteroatom group(s) kept, {n_hetero_removed} record(s) removed per user selection",
                "action": "confirm kept cofactors/metals are correct",
            }
        )

    base_fixed = output_dir / "receptor_fixed.pdb"
    fixer_output = (
        output_dir / "receptor_fixed_protein.pdb"
        if keep_hetero is not None
        else base_fixed
    )
    try:
        fx = _run_pdbfixer(
            prefixer_src,
            fixer_output,
            keep_waters=keep_waters if keep_hetero is None else False,
            remove_heterogens=True,
        )
    except ReceptorPrepError:
        raise
    except Exception as exc:
        raise ReceptorPrepError(f"pdbfixer error: {exc}") from exc

    if keep_hetero is not None:
        _append_selected_heterogens(
            fixer_output, prefixer_src, base_fixed,
        )

    if fx["missing_residues"]:
        warnings.append(
            {
                "severity": "medium",
                "category": "missing_residues_unresolved",
                "affected_count": fx["missing_residues"],
                "message": f"{fx['missing_residues']} missing residues detected; not rebuilt by this preparation protocol",
                "action": "review unresolved regions near the binding site before docking; consider a more complete experimental structure",
            }
        )

    protonation = {
        "applied": False,
        "tool": None,
        "assignment_count": 0,
        "assignments": [],
        "error": None,
    }
    protonated: Path | None = None
    receptor_variant_modes = ["default"]
    if receptor_isomers == "single":
        protonated_candidate = output_dir / "receptor_protonated.pdb"
        protonation = _run_pdb2pqr_protonation(base_fixed, protonated_candidate, ph)
        if protonation["applied"]:
            protonated = protonated_candidate
            warnings.append(
                {
                    "severity": "low",
                    "category": "receptor_protonation",
                    "affected_count": protonation["assignment_count"],
                    "message": f"PDB2PQR/PROPKA assigned receptor protonation at pH {ph}",
                    "action": "inspect titratable residues near the binding site",
                }
            )
        else:
            receptor_variant_modes = ["propka_ph"]
            protonation = {
                **protonation,
                "tool": protonation.get("tool") or "pdb2pqr_on_kaggle",
                "error": f"local_failed_deferred_to_kaggle: {protonation.get('error')}",
            }
            warnings.append(
                {
                    "severity": "medium",
                    "category": "receptor_protonation_deferred",
                    "affected_count": 0,
                    "message": "Local PDB2PQR/PROPKA failed; single-state receptor protonation will run inside Kaggle notebooks",
                    "action": "inspect Kaggle notebook output if receptor prep fails",
                }
            )
    elif receptor_isomers == "variants":
        receptor_variant_modes = ["default", "propka_ph"]
        protonation = {
            "applied": False,
            "tool": "pdb2pqr_on_kaggle",
            "assignment_count": 0,
            "assignments": [],
            "error": "deferred_to_kaggle",
        }
        warnings.append(
            {
                "severity": "low",
                "category": "receptor_protonation_deferred",
                "affected_count": 0,
                "message": "PDB2PQR/PROPKA receptor variant prep will run inside Kaggle notebooks",
                "action": "inspect Kaggle notebook output if variant prep fails",
            }
        )
    else:
        warnings.append(
            {
                "severity": "low",
                "category": "histidine_protonation",
                "affected_count": 0,
                "message": f"Histidine states assigned at pH {ph} - guesses only",
                "action": "verify HIE/HID if histidines are in binding site",
            }
        )

    if receptor_isomers == "single" and protonated is not None:
        primary_fixed = protonated
    else:
        primary_fixed = base_fixed

    manual_states = {
        "assignment_count": 0,
        "assignments": [],
    }
    if residue_state_overrides:
        manual_fixed = output_dir / "receptor_manual_states.pdb"
        manual_states = _apply_residue_state_overrides(
            primary_fixed, manual_fixed, residue_state_overrides
        )
        primary_fixed = manual_fixed
        warnings.append(
            {
                "severity": "low",
                "category": "manual_residue_states",
                "affected_count": manual_states["assignment_count"],
                "message": "User-selected residue protonation states override automatic assignments",
                "action": "confirm the selected states are appropriate for the binding site",
            }
        )

    rama_angles = ramachandran_angles(primary_fixed)
    rama_plot = _render_ramachandran_plot(rama_angles)
    sanity = receptor_sanity_checks(primary_fixed)
    prepared_metal_qc = metal_site_qc(primary_fixed)
    prep_protocol_audit = _prep_protocol_audit(chain_filtered, primary_fixed)
    if prepared_metal_qc["metal_count"]:
        warnings.append(
            {
                "severity": "high",
                "category": "metal_scoring_validation_required",
                "affected_count": prepared_metal_qc["metal_count"],
                "message": "Retained receptor metal(s) require target-specific scoring validation",
                "action": "review metal-site QC and choose a user-selected validated scoring method",
            }
        )

    warnings.extend(_qc_warnings(sanity))

    try:
        primary_pdbqt, mv = _run_meeko_receptor(primary_fixed, output_dir)
    except ReceptorPrepError:
        raise
    except Exception as exc:
        raise ReceptorPrepError(f"Meeko error: {exc}") from exc

    hetero_retention = _hetero_retention_report(primary_pdbqt, keep_hetero)
    retention_path = output_dir / "receptor_hetero_retention.json"
    retention_path.write_text(json.dumps(hetero_retention, indent=2) + "\n")
    if not hetero_retention["complete"]:
        missing = ", ".join(
            f"{item['resname']} {item['chain']}:{item['resseq']}"
            for item in hetero_retention["missing"]
        )
        raise ReceptorPrepError(
            "Explicitly retained heterogen(s) disappeared during receptor PDBQT "
            f"preparation: {missing}. See {retention_path}."
        )

    variant_entries: list[dict[str, Any]] = [
        {"name": "default", "pdb_path": primary_fixed, "pdbqt_path": primary_pdbqt}
    ]
    if (
        receptor_isomers == "single"
        and protonated is not None
        and protonated != primary_fixed
    ):
        try:
            protonated_pdbqt, _ = _run_meeko_receptor(protonated, output_dir)
            variant_entries.append(
                {
                    "name": "propka_ph",
                    "pdb_path": protonated,
                    "pdbqt_path": protonated_pdbqt,
                }
            )
        except ReceptorPrepError as exc:
            warnings.append(
                {
                    "severity": "medium",
                    "category": "receptor_variant_unavailable",
                    "affected_count": 0,
                    "message": f"Could not prepare PROPKA receptor variant ({exc})",
                    "action": "run with a single receptor or inspect receptor prep logs",
                }
            )

    import hashlib

    unique_variants: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for entry in variant_entries:
        pdbqt_path = Path(entry["pdbqt_path"])
        digest = hashlib.sha256(pdbqt_path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_variants.append(entry)
    if receptor_isomers == "variants" and len(unique_variants) < len(variant_entries):
        warnings.append(
            {
                "severity": "low",
                "category": "receptor_variants_deduped",
                "affected_count": len(variant_entries) - len(unique_variants),
                "message": "Duplicate receptor variant PDBQT files were removed before docking",
                "action": "none required",
            }
        )

    receptor_pdb = output_dir / "receptor_prep.pdb"
    shutil.copy2(primary_fixed, receptor_pdb)

    return {
        "pdbqt_path": primary_pdbqt,
        "variant_pdbqt_paths": [Path(v["pdbqt_path"]) for v in unique_variants],
        "receptor_variant_modes": receptor_variant_modes,
        "receptor_pdb_path": receptor_pdb,
        "fixed_pdb_path": primary_fixed,
        "report": {
            "chains_selected": chains,
            "residue_count": fx["residue_count"],
            "missing_residues": fx["missing_residues"],
            "alternates_resolved": n_alt,
            "ramachandran": {
                "angle_count": len(rama_angles),
                "angles": rama_angles,
                "plot_png_base64": rama_plot,
            },
            "sanity_checks": sanity,
            "metal_site_qc": prepared_metal_qc,
            "prep_protocol_audit": prep_protocol_audit,
            "receptor_protonation": protonation,
            "manual_residue_states": manual_states,
            "heteroatom_retention": hetero_retention,
            "heteroatom_retention_path": str(retention_path),
            "receptor_variants": [
                {
                    "name": v["name"],
                    "pdb_path": str(v["pdb_path"]),
                    "pdbqt_path": str(v["pdbqt_path"]),
                }
                for v in unique_variants
            ],
            "receptor_variant_modes": receptor_variant_modes,
            "tools": {
                "pdbfixer": fx["pdbfixer_version"],
                "meeko": mv,
                "protonation": protonation.get("tool"),
            },
            "warnings": warnings,
        },
    }
