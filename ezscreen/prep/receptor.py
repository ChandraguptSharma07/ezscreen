from __future__ import annotations

import base64
import io
import math
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
    keep_records = {"TER", "MODEL", "ENDMDL", "END", "CRYST1", "REMARK", "HEADER", "TITLE"}  # noqa: E501
    lines = []
    for line in src.read_text(errors="ignore").splitlines():
        record = line[:6].rstrip()
        if record in ("ATOM", "HETATM"):
            if line[21:22] in chain_set:
                lines.append(line)
        elif record == "SEQRES":
            if line[11:12] in chain_set:
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
# Heteroatom inventory + selection
# ---------------------------------------------------------------------------

_WATERS: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "D2O"})

_METALS: frozenset[str] = frozenset({
    "ZN", "MG", "MN", "FE", "FE2", "CA", "NA", "K", "CU", "CU1", "CO", "NI",
    "CD", "HG", "MO", "W",
})

# Organic prosthetic groups / cofactors worth keeping in the receptor by default.
_COFACTORS: frozenset[str] = frozenset({
    "HEM", "HEC", "HEA", "HEB", "SRM", "VER",
    "NAD", "NAP", "NAI", "NDP", "NAJ",
    "FAD", "FMN", "FDA", "FADH",
    "ATP", "ADP", "AMP", "ANP", "AGS", "GTP", "GDP", "GNP", "GSP",
    "SAM", "SAH", "PLP", "PMP", "TPP", "TDP", "COA", "ACO", "BTN",
    "CLA", "BCL", "B12", "COB", "PQQ", "MGD", "F43", "H4B", "BH4",
})

# Crystallographic additives that are never biologically relevant.
_ADDITIVE_HET: frozenset[str] = frozenset({
    "GOL", "EDO", "MPD", "PGO", "PG4", "1PE", "PE8", "PE3", "PE4", "PE5", "PE6",
    "SO4", "PO4", "NO3", "CL", "BR", "IOD", "ACT", "ACY", "ACE", "DMS", "DMF",
    "EOH", "IPA", "TRS", "HEP", "MES", "MOH", "TAR", "CIT", "FMT", "IMD", "DIO",
    "TLA", "BOG", "OES", "FLC", "BME", "PEG",
})


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
        return "ligand", False   # a co-crystal ligand — usually the thing you displace
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
    for (resname, chain, resseq) in order:
        n = groups[(resname, chain, resseq)]
        category, keep = _classify_hetero(resname, n)
        inventory.append({
            "resname": resname, "chain": chain, "resseq": resseq,
            "n_atoms": n, "category": category, "keep_default": keep,
        })
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
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    try:
        from pdbfixer import __version__ as v
    except ImportError:
        v = "unknown"
    return {"missing_residues": n_missing, "residue_count": residue_count, "pdbfixer_version": v}


def _infer_element(atom_name: str) -> str:
    """Guess element symbol from PDB atom name (e.g. 'CA' → 'C', 'FE' → 'FE')."""
    name = atom_name.strip().lstrip("0123456789")
    if not name:
        return ""
    # Two-letter elements common in proteins
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
                elem = _infer_element(line[12:16])
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
            element   = line[76:78].strip() if len(line) >= 78 else ""
            atom_name = line[12:16].strip()
            if element == "H" or (not element and atom_name.startswith("H")):
                continue
            if atom_name in {"OXT", "OT1", "OT2"}:
                continue
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def _run_meeko_receptor(src: Path, output_dir: Path) -> tuple[Path, str]:
    exe = shutil.which("mk_prepare_receptor")
    if exe is None:
        raise ReceptorPrepError("mk_prepare_receptor not found — pip install meeko")

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
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    console.print("  [dim]meeko: converting receptor to PDBQT...[/dim]")
    try:
        result = subprocess.run(
            [exe, "-i", str(clean), "-p", str(pdbqt), "--allow_bad_res"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReceptorPrepError("mk_prepare_receptor timed out after 180 s") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip() if isinstance(result.stderr, bytes) else result.stderr.strip()
        meeko_err = stderr
        console.print("  [yellow]meeko failed — trying obabel fallback...[/yellow]")
        try:
            return _run_obabel_receptor(clean, output_dir)
        except ReceptorPrepError as ob_exc:
            raise ReceptorPrepError(f"mk_prepare_receptor failed:\n{meeko_err}") from ob_exc
    try:
        import meeko
        mv = meeko.__version__
    except (ImportError, AttributeError):
        mv = "unknown"
    return pdbqt, mv


def _run_obabel_receptor(src: Path, output_dir: Path) -> tuple[Path, str]:
    exe = shutil.which("obabel")
    if exe is None:
        raise ReceptorPrepError("obabel not found — install Open Babel: https://openbabel.org")
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


# ---------------------------------------------------------------------------
# Receptor QC
# ---------------------------------------------------------------------------

_STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})

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
            atoms.append({
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
            })
        except (IndexError, ValueError):
            continue
    return atoms


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
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
    residues: dict[tuple[str, int, str, str], dict[str, tuple[float, float, float]]] = {}
    order: list[tuple[str, int, str, str]] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in {"N", "CA", "C"}:
            continue
        try:
            key = _atom_key(line)
            xyz = (_pdb_float(line, 30, 38), _pdb_float(line, 38, 46), _pdb_float(line, 46, 54))
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
            "C" in prev_atoms
            and {"N", "CA", "C"}.issubset(atoms)
            and "N" in next_atoms
        )
        if not required:
            continue
        try:
            phi = _dihedral_degrees(prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"])
            psi = _dihedral_degrees(atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"])
        except ValueError:
            continue
        chain, resseq, icode, resname = key
        angles.append({
            "chain": chain,
            "resseq": resseq,
            "icode": icode,
            "resname": resname,
            "phi": round(phi, 2),
            "psi": round(psi, 2),
        })
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
        missing = sorted({"N", "CA", "C", "O"} - residues[(chain, resseq, icode, resname)])
        if missing:
            missing_backbone.append({
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "resname": resname,
                "missing": missing,
            })

    heavy = [a for a in atoms if a["element"] != "H"]
    clashes = []
    valence_counts = [0] * len(heavy)
    for i, atom_i in enumerate(heavy):
        for j in range(i + 1, len(heavy)):
            atom_j = heavy[j]
            same_residue = (
                atom_i["chain"], atom_i["resseq"], atom_i["icode"]
            ) == (atom_j["chain"], atom_j["resseq"], atom_j["icode"])
            dist = _distance(atom_i["xyz"], atom_j["xyz"])
            if not same_residue and dist < 1.0:
                clashes.append({
                    "a": f"{atom_i['chain']}:{atom_i['resname']}{atom_i['resseq']}:{atom_i['name']}",
                    "b": f"{atom_j['chain']}:{atom_j['resname']}{atom_j['resseq']}:{atom_j['name']}",
                    "distance": round(dist, 2),
                })
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
            odd_valences.append({
                "atom": f"{atom['chain']}:{atom['resname']}{atom['resseq']}:{atom['name']}",
                "element": atom["element"],
                "inferred_valence": valence,
                "max_expected": max_valence,
            })

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
        warnings.append({
            "severity": "medium",
            "category": "missing_backbone_atoms",
            "affected_count": len(qc["missing_backbone"]),
            "message": f"{len(qc['missing_backbone'])} residue(s) are missing backbone atoms after prep",
            "action": "inspect the prepared receptor before docking",
        })
    if qc["steric_clash_count"]:
        warnings.append({
            "severity": "high" if qc["steric_clash_count"] > 10 else "medium",
            "category": "steric_clashes",
            "affected_count": qc["steric_clash_count"],
            "message": f"{qc['steric_clash_count']} close heavy-atom contact(s) found in the prepared receptor",
            "action": "inspect clashes near the binding site before docking",
        })
    if qc["odd_valence_count"]:
        warnings.append({
            "severity": "medium",
            "category": "possible_odd_valences",
            "affected_count": qc["odd_valence_count"],
            "message": f"{qc['odd_valence_count']} atom(s) have unusual inferred valence in the prepared receptor",
            "action": "check residue chemistry and retained cofactors/metals",
        })
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

    # Honour explicit heteroatom keep/remove choices from the wizard. When any are
    # kept, pre-strip the rest here and tell pdbfixer not to remove heterogens.
    prefixer_src = chain_filtered
    remove_heterogens = True
    n_hetero_removed = 0
    if keep_hetero is not None:
        keep_keys = {
            (d["resname"], d.get("chain", ""), int(d["resseq"])) for d in keep_hetero
        }
        hetero_filtered = output_dir / "receptor_hetero.pdb"
        n_hetero_removed = _apply_hetero_selection(chain_filtered, hetero_filtered, keep_keys)
        prefixer_src = hetero_filtered
        remove_heterogens = False
        warnings.append({
            "severity": "low", "category": "heteroatoms_selected",
            "affected_count": len(keep_keys),
            "message": f"{len(keep_keys)} heteroatom group(s) kept, {n_hetero_removed} record(s) removed per user selection",
            "action": "confirm kept cofactors/metals are correct",
        })

    fixed = output_dir / "receptor_fixed.pdb"
    try:
        fx = _run_pdbfixer(
            prefixer_src, fixed, keep_waters=keep_waters,
            remove_heterogens=remove_heterogens,
        )
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

    rama_angles = ramachandran_angles(fixed)
    rama_plot = _render_ramachandran_plot(rama_angles)
    sanity = receptor_sanity_checks(fixed)
    warnings.extend(_qc_warnings(sanity))

    try:
        pdbqt, mv = _run_meeko_receptor(fixed, output_dir)
    except ReceptorPrepError:
        raise
    except Exception as exc:
        raise ReceptorPrepError(f"Meeko error: {exc}") from exc

    receptor_pdb = output_dir / "receptor_prep.pdb"
    shutil.copy2(fixed, receptor_pdb)

    return {
        "pdbqt_path": pdbqt,
        "receptor_pdb_path": receptor_pdb,
        "fixed_pdb_path": fixed,
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
            "tools": {"pdbfixer": fx["pdbfixer_version"], "meeko": mv},
            "warnings": warnings,
        },
    }
