from __future__ import annotations

import csv
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Crystallographic additives — never use as binding site reference
# ---------------------------------------------------------------------------

_ADDITIVES: frozenset[str] = frozenset({
    # Water
    "HOH", "WAT", "DOD", "D2O",
    # Cryoprotectants / PEG
    "GOL", "EDO", "MPD", "PGO", "PG4", "1PE", "PE8", "PE3", "PE4", "PE5", "PE6",
    # Common ions and salts
    "SO4", "PO4", "NO3", "CL", "BR", "NA", "MG", "ZN", "CA", "MN", "FE", "CU", "CD",
    # Solvents and buffers
    "DMS", "DMF", "ACT", "ACY", "ACE", "EOH", "IPA", "TRS", "HEP", "MES", "MOH",
    # Other common additives
    "TAR", "CIT", "FMT", "IMD", "DIO", "TLA", "BOG", "OES",
})

P2RANK_DIR = Path.home() / ".ezscreen" / "tools" / "p2rank"


# ---------------------------------------------------------------------------
# PDB coordinate parsing
# ---------------------------------------------------------------------------

def _parse_atoms(pdb_path: Path, record_prefix: str) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line.startswith(record_prefix):
            try:
                atoms.append({
                    "name":    line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain":   line[21:22].strip(),
                    "resseq":  int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                pass
    return atoms


def _box_from_coords(
    coords: list[tuple[float, float, float]], padding: float
) -> dict[str, Any]:
    if not coords:
        raise ValueError("No coordinates provided for box calculation")
    xs, ys, zs = zip(*coords)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    sx = max(xs) - min(xs) + 2 * padding
    sy = max(ys) - min(ys) + 2 * padding
    sz = max(zs) - min(zs) + 2 * padding
    return {
        "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
        "size":   [round(sx, 3), round(sy, 3), round(sz, 3)],
        "volume_angstrom3": round(sx * sy * sz, 1),
    }


# ---------------------------------------------------------------------------
# Co-crystal ligand detection
# ---------------------------------------------------------------------------

def find_cocrystal_ligands(pdb_path: Path) -> list[dict[str, Any]]:
    """Return drug-like HETATM residues (filtered of crystallographic additives)."""
    atoms = _parse_atoms(pdb_path, "HETATM")
    groups: dict[tuple, list] = {}
    for a in atoms:
        groups.setdefault((a["resname"], a["chain"], a["resseq"]), []).append(
            (a["x"], a["y"], a["z"])
        )
    ligands = []
    for (resname, chain, resseq), coords in groups.items():
        if resname in _ADDITIVES or len(coords) < 5:
            continue
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        ligands.append({
            "resname": resname, "chain": chain, "resseq": resseq,
            "coords": coords, "centroid": [round(cx, 3), round(cy, 3), round(cz, 3)],
        })
    return ligands


def box_from_cocrystal(ligand: dict[str, Any], padding: float = 5.0) -> dict[str, Any]:
    box = _box_from_coords(ligand["coords"], padding)
    box.update({"method": "co_crystal", "reference_ligand": ligand["resname"],
                "reference_chain": ligand["chain"]})
    return box


# ---------------------------------------------------------------------------
# Residue-defined box
# ---------------------------------------------------------------------------

def box_from_residues(
    pdb_path: Path,
    residue_ids: list[int],
    chains: list[str],
    padding: float = 8.0,
) -> dict[str, Any]:
    atoms = _parse_atoms(pdb_path, "ATOM  ")
    ca_coords = [
        (a["x"], a["y"], a["z"]) for a in atoms
        if a["name"] == "CA" and a["resseq"] in residue_ids and a["chain"] in chains
    ]
    if not ca_coords:
        raise ValueError(f"No Cα atoms found for residues {residue_ids} in chains {chains}")
    box = _box_from_coords(ca_coords, padding)
    box.update({"method": "residue_defined", "residues": residue_ids})
    return box


# ---------------------------------------------------------------------------
# P2Rank
# ---------------------------------------------------------------------------

_P2RANK_RELEASE = "https://github.com/rdk/p2rank/releases/download/2.5/p2rank_2.5.tar.gz"


def _p2rank_exe() -> Path | None:
    import sys
    candidates = (
        [P2RANK_DIR / "prank.bat", P2RANK_DIR / "prank", P2RANK_DIR / "prank.sh"]
        if sys.platform == "win32"
        else [P2RANK_DIR / "prank", P2RANK_DIR / "prank.sh"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = shutil.which("prank.bat" if sys.platform == "win32" else "prank")
    return Path(found) if found else None


def download_p2rank(progress_cb=None) -> Path:
    """Download and unpack P2Rank into ~/.ezscreen/tools/p2rank/. Returns exe path."""
    import tarfile
    P2RANK_DIR.mkdir(parents=True, exist_ok=True)
    archive = P2RANK_DIR.parent / "p2rank.tar.gz"

    def _reporthook(count, block, total):
        if progress_cb and total > 0:
            progress_cb(min(count * block, total), total)

    console.print("[dim]Downloading P2Rank 2.5...[/dim]")
    urllib.request.urlretrieve(_P2RANK_RELEASE, archive, reporthook=_reporthook)

    console.print("[dim]Unpacking P2Rank...[/dim]")
    with tarfile.open(archive) as tf:
        tf.extractall(P2RANK_DIR.parent)

    # The tarball unpacks to p2rank_2.5/ — move contents into p2rank/
    extracted = P2RANK_DIR.parent / "p2rank_2.5"
    if extracted.exists() and extracted != P2RANK_DIR:
        if P2RANK_DIR.exists():
            shutil.rmtree(P2RANK_DIR)
        extracted.rename(P2RANK_DIR)

    archive.unlink(missing_ok=True)

    exe = _p2rank_exe()
    if exe is None:
        raise RuntimeError("P2Rank downloaded but prank executable not found")
    exe.chmod(exe.stat().st_mode | 0o111)
    return exe


def run_p2rank(
    pdb_path: Path, output_dir: Path, alphafold: bool = False
) -> list[dict[str, Any]]:
    """Run P2Rank and return up to 3 pocket candidates. Downloads P2Rank if missing."""
    import sys

    exe = _p2rank_exe()
    if exe is None:
        console.print("[dim]P2Rank not found — downloading...[/dim]")
        try:
            exe = download_p2rank()
        except Exception as exc:
            raise RuntimeError(f"P2Rank download failed: {exc}") from exc

    # P2Rank requires Java — check upfront for a clear error message
    java_bin = shutil.which("java")
    if java_bin is None:
        raise RuntimeError(
            "Java not found. P2Rank requires Java 11+. "
            "Install from https://adoptium.net and restart."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # On Windows, invoke Java directly — avoids prank.bat path/JAVA_HOME issues
        p2rank_jar = exe.parent / "bin" / "p2rank.jar"
        lib_glob   = str(exe.parent / "bin" / "lib" / "*")
        classpath  = f"{p2rank_jar};{lib_glob}"
        cmd = [
            java_bin, "-Xmx2048m", "-cp", classpath,
            "cz.siret.prank.program.Main",
            "predict", "-f", str(pdb_path), "-o", str(output_dir),
        ]
        if alphafold:
            cmd += ["-profile", "alphafold"]
    else:
        cmd = [str(exe), "predict", "-f", str(pdb_path), "-o", str(output_dir)]
        if alphafold:
            cmd += ["-profile", "alphafold"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"P2Rank exited with code {result.returncode}: {err[:300]}")

    csv_path: Path | None = None
    for f in output_dir.rglob("*_predictions.csv"):
        csv_path = f
        break
    if not csv_path:
        return []

    pockets: list[dict[str, Any]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for i, row in enumerate(reader):
            if i >= 3:
                break
            try:
                cx, cy, cz = float(row["center_x"]), float(row["center_y"]), float(row["center_z"])
                score = float(row.get("score", 0))
                prob  = float(row.get("probability", 0))
                size  = max(20.0, float(row.get("sas_points", 100)) ** (1/3) * 2 + 6)
                pockets.append({
                    "rank": i + 1, "score": round(score, 3), "probability": round(prob, 3),
                    "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
                    "size": [round(size, 3)] * 3,
                    "volume_angstrom3": round(size ** 3, 1),
                    "method": "p2rank",
                })
            except (ValueError, KeyError):
                continue
    return pockets


# ---------------------------------------------------------------------------
# Blind docking
# ---------------------------------------------------------------------------

def box_blind(pdb_path: Path, padding: float = 4.0) -> dict[str, Any]:
    atoms = _parse_atoms(pdb_path, "ATOM  ")
    if not atoms:
        raise ValueError("No ATOM records found")
    box = _box_from_coords([(a["x"], a["y"], a["z"]) for a in atoms], padding)
    box.update({
        "method": "blind",
        "warning": (
            "Blind docking uses the entire protein — slow, noisy, "
            "and misses buried pockets. Only use without prior binding site knowledge."
        ),
    })
    return box


# ---------------------------------------------------------------------------
# Box volume validation
# ---------------------------------------------------------------------------

def validate_box(box: dict[str, Any]) -> list[dict[str, Any]]:
    vol = box.get("volume_angstrom3", 0)
    warnings: list[dict[str, Any]] = []
    if vol > 30_000:
        warnings.append({
            "severity": "high", "category": "box_too_large", "affected_count": 1,
            "message": f"Box volume {vol:.0f} Å³ is very large (>30 000) — docking will be slow and non-specific",
            "action": "narrow the binding site definition",
        })
    elif vol < 1_500:
        warnings.append({
            "severity": "high", "category": "box_too_small", "affected_count": 1,
            "message": f"Box volume {vol:.0f} Å³ is very small (<1 500) — ligands may not fit",
            "action": "expand the binding site or increase padding",
        })
    return warnings
