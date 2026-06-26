from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Gypsum-DL is an optional prep backend. It is a CLI/subprocess tool (no clean
# import API), so we locate a runnable entry point and shell out to it. Every
# failure path returns the input SMILES unchanged, so prep degrades gracefully
# to the current single-form behaviour when Gypsum-DL is absent or errors.


def _gypsum_command() -> list[str] | None:
    exe = shutil.which("gypsum-dl") or shutil.which("gypsum_dl")
    if exe:
        return [exe]
    try:
        import gypsum_dl  # noqa: F401
    except ImportError:
        return None
    driver = Path(gypsum_dl.__file__).parent / "run_gypsum_dl.py"
    if driver.exists():
        return [sys.executable, str(driver)]
    return [sys.executable, "-m", "gypsum_dl"]


def gypsum_available() -> bool:
    return _gypsum_command() is not None


def _build_args(cmd, src, out_dir, ph, max_variants, protonation, tautomers, stereo, ring) -> list[str]:
    args = [
        *cmd,
        "--source", str(src),
        "--output_folder", str(out_dir),
        "--2d_output_only",          # emit enumerated 2D; our own embed does 3D
        "--separate_output_files",
        "--max_variants_per_compound", str(max_variants),
        "--job_manager", "serial",
    ]
    if protonation:
        args += ["--min_ph", str(ph - 1.0), "--max_ph", str(ph + 1.0)]
    else:
        # collapse the pH window so only the dominant state at ph survives
        args += ["--min_ph", str(ph), "--max_ph", str(ph)]
    if not tautomers:
        args.append("--skip_making_tautomers")
    if not stereo:
        args.append("--skip_enumerate_chiral_mol")
    if not ring:
        args.append("--skip_alternate_ring_conformations")
    return args


def _collect_variants(out_dir: Path, fallback: str) -> list[str]:
    from rdkit import Chem

    seen: set[str] = set()
    variants: list[str] = []
    for sdf in sorted(Path(out_dir).glob("*.sdf")):
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
        for mol in supplier:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if smi and smi not in seen:
                seen.add(smi)
                variants.append(smi)
    return variants or [fallback]


def enumerate_variants(
    smiles: str,
    ph: float = 7.4,
    max_variants: int = 4,
    *,
    protonation: bool = True,
    tautomers: bool = True,
    stereo: bool = False,
    ring: bool = False,
    timeout: int = 120,
) -> list[str]:
    cmd = _gypsum_command()
    if cmd is None:
        return [smiles]
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src = tdp / "in.smi"
            src.write_text(f"{smiles}\tlig\n")
            out_dir = tdp / "out"
            out_dir.mkdir()
            args = _build_args(cmd, src, out_dir, ph, max_variants, protonation, tautomers, stereo, ring)
            subprocess.run(args, check=True, capture_output=True, timeout=timeout)
            return _collect_variants(out_dir, smiles)
    except Exception:
        return [smiles]
