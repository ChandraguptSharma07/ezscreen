from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn

from ezscreen.errors import LigandPrepError

console = Console()
_SHARD_SIZE = 5_000


# ---------------------------------------------------------------------------
# Scrubber import — git+ install preferred, vendor fallback
# ---------------------------------------------------------------------------

def _get_scrubber():
    try:
        from scrubber import Scrubber
        return Scrubber
    except ImportError:
        pass
    try:
        from ezscreen.vendor.scrubber import SCRUBBER_AVAILABLE, Scrubber
        return Scrubber if SCRUBBER_AVAILABLE else None
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Input scanning
# ---------------------------------------------------------------------------

def scan_input(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in (".sdf", ".smi", ".smiles"):
            return [path]
        raise LigandPrepError(f"Unsupported file type: {path.suffix}")
    files: list[Path] = []
    for ext in ("*.sdf", "*.smi", "*.smiles"):
        files.extend(sorted(path.rglob(ext)))
    if not files:
        raise LigandPrepError(f"No .sdf or .smi files found in {path}")
    return files


# ---------------------------------------------------------------------------
# Per-molecule prep
# ---------------------------------------------------------------------------

def _load_supplier(file_path: Path):
    from rdkit.Chem import SDMolSupplier, SmilesMolSupplier
    if file_path.suffix.lower() == ".sdf":
        return SDMolSupplier(str(file_path), removeHs=False, sanitize=True)
    return SmilesMolSupplier(str(file_path), delimiter="\t ", titleLine=False)


def _scrub(mol, Scrubber, ph: float):
    if Scrubber is None:
        return mol
    try:
        result = Scrubber(pH=ph)(mol)
        return result
    except Exception:
        return None


def _embed_3d(mol):
    from rdkit.Chem import AddHs, AllChem
    mol_h = AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    return mol_h


def _to_pdbqt(mol) -> str | None:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        setups = MoleculePreparation().prepare(mol)
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        return pdbqt if ok else None
    except Exception:
        return None


# Atomic numbers AutoDock4 / UniDock can handle.
# Elements outside this set produce atom-type parse errors at docking time.
_AUTODOCK_SUPPORTED_ATOMIC_NUMS = frozenset({
    1,   # H
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    12,  # Mg
    15,  # P
    16,  # S
    17,  # Cl
    20,  # Ca
    25,  # Mn
    26,  # Fe
    30,  # Zn
    35,  # Br
    53,  # I
})


def _prep_one(mol, Scrubber, ph: float) -> tuple[str | None, str | None]:
    try:
        from rdkit.Chem import SanitizeMol
        SanitizeMol(mol)
    except Exception:
        return None, "sanitization"

    # Reject molecules with elements AutoDock/UniDock can't handle.
    # Meeko writes these as bare element symbols (e.g. "B" for Boron) which
    # UniDock rejects at runtime with "Atom type B is not a valid AutoDock type".
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in _AUTODOCK_SUPPORTED_ATOMIC_NUMS:
            return None, "unsupported_atoms"

    scrubbed = _scrub(mol, Scrubber, ph)
    if scrubbed is None:
        return None, "sanitization"

    mol_3d = _embed_3d(scrubbed)
    if mol_3d is None:
        return None, "conformer_generation"

    pdbqt = _to_pdbqt(mol_3d)
    if pdbqt is None:
        return None, "unsupported_atoms"

    return pdbqt, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prep_ligands(
    input_path: Path,
    output_dir: Path,
    ph: float = 7.4,
    enumerate_tautomers: bool = False,
    shard_size: int = _SHARD_SIZE,
    n_shards: int | None = None,
) -> dict[str, Any]:
    import math
    from rdkit.Chem import SDWriter

    output_dir.mkdir(parents=True, exist_ok=True)
    Scrubber = _get_scrubber()
    files = scan_input(input_path)

    failed_path = output_dir / "failed_prep.sdf"
    failed_writer = SDWriter(str(failed_path))

    failures = {"sanitization": 0, "conformer_generation": 0, "unsupported_atoms": 0}
    shard_buf: list[str] = []
    shard_paths: list[Path] = []
    index_rows: list[dict] = []   # ligand id → name + smiles
    shard_idx = total = prep_passed = prep_failed = 0

    def _flush() -> None:
        nonlocal shard_idx
        if not shard_buf:
            return
        p = output_dir / f"shard_{shard_idx:03d}.pdbqt"
        p.write_text("\n".join(shard_buf))
        shard_paths.append(p)
        shard_buf.clear()
        shard_idx += 1

    from rdkit.Chem import MolToSmiles

    # Phase 1: load all molecules (sequential — suppliers are not thread-safe)
    all_mols:  list = []
    all_meta:  list[tuple[str, str]] = []   # (smiles, mol_name)
    bad_mols:  list = []                     # None-supplier failures to count
    for fp in files:
        for mol in _load_supplier(fp):
            if mol is None:
                bad_mols.append(None)
                continue
            total += 1
            smiles   = MolToSmiles(mol)
            mol_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            if not mol_name:
                for _prop in ("Catalog_ID", "ID", "Name", "IDNUMBER", "PUBCHEM_COMPOUND_CID"):
                    if mol.HasProp(_prop):
                        mol_name = mol.GetProp(_prop).strip()
                        break
            all_mols.append(mol)
            all_meta.append((smiles, mol_name))

    prep_failed += len(bad_mols)
    failures["sanitization"] += len(bad_mols)

    # Phase 2: parallel conformer generation + PDBQT conversion.
    # RDKit's ETKDGv3 and MMFF are C++ extensions that release the GIL,
    # so ThreadPoolExecutor gives near-linear speedup without pickling.
    n_workers = min(max(os.cpu_count() or 1, 1), 16)
    console.print(f"  [dim]Prepping {total} ligand(s) with {n_workers} worker(s)...[/dim]")

    results: list[tuple | None] = [None] * len(all_mols)

    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  BarColumn(), TaskProgressColumn(), console=console) as prog:
        task = prog.add_task("Prepping ligands...", total=len(all_mols))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(_prep_one, mol, Scrubber, ph): i
                for i, mol in enumerate(all_mols)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                prog.advance(task)

    # If a target shard count was requested, recompute shard_size now that we
    # know how many ligands actually passed prep (can't know this before Phase 2).
    if n_shards is not None and n_shards > 0:
        n_passing = sum(1 for r in results if r is not None and r[0] is not None)
        if n_passing > 0:
            shard_size = max(1, math.ceil(n_passing / n_shards))

    # Phase 3: assemble shards in original order (preserves lig_id sequence)
    for (smiles, mol_name), mol, result in zip(all_meta, all_mols, results):
        pdbqt, reason = result  # type: ignore[misc]
        if pdbqt:
            lig_id = f"lig_{prep_passed:05d}"
            index_rows.append({"ligand": lig_id, "name": mol_name, "smiles": smiles})
            shard_buf.append(f"REMARK lig_id {lig_id}\n" + pdbqt)
            prep_passed += 1
            if len(shard_buf) >= shard_size:
                _flush()
        else:
            prep_failed += 1
            failures[reason] += 1
            failed_writer.write(mol)
    _flush()

    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ligand", "name", "smiles"])
        w.writeheader()
        w.writerows(index_rows)

    failed_writer.close()

    warnings: list[dict] = []
    if Scrubber is None:
        warnings.append({
            "severity": "medium", "category": "scrubber_unavailable",
            "affected_count": total,
            "message": "scrub.py not available — protonation/cleanup skipped",
            "action": "pip install 'ezscreen[scrubber]'",
        })

    try:
        import rdkit
        rv = rdkit.__version__
    except Exception:
        rv = "unknown"
    try:
        import meeko
        mv = meeko.__version__
    except Exception:
        mv = "unknown"
    sv = "vendor" if (Scrubber and "vendor" in getattr(Scrubber, "__module__", "")) else "git+"

    return {
        "shard_paths": shard_paths,
        "report": {
            "input_source": str(input_path),
            "input_files": len(files),
            "total_input": total,
            "prep_passed": prep_passed,
            "prep_failed": prep_failed,
            "prep_failures": failures,
            "failed_prep_file": str(failed_path) if prep_failed else None,
            "tautomers_enumerated": enumerate_tautomers,
            "protonation_ph": ph,
            "tools": {"scrubber": sv, "rdkit": rv, "meeko": mv},
            "warnings": warnings,
        },
    }
