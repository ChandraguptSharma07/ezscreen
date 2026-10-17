from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn

from ezscreen.errors import LigandPrepError

console = Console()
_SHARD_SIZE = 5_000
_RDKIT_INPUT_EXTS = (".sdf", ".smi", ".smiles", ".ism")
_OPENBABEL_INPUT_EXTS = (".mol2", ".mol", ".pdb")
_LIGAND_INPUT_EXTS = _RDKIT_INPUT_EXTS + _OPENBABEL_INPUT_EXTS
_NORMALIZATION_PROFILES = frozenset({"legacy", "generalized_v1"})
_RECOGNIZED_MONATOMIC_COUNTERIONS = frozenset(
    {
        (3, 1),  # lithium
        (9, -1),  # fluoride
        (11, 1),  # sodium
        (12, 2),  # magnesium
        (17, -1),  # chloride
        (19, 1),  # potassium
        (20, 2),  # calcium
        (35, -1),  # bromide
        (53, -1),  # iodide
    }
)


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
        if path.suffix.lower() in _LIGAND_INPUT_EXTS:
            return [path]
        raise LigandPrepError(f"Unsupported file type: {path.suffix}")
    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in _LIGAND_INPUT_EXTS
    )
    if not files:
        raise LigandPrepError(
            f"No ligand files (.sdf, .smi, .smiles, .ism, .mol2, .mol, .pdb) found in {path}"
        )
    return files


# ---------------------------------------------------------------------------
# Per-molecule prep
# ---------------------------------------------------------------------------


def _load_supplier(file_path: Path):
    from rdkit.Chem import SDMolSupplier, SmilesMolSupplier

    if file_path.suffix.lower() == ".sdf":
        return SDMolSupplier(str(file_path), removeHs=False, sanitize=True)
    return SmilesMolSupplier(str(file_path), delimiter="\t ", titleLine=False)


def _convert_ligand_to_sdf(src: Path, output_dir: Path, index: int) -> Path:
    obabel = shutil.which("obabel")
    if obabel is None:
        raise LigandPrepError(
            "Open Babel is required for .mol2/.mol/.pdb ligand input. "
            "Install Open Babel or convert the ligand to SDF/SMILES first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{index:04d}_{src.stem}.sdf"
    try:
        proc = subprocess.run(
            [obabel, str(src), "-O", str(dst)],
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise LigandPrepError(f"Open Babel timed out converting {src}") from exc
    except OSError as exc:
        raise LigandPrepError(f"Open Babel failed to start: {exc}") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        msg = f"Open Babel failed to convert {src.name}"
        if detail:
            msg += f": {detail}"
        raise LigandPrepError(msg)
    if not dst.exists() or dst.stat().st_size == 0:
        raise LigandPrepError(f"Open Babel produced no SDF output for {src.name}")
    return dst


def _normalise_input_files(files: list[Path], output_dir: Path) -> list[Path]:
    converted_dir = output_dir / "converted_inputs"
    normalised: list[Path] = []
    for index, src in enumerate(files):
        suffix = src.suffix.lower()
        if suffix in _RDKIT_INPUT_EXTS:
            normalised.append(src)
        elif suffix in _OPENBABEL_INPUT_EXTS:
            normalised.append(_convert_ligand_to_sdf(src, converted_dir, index))
        else:
            raise LigandPrepError(f"Unsupported file type: {src.suffix}")
    return normalised


def _copy_molecule_properties(source, destination) -> None:
    for name in source.GetPropNames(includePrivate=True, includeComputed=False):
        destination.SetProp(name, source.GetProp(name))


def _is_recognized_counterion(fragment) -> bool:
    if fragment.GetNumAtoms() != 1:
        return False
    atom = fragment.GetAtomWithIdx(0)
    return (atom.GetAtomicNum(), atom.GetFormalCharge()) in (
        _RECOGNIZED_MONATOMIC_COUNTERIONS
    )


def _normalization_record(mol) -> dict[str, Any]:
    from rdkit.Chem import FindPotentialStereo, MolToSmiles

    unspecified = sum(
        1
        for stereo in FindPotentialStereo(mol)
        if str(stereo.specified) == "Unspecified"
    )
    return {
        "status": "unchanged",
        "reason": None,
        "source_smiles": MolToSmiles(mol),
        "normalized_smiles": MolToSmiles(mol),
        "removed_counterions": [],
        "unspecified_stereocenters": unspecified,
        # Normalization does not predict pKa. A later state-selection feature
        # must replace this with an assessed result before the profile is made
        # the production default.
        "protonation_uncertainty": "not_assessed",
    }


def _normalize_ligand(mol, profile: str):
    """Apply target-independent fragment normalization with full provenance."""
    from rdkit.Chem import GetMolFrags, MolToSmiles

    if profile not in _NORMALIZATION_PROFILES:
        raise LigandPrepError(f"Unknown ligand normalization profile: {profile}")

    record = _normalization_record(mol)
    if profile == "legacy":
        record["protonation_uncertainty"] = "not_reported_legacy"
        return mol, record

    fragments = list(GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if len(fragments) == 1:
        return mol, record

    retained = [frag for frag in fragments if not _is_recognized_counterion(frag)]
    removed = [frag for frag in fragments if _is_recognized_counterion(frag)]
    if len(retained) != 1 or len(removed) != len(fragments) - 1:
        record.update(status="rejected", reason="ambiguous_fragments")
        return None, record

    parent = retained[0]
    _copy_molecule_properties(mol, parent)
    removed_smiles = sorted(MolToSmiles(frag) for frag in removed)
    record.update(
        status="normalized",
        normalized_smiles=MolToSmiles(parent),
        removed_counterions=removed_smiles,
    )
    return parent, record


def _scrub(mol, Scrubber, ph: float):
    if Scrubber is None:
        return mol
    try:
        result = Scrubber(pH=ph)(mol)
        return result
    except Exception:
        return None


def _optimise(mol_h, force_field: str, max_iters: int) -> None:
    from rdkit.Chem import AllChem

    ff = (force_field or "MMFF94").upper()
    if ff == "UFF":
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=max_iters)
    else:
        variant = "MMFF94s" if ff == "MMFF94S" else "MMFF94"
        AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant=variant, maxIters=max_iters)


def _embed_3d(mol, mmff_max_iters: int = 0, force_field: str = "MMFF94"):
    from rdkit.Chem import AddHs, AllChem

    mol_h = AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) == -1:
        return None
    _optimise(mol_h, force_field, mmff_max_iters)
    return mol_h


def _to_pdbqt(mol, *, rigid_macrocycles: bool = False) -> str | None:
    # Import outside the try so a broken/missing meeko install raises loudly
    # instead of being charged to every ligand as an "unsupported atoms" failure.
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    try:
        setups = MoleculePreparation(
            rigid_macrocycles=rigid_macrocycles
        ).prepare(mol)
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        return pdbqt if ok else None
    except Exception:
        return None


def _conformer_qc(mol_h) -> str | None:
    """Flag an obviously bad 3D conformer; return a reason string or None.

    Catches the embed/optimise failures that still produce a writable molecule:
    non-finite coordinates, bonded atoms at impossible distances, and internal
    steric clashes between heavy atoms that aren't 1-2/1-3 bonded.
    """
    try:
        conf = mol_h.GetConformer()
    except Exception:
        return "no_conformer"
    n = conf.GetNumAtoms()
    pos = [conf.GetAtomPosition(i) for i in range(n)]

    for p in pos:
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            return "non_finite_coords"

    excluded: set[tuple[int, int]] = set()
    for b in mol_h.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = pos[i].Distance(pos[j])
        if d < 0.5 or d > 3.0:
            return "bad_bond_length"
        excluded.add((min(i, j), max(i, j)))

    # 1-3 pairs (two atoms sharing a neighbour) are not clashes
    for atom in mol_h.GetAtoms():
        nbrs = [nb.GetIdx() for nb in atom.GetNeighbors()]
        for a in range(len(nbrs)):
            for c in range(a + 1, len(nbrs)):
                i, j = nbrs[a], nbrs[c]
                excluded.add((min(i, j), max(i, j)))

    heavy = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
    for a in range(len(heavy)):
        for c in range(a + 1, len(heavy)):
            i, j = heavy[a], heavy[c]
            if (min(i, j), max(i, j)) in excluded:
                continue
            if pos[i].Distance(pos[j]) < 0.9:
                return "steric_clash"
    return None


# Atomic numbers AutoDock4 / UniDock can handle.
# Elements outside this set produce atom-type parse errors at docking time.
_AUTODOCK_SUPPORTED_ATOMIC_NUMS = frozenset(
    {
        1,  # H
        6,  # C
        7,  # N
        8,  # O
        9,  # F
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
    }
)


def _prep_one(
    mol,
    Scrubber,
    ph: float,
    gpu_filter: dict | None = None,
    mmff_max_iters: int = 0,
    force_field: str = "MMFF94",
    rigid_macrocycles: bool = False,
) -> tuple[str | None, str | None, str | None]:
    try:
        from rdkit.Chem import SanitizeMol

        SanitizeMol(mol)
    except Exception:
        return None, "sanitization", None

    # GPU size pre-filter — cheapest check, done before expensive 3-D embedding.
    # UniDock GPU overflows float32 for large/flexible molecules, producing FLT_MAX
    # scores. Thresholds match UniDock paper recommendations (78 HA hard crash limit;
    # we use 70 as a conservative margin).
    if gpu_filter:
        from rdkit.Chem.Descriptors import MolWt
        from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms, CalcNumRotatableBonds

        if CalcNumHeavyAtoms(mol) > gpu_filter["max_heavy_atoms"]:
            return None, "too_large_for_gpu", None
        if MolWt(mol) > gpu_filter["max_mw"]:
            return None, "too_large_for_gpu", None
        if CalcNumRotatableBonds(mol) > gpu_filter["max_rotatable_bonds"]:
            return None, "too_large_for_gpu", None

    # Reject molecules with elements AutoDock/UniDock can't handle.
    # Meeko writes these as bare element symbols (e.g. "B" for Boron) which
    # UniDock rejects at runtime with "Atom type B is not a valid AutoDock type".
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in _AUTODOCK_SUPPORTED_ATOMIC_NUMS:
            return None, "unsupported_atoms", None

    scrubbed = _scrub(mol, Scrubber, ph)
    if scrubbed is None:
        return None, "sanitization", None

    mol_3d = _embed_3d(scrubbed, mmff_max_iters, force_field)
    if mol_3d is None:
        return None, "conformer_generation", None

    qc = _conformer_qc(mol_3d)

    pdbqt = _to_pdbqt(mol_3d, rigid_macrocycles=rigid_macrocycles)
    if pdbqt is None:
        return None, "unsupported_atoms", None

    return pdbqt, None, qc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def shard_raw(
    input_path: Path,
    output_dir: Path,
    shard_size: int = _SHARD_SIZE,
    n_shards: int | None = None,
) -> dict[str, Any]:
    """Create raw ligand shards for Kaggle-side prep.

    SMILES/SDF inputs are converted to SMILES shards locally. Mol2/Mol/PDB
    inputs are copied through unchanged so Kaggle can do the Open Babel
    conversion in the notebook environment.
    """
    from rdkit.Chem import MolToSmiles

    output_dir.mkdir(parents=True, exist_ok=True)
    files = scan_input(input_path)
    rdkit_files = [fp for fp in files if fp.suffix.lower() in _RDKIT_INPUT_EXTS]
    backend_files = [fp for fp in files if fp.suffix.lower() in _OPENBABEL_INPUT_EXTS]

    all_entries: list[tuple[str, str]] = []
    for fp in rdkit_files:
        for mol in _load_supplier(fp):
            if mol is None:
                continue
            smiles = MolToSmiles(mol)
            mol_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            if not mol_name:
                for _prop in (
                    "Catalog_ID",
                    "ID",
                    "Name",
                    "IDNUMBER",
                    "PUBCHEM_COMPOUND_CID",
                ):
                    if mol.HasProp(_prop):
                        mol_name = mol.GetProp(_prop).strip()
                        break
            all_entries.append((smiles, mol_name))

    counted_input = len(all_entries) + len(backend_files)
    if n_shards is not None and n_shards > 0 and all_entries:
        shard_size = max(1, math.ceil(len(all_entries) / n_shards))

    shard_paths: list[Path] = []
    for i in range(0, max(len(all_entries), 1), shard_size):
        chunk = all_entries[i : i + shard_size]
        if not chunk:
            break
        p = output_dir / f"shard_{len(shard_paths):03d}.smi"
        p.write_text("\n".join(f"{smi}\t{name}" for smi, name in chunk))
        shard_paths.append(p)

    for idx, src in enumerate(backend_files):
        dst = output_dir / f"raw_{idx:04d}_{src.name}"
        shutil.copy2(src, dst)
        shard_paths.append(dst)

    return {
        "shard_paths": shard_paths,
        "filtered_gpu_size_csv": None,
        "report": {
            "input_source": str(input_path),
            "input_files": len(files),
            "converted_input_files": 0,
            "backend_input_files": len(backend_files),
            "total_input": counted_input,
            "prep_passed": counted_input,
            "prep_failed": 0,
        },
    }


def prep_ligands(
    input_path: Path,
    output_dir: Path,
    ph: float = 7.4,
    shard_size: int = _SHARD_SIZE,
    n_shards: int | None = None,
    force_field: str | None = None,
    enumerate_opts: dict | None = None,
    normalization_profile: str | None = None,
    rigid_macrocycles: bool | None = None,
) -> dict[str, Any]:
    from rdkit.Chem import SDWriter

    output_dir.mkdir(parents=True, exist_ok=True)
    Scrubber = _get_scrubber()
    files = scan_input(input_path)
    converted_input_files = sum(
        1 for fp in files if fp.suffix.lower() in _OPENBABEL_INPUT_EXTS
    )
    load_files = _normalise_input_files(files, output_dir)

    failed_path = output_dir / "failed_prep.sdf"
    failed_writer = SDWriter(str(failed_path))

    try:
        from ezscreen import config as _cfg

        _pc = _cfg.load().get("prep", {})
        gpu_filter: dict | None = (
            {
                "max_heavy_atoms": int(_pc.get("max_heavy_atoms", 70)),
                "max_mw": float(_pc.get("max_mw", 700.0)),
                "max_rotatable_bonds": int(_pc.get("max_rotatable_bonds", 20)),
            }
            if _pc.get("enable_gpu_size_filter", True)
            else None
        )
        mmff_max_iters: int = int(_pc.get("mmff_max_iters", 0))
        _cfg_force_field = str(_pc.get("force_field", "MMFF94"))
        _cfg_normalization_profile = str(_pc.get("normalization_profile", "legacy"))
        _cfg_rigid_macrocycles = bool(_pc.get("rigid_macrocycles", True))
    except Exception:
        gpu_filter = {"max_heavy_atoms": 70, "max_mw": 700.0, "max_rotatable_bonds": 20}
        mmff_max_iters = 0
        _cfg_force_field = "MMFF94"
        _cfg_normalization_profile = "legacy"
        _cfg_rigid_macrocycles = True
        _pc = {}

    # explicit per-run override wins over the config default
    force_field = force_field or _cfg_force_field
    normalization_profile = normalization_profile or _cfg_normalization_profile
    if rigid_macrocycles is None:
        rigid_macrocycles = _cfg_rigid_macrocycles
    if normalization_profile not in _NORMALIZATION_PROFILES:
        raise LigandPrepError(
            f"Unknown ligand normalization profile: {normalization_profile}"
        )

    # enumeration: per-run override (enumerate_opts) wins over the config defaults
    _eo = (
        enumerate_opts
        if enumerate_opts is not None
        else {
            "enabled": bool(_pc.get("enumerate_enabled", False)),
            "protonation": bool(_pc.get("enumerate_protonation", True)),
            "tautomers": bool(_pc.get("enumerate_tautomers", True)),
            "stereo": bool(_pc.get("enumerate_stereo", False)),
            "ring": bool(_pc.get("enumerate_ring", False)),
            "max_variants": int(_pc.get("max_variants_per_ligand", 4)),
        }
    )
    _enum_on = bool(_eo.get("enabled"))

    failures = {
        "sanitization": 0,
        "ambiguous_fragments": 0,
        "conformer_generation": 0,
        "unsupported_atoms": 0,
        "too_large_for_gpu": 0,
    }
    filtered_too_large: list[dict] = []
    shard_buf: list[str] = []
    shard_paths: list[Path] = []
    index_rows: list[dict] = []  # ligand id → name + smiles
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
    all_mols: list = []
    all_meta: list[tuple[str, str]] = []  # (smiles, mol_name)
    all_normalization: list[dict[str, Any]] = []
    normalization_rejections: list[dict[str, Any]] = []
    bad_mols: list = []  # None-supplier failures to count
    for fp in load_files:
        for source_ordinal, mol in enumerate(_load_supplier(fp), 1):
            if mol is None:
                bad_mols.append(None)
                continue
            total += 1
            source_smiles = MolToSmiles(mol)
            mol_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
            if not mol_name:
                for _prop in (
                    "Catalog_ID",
                    "ID",
                    "Name",
                    "IDNUMBER",
                    "PUBCHEM_COMPOUND_CID",
                ):
                    if mol.HasProp(_prop):
                        mol_name = mol.GetProp(_prop).strip()
                        break
            normalized, normalization = _normalize_ligand(mol, normalization_profile)
            normalization["source_record"] = f"{fp}:{source_ordinal}"
            if normalized is None:
                prep_failed += 1
                failures[normalization["reason"]] += 1
                normalization_rejections.append(
                    {
                        "name": mol_name,
                        "source_smiles": source_smiles,
                        "source_record": normalization["source_record"],
                        "reason": normalization["reason"],
                    }
                )
                failed_writer.write(mol)
                continue
            all_mols.append(normalized)
            all_meta.append((MolToSmiles(normalized), mol_name))
            all_normalization.append(normalization)

    prep_failed += len(bad_mols)
    failures["sanitization"] += len(bad_mols)
    counterions_removed = sum(
        len(record["removed_counterions"]) for record in all_normalization
    )

    # Phase 1b: optional protonation/tautomer/stereo/ring enumeration.
    # Each input SMILES fans out to its variants; the source name gets a _v{k}
    # suffix so variants stay traceable through merge. Fails soft per ligand —
    # enumerate_variants returns [smiles] unchanged when Gypsum-DL is unavailable.
    variants_generated = 0
    if _enum_on and all_mols:
        from rdkit.Chem import MolFromSmiles

        from ezscreen.prep.enumerate import enumerate_variants

        exp_mols: list = []
        exp_meta: list[tuple[str, str]] = []
        exp_normalization: list[dict[str, Any]] = []
        for (smiles, mol_name), mol, normalization in zip(
            all_meta, all_mols, all_normalization
        ):
            variants = enumerate_variants(
                smiles,
                ph,
                int(_eo.get("max_variants", 4)),
                protonation=bool(_eo.get("protonation", True)),
                tautomers=bool(_eo.get("tautomers", True)),
                stereo=bool(_eo.get("stereo", False)),
                ring=bool(_eo.get("ring", False)),
                fail_closed=bool(_eo.get("fail_closed", False)),
            )
            if len(variants) <= 1 and not bool(_eo.get("fail_closed", False)):
                exp_mols.append(mol)
                exp_meta.append((smiles, mol_name))
                exp_normalization.append(normalization)
                continue
            for k, vsmi in enumerate(variants, 1):
                vmol = MolFromSmiles(vsmi)
                if vmol is None:
                    continue
                exp_mols.append(vmol)
                exp_meta.append((vsmi, f"{mol_name}_v{k}" if mol_name else f"v{k}"))
                exp_normalization.append(dict(normalization))
                variants_generated += 1
        all_mols, all_meta, all_normalization = (
            exp_mols,
            exp_meta,
            exp_normalization,
        )

    # Phase 2: parallel conformer generation + PDBQT conversion.
    # RDKit's ETKDGv3 and MMFF are C++ extensions that release the GIL,
    # so ThreadPoolExecutor gives near-linear speedup without pickling.
    n_workers = min(max(os.cpu_count() or 1, 1), 16)
    console.print(
        f"  [dim]Prepping {total} ligand(s) with {n_workers} worker(s)...[/dim]"
    )

    results: list[tuple | None] = [None] * len(all_mols)

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Prepping ligands...", total=len(all_mols))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(
                    _prep_one,
                    mol,
                    Scrubber,
                    ph,
                    gpu_filter,
                    mmff_max_iters,
                    force_field,
                    rigid_macrocycles,
                ): i
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
    qc_flagged: list[dict] = []
    for (smiles, mol_name), mol, normalization, result in zip(
        all_meta, all_mols, all_normalization, results
    ):
        pdbqt, reason, qc = result  # type: ignore[misc]
        if pdbqt:
            lig_id = f"lig_{prep_passed:05d}"
            index_rows.append(
                {
                    "ligand": lig_id,
                    "name": mol_name,
                    "smiles": smiles,
                    "source_smiles": normalization["source_smiles"],
                    "source_record": normalization["source_record"],
                    "normalization_status": normalization["status"],
                    "removed_counterions": ";".join(
                        normalization["removed_counterions"]
                    ),
                    "unspecified_stereocenters": normalization[
                        "unspecified_stereocenters"
                    ],
                    "protonation_uncertainty": normalization["protonation_uncertainty"],
                    "conformer_qc": qc or "",
                }
            )
            shard_buf.append(f"REMARK lig_id {lig_id}\n" + pdbqt)
            if qc:
                qc_flagged.append({"ligand": lig_id, "name": mol_name, "reason": qc})
            prep_passed += 1
            if len(shard_buf) >= shard_size:
                _flush()
        else:
            prep_failed += 1
            failures[reason] += 1
            if reason == "too_large_for_gpu":
                filtered_too_large.append({"name": mol_name, "smiles": smiles})
            else:
                failed_writer.write(mol)
    _flush()

    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ligand",
                "name",
                "smiles",
                "source_smiles",
                "source_record",
                "normalization_status",
                "removed_counterions",
                "unspecified_stereocenters",
                "protonation_uncertainty",
                "conformer_qc",
            ],
        )
        w.writeheader()
        w.writerows(index_rows)

    filtered_path = output_dir / "filtered_gpu_size.csv"
    if filtered_too_large:
        with filtered_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "smiles"])
            w.writeheader()
            w.writerows(filtered_too_large)

    failed_writer.close()

    warnings: list[dict] = []
    if Scrubber is None:
        warnings.append(
            {
                "severity": "medium",
                "category": "scrubber_unavailable",
                "affected_count": total,
                "message": "scrub.py not available — protonation/cleanup skipped",
                "action": "pip install 'ezscreen[scrubber]'",
            }
        )

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
    sv = (
        "vendor"
        if (Scrubber and "vendor" in getattr(Scrubber, "__module__", ""))
        else "git+"
    )
    _enum_protonation = _enum_on and bool(_eo.get("protonation", True))
    if Scrubber is not None:
        protonation_status = "applied_scrubber"
        protonation_method = "Scrubber"
    elif _enum_protonation:
        # Enumeration is deliberately fail-soft unless fail_closed was requested,
        # so a run-level report cannot claim that every ligand was protonated.
        protonation_status = "enumeration_requested_fail_soft"
        protonation_method = "Gypsum-DL"
    else:
        protonation_status = "not_applied_input_state_preserved"
        protonation_method = None

    return {
        "shard_paths": shard_paths,
        "filtered_gpu_size_csv": filtered_path if filtered_too_large else None,
        "report": {
            "input_source": str(input_path),
            "input_files": len(files),
            "converted_input_files": converted_input_files,
            "total_input": total,
            "prep_passed": prep_passed,
            "prep_failed": prep_failed,
            "prep_failures": failures,
            "failed_prep_file": str(failed_path) if prep_failed else None,
            "filtered_gpu_size": len(filtered_too_large),
            "enumeration_enabled": _enum_on,
            "variants_generated": variants_generated,
            "conformer_qc_flagged": len(qc_flagged),
            "conformer_qc_list": qc_flagged,
            "protonation_ph": ph,
            "protonation_status": protonation_status,
            "protonation_method": protonation_method,
            "normalization_profile": normalization_profile,
            "counterions_removed": counterions_removed,
            "normalization_rejected": len(normalization_rejections),
            "normalization_rejections": normalization_rejections,
            "force_field": force_field,
            "macrocycle_policy": "rigid" if rigid_macrocycles else "flexible",
            "rigid_macrocycles": rigid_macrocycles,
            "tools": {"scrubber": sv, "rdkit": rv, "meeko": mv},
            "warnings": warnings,
        },
    }
