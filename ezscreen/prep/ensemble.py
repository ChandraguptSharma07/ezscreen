from __future__ import annotations

import hashlib
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



_AUTO_QC_POCKET_RADIUS = 10.0
_AUTO_QC_MIN_POCKET_CA_RMSD = 1.0
_AUTO_QC_MAX_POCKET_CA_RMSD = 3.0


def _parse_ca_atoms(pdb_path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            atoms.append(
                {
                    "chain": line[21:22].strip(),
                    "resseq": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            )
        except (ValueError, IndexError):
            continue
    return atoms


def _site_center_from_details(
    pdb_path: Path,
    chains: list[str] | None,
    site_details: dict[str, Any],
) -> tuple[float, float, float] | None:
    site_type = str(site_details.get("type") or "").lower()
    if site_type == "cocrystal":
        ligands = site_details.get("ligands") or []
        if ligands:
            centroid = ligands[0].get("centroid")
            if centroid and len(centroid) >= 3:
                return (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    if site_type == "p2rank":
        pocket = site_details.get("pocket") or {}
        center = pocket.get("center")
        if center and len(center) >= 3:
            return (float(center[0]), float(center[1]), float(center[2]))
    if site_type == "manual":
        center = site_details.get("center")
        if center and len(center) >= 3:
            return (float(center[0]), float(center[1]), float(center[2]))
    if site_type == "residues":
        residues = set(int(r) for r in (site_details.get("residues") or []))
        selected_chains = set(chains or [])
        coords: list[tuple[float, float, float]] = []
        for atom in _parse_ca_atoms(pdb_path):
            if residues and atom["resseq"] not in residues:
                continue
            if selected_chains and atom["chain"] not in selected_chains:
                continue
            coords.append((atom["x"], atom["y"], atom["z"]))
        if coords:
            n = len(coords)
            return (
                sum(c[0] for c in coords) / n,
                sum(c[1] for c in coords) / n,
                sum(c[2] for c in coords) / n,
            )
    return None


def _pocket_ca_rmsd(
    reference_pdb: Path,
    conformer_pdb: Path,
    center: tuple[float, float, float],
    radius: float = _AUTO_QC_POCKET_RADIUS,
) -> tuple[float | None, int]:
    ref_atoms = {
        (a["chain"], a["resseq"]): (a["x"], a["y"], a["z"])
        for a in _parse_ca_atoms(reference_pdb)
    }
    sq_diffs: list[float] = []
    matched = 0
    for atom in _parse_ca_atoms(conformer_pdb):
        key = (atom["chain"], atom["resseq"])
        ref = ref_atoms.get(key)
        if ref is None:
            continue
        dx0 = ref[0] - center[0]
        dy0 = ref[1] - center[1]
        dz0 = ref[2] - center[2]
        if dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > radius * radius:
            continue
        matched += 1
        sq_diffs.append(
            (ref[0] - atom["x"]) ** 2
            + (ref[1] - atom["y"]) ** 2
            + (ref[2] - atom["z"]) ** 2
        )
    if not sq_diffs:
        return None, matched
    return math.sqrt(sum(sq_diffs) / len(sq_diffs)), matched


def _passes_auto_qc(
    reference_pdb: Path,
    conformer_pdb: Path,
    chains: list[str] | None,
    site_details: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    center = _site_center_from_details(reference_pdb, chains, site_details)
    if center is None:
        return True, {"qc_status": "skipped", "qc_reason": "no_site_center"}
    rmsd, matched = _pocket_ca_rmsd(reference_pdb, conformer_pdb, center)
    if rmsd is None or matched < 4:
        return True, {
            "qc_status": "skipped",
            "qc_reason": "insufficient_pocket_atoms",
            "qc_matched_ca": matched,
        }
    meta = {
        "qc_status": "accepted",
        "qc_reason": "pocket_ca_rmsd",
        "qc_matched_ca": matched,
        "pocket_ca_rmsd": round(rmsd, 3),
    }
    if rmsd < _AUTO_QC_MIN_POCKET_CA_RMSD:
        meta["qc_status"] = "rejected"
        meta["qc_reason"] = "near_duplicate"
        return False, meta
    if rmsd > _AUTO_QC_MAX_POCKET_CA_RMSD:
        meta["qc_status"] = "rejected"
        meta["qc_reason"] = "pocket_distortion"
        return False, meta
    return True, meta

@dataclass(frozen=True)
class EnsembleConformer:
    name: str
    source: str
    pdb_path: Path | None
    pdbqt_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or fallback


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe_conformers(conformers: list[EnsembleConformer]) -> list[EnsembleConformer]:
    unique: list[EnsembleConformer] = []
    seen: set[str] = set()
    for conformer in conformers:
        digest = _hash_file(conformer.pdbqt_path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(
            EnsembleConformer(
                name=conformer.name,
                source=conformer.source,
                pdb_path=conformer.pdb_path,
                pdbqt_path=conformer.pdbqt_path,
                metadata={**conformer.metadata, "pdbqt_sha256": digest},
            )
        )
    return unique


def _prepare_one(
    src: Path,
    *,
    idx: int,
    output_dir: Path,
    chains: list[str] | None,
    ph: float,
    keep_hetero: list[dict] | None,
    source: str,
    rec_prep_module: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> EnsembleConformer:
    from ezscreen.prep import receptor as default_rec_prep

    rec_prep = rec_prep_module or default_rec_prep
    src = Path(src)
    stem = _safe_name(src.stem, f"conformer_{idx:03d}")
    name = f"conformer_{idx:03d}_{stem}"
    out_pdbqt = output_dir / f"receptor_{name}.pdbqt"
    pdb_path: Path | None = src if src.suffix.lower() == ".pdb" else None

    if src.suffix.lower() == ".pdbqt":
        shutil.copy2(src, out_pdbqt)
    elif src.suffix.lower() == ".pdb":
        rec_result = rec_prep.prep_receptor(
            pdb_path=src,
            chains=chains,
            output_dir=output_dir / f"prep_{idx:03d}_{stem}",
            ph=ph,
            keep_hetero=keep_hetero,
            receptor_isomers="off",
        )
        shutil.copy2(rec_result["pdbqt_path"], out_pdbqt)
        pdb_path = Path(
            rec_result.get("fixed_pdb_path")
            or rec_result.get("receptor_pdb_path")
            or src
        )
    else:
        raise ValueError(f"Unsupported ensemble receptor format: {src.suffix}")

    return EnsembleConformer(
        name=name,
        source=source,
        pdb_path=pdb_path,
        pdbqt_path=out_pdbqt,
        metadata=metadata or {},
    )


def prepare_manual_ensemble(
    paths: list[Path],
    *,
    output_dir: Path,
    chains: list[str] | None = None,
    ph: float = 7.4,
    keep_hetero: list[dict] | None = None,
    rec_prep_module: Any | None = None,
) -> list[EnsembleConformer]:
    """Prepare user-supplied receptor conformers as PDBQT files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conformers: list[EnsembleConformer] = []
    for idx, raw_path in enumerate(paths, 1):
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Ensemble receptor not found: {path}")
        if path.suffix.lower() not in {".pdb", ".pdbqt"}:
            raise ValueError(f"Unsupported ensemble receptor format: {path.suffix}")
        conformers.append(
            _prepare_one(
                path,
                idx=idx,
                output_dir=output_dir,
                chains=chains,
                ph=ph,
                keep_hetero=keep_hetero,
                source="manual_pdbqt" if path.suffix.lower() == ".pdbqt" else "manual_pdb",
                rec_prep_module=rec_prep_module,
                metadata={"input_path": str(path)},
            )
        )
    return _dedupe_conformers(conformers)


def _pdb_derived_candidates(
    pdb_path: Path,
    chains: list[str],
    site_details: dict[str, Any],
    output_dir: Path,
    max_conformers: int,
) -> list[Path]:
    """Return aligned same-target/homolog PDB conformers.

    The default implementation is deliberately conservative: when a four-character
    PDB id can be inferred from the input filename, it fetches that structure from
    RCSB as a same-target candidate. Downstream integrations can replace this hook
    with a richer sequence-search, alignment, and pocket-RMSD clustering provider.
    """
    pdb_id = None
    stem = Path(pdb_path).stem
    if len(stem) == 4 and stem.isalnum():
        pdb_id = stem.upper()
    if not pdb_id:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{pdb_id}.pdb"
    if out.exists():
        return [out]
    try:
        import requests

        resp = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            timeout=20,
        )
        resp.raise_for_status()
        out.write_text(resp.text, encoding="utf-8")
        return [out]
    except Exception:
        return []


def _nma_fallback_candidates(
    pdb_path: Path,
    chains: list[str],
    site_details: dict[str, Any],
    output_dir: Path,
    max_conformers: int,
) -> list[Path]:
    """Return low-frequency normal-mode conformers from the input receptor.

    Uses ProDy when available. If optional NMA tooling is absent or fails, emits one
    marked copy of the input receptor so the auto path remains operational.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import prody  # type: ignore

        atoms = prody.parsePDB(str(pdb_path))
        selection = atoms.select("protein and name CA")
        if selection is None or len(selection) < 3:
            raise ValueError("not enough CA atoms for ANM")
        anm = prody.ANM("ezscreen_auto_ensemble")
        anm.buildHessian(selection)
        anm.calcModes(n_modes=max(1, min(3, max_conformers)))
        candidates: list[Path] = []
        for idx, mode in enumerate(anm[:max_conformers], 1):
            ensemble = prody.traverseMode(mode, selection, rmsd=1.0, n_steps=1)
            coords = ensemble.getCoordsets()[-1]
            moved = atoms.copy()
            moved.select("protein and name CA").setCoords(coords)
            out = output_dir / f"nma_conformer_{idx:03d}.pdb"
            prody.writePDB(str(out), moved)
            candidates.append(out)
        if candidates:
            return candidates
    except Exception:
        pass

    out = output_dir / "nma_conformer_001.pdb"
    text = Path(pdb_path).read_text(encoding="utf-8", errors="replace")
    out.write_text(
        "REMARK EZSCREEN AUTO ENSEMBLE NMA FALLBACK\n" + text,
        encoding="utf-8",
    )
    return [out]


def generate_auto_ensemble(
    pdb_path: Path,
    chains: list[str],
    site_details: dict[str, Any],
    *,
    max_conformers: int = 3,
    output_dir: Path,
    ph: float = 7.4,
    keep_hetero: list[dict] | None = None,
    rec_prep_module: Any | None = None,
) -> list[EnsembleConformer]:
    """Generate receptor conformers using PDB-derived structures, then NMA fallback."""
    from ezscreen.errors import ReceptorPrepError

    if max_conformers < 1:
        raise ValueError("max_conformers must be at least 1")
    max_conformers = min(int(max_conformers), 8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_dir = output_dir / "candidates"
    candidate_sources = [
        (
            "pdb",
            lambda: _pdb_derived_candidates(
                Path(pdb_path),
                chains,
                site_details,
                candidate_dir / "pdb",
                max_conformers,
            ),
        ),
        (
            "nma",
            lambda: _nma_fallback_candidates(
                Path(pdb_path),
                chains,
                site_details,
                candidate_dir / "nma",
                max_conformers,
            ),
        ),
    ]

    for source, provider in candidate_sources:
        candidates = list(provider()[:max_conformers])
        if not candidates:
            continue
        conformers: list[EnsembleConformer] = []
        for idx, candidate in enumerate(candidates, 1):
            try:
                conformer = _prepare_one(
                    Path(candidate),
                    idx=idx,
                    output_dir=output_dir,
                    chains=chains,
                    ph=ph,
                    keep_hetero=keep_hetero,
                    source=f"auto_{source}",
                    rec_prep_module=rec_prep_module,
                    metadata={
                        "method": "pdb_then_nma",
                        "candidate_path": str(candidate),
                        "fallback_used": source == "nma",
                    },
                )
                passes_qc, qc_meta = _passes_auto_qc(
                    Path(pdb_path),
                    conformer.pdb_path or Path(candidate),
                    chains,
                    site_details,
                )
                if not passes_qc:
                    continue
                conformers.append(
                    EnsembleConformer(
                        name=conformer.name,
                        source=conformer.source,
                        pdb_path=conformer.pdb_path,
                        pdbqt_path=conformer.pdbqt_path,
                        metadata={**conformer.metadata, **qc_meta},
                    )
                )
            except ReceptorPrepError:
                continue
        deduped = _dedupe_conformers(conformers)
        if deduped:
            return deduped

    return []
