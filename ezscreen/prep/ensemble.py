from __future__ import annotations

import hashlib
import math
import re
import shutil
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

_AUTO_QC_POCKET_RADIUS = 10.0
_AUTO_QC_MIN_POCKET_CA_RMSD = 1.0
_AUTO_QC_MAX_POCKET_CA_RMSD = 3.0
_AUTO_PDB_SEARCH_ROWS = 24
_AUTO_PDB_MIN_ALIGNMENT_RESIDUES = 8
_AUTO_PDB_MIN_DIVERSITY_RMSD = 0.35
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


@dataclass(frozen=True)
class EnsembleConformer:
    name: str
    source: str
    pdb_path: Path | None
    pdbqt_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AlignedCandidate:
    path: Path
    source: str
    pocket_ca_rmsd: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


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
                    "resname": line[17:20].strip().upper(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            )
        except (ValueError, IndexError):
            continue
    return atoms


def _parse_ca_residues(
    pdb_path: Path, chain_id: str | None = None
) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for atom in _parse_ca_atoms(pdb_path):
        if chain_id and atom["chain"] != chain_id:
            continue
        key = (atom["chain"], atom["resseq"])
        if key in seen:
            continue
        seen.add(key)
        residues.append(atom)
    return residues


def _chain_sequences(pdb_path: Path) -> dict[str, str]:
    by_chain: dict[str, list[str]] = {}
    for residue in _parse_ca_atoms(pdb_path):
        chain = residue["chain"] or "_"
        by_chain.setdefault(chain, []).append(
            _THREE_TO_ONE.get(residue["resname"], "X")
        )
    return {chain: "".join(seq) for chain, seq in by_chain.items() if seq}


def _parse_protein_atoms(pdb_path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
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


def _distance_sq(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _site_reference_points(
    pdb_path: Path,
    chains: list[str] | None,
    site_details: dict[str, Any],
) -> list[tuple[float, float, float]]:
    site_type = str(site_details.get("type") or "").lower()
    if site_type == "cocrystal":
        ligands = site_details.get("ligands") or []
        if ligands:
            coords = ligands[0].get("coords") or []
            return [tuple(map(float, xyz[:3])) for xyz in coords if len(xyz) >= 3]
    if site_type == "residues":
        residues = set(int(r) for r in (site_details.get("residues") or []))
        selected_chains = set(chains or [])
        points: list[tuple[float, float, float]] = []
        for atom in _parse_ca_atoms(pdb_path):
            if residues and atom["resseq"] not in residues:
                continue
            if selected_chains and atom["chain"] not in selected_chains:
                continue
            points.append((atom["x"], atom["y"], atom["z"]))
        return points
    center = _site_center_from_details(pdb_path, chains, site_details)
    return [center] if center is not None else []


def _infer_pocket_chains(
    pdb_path: Path,
    chains: list[str],
    site_details: dict[str, Any],
    *,
    contact_radius: float = 8.0,
) -> list[str]:
    if not chains:
        return []
    points = _site_reference_points(pdb_path, chains, site_details)
    if not points:
        return list(chains)
    selected = set(chains)
    radius_sq = contact_radius * contact_radius
    hits: dict[str, int] = {chain: 0 for chain in chains}
    for atom in _parse_protein_atoms(pdb_path):
        chain = atom["chain"]
        if chain not in selected:
            continue
        point = (atom["x"], atom["y"], atom["z"])
        if any(_distance_sq(point, site_point) <= radius_sq for site_point in points):
            hits[chain] = hits.get(chain, 0) + 1
    pocket_chains = [chain for chain in chains if hits.get(chain, 0) > 0]
    return pocket_chains or list(chains)


def _anchor_query_chain(
    pdb_path: Path,
    pocket_chains: list[str],
    site_details: dict[str, Any],
) -> str | None:
    if not pocket_chains:
        return None
    site_type = str(site_details.get("type") or "").lower()
    if site_type == "cocrystal":
        ligands = site_details.get("ligands") or []
        if ligands:
            lig_chain = str(ligands[0].get("chain") or "").strip()
            if lig_chain in pocket_chains:
                return lig_chain
    points = _site_reference_points(pdb_path, pocket_chains, site_details)
    if not points:
        return pocket_chains[0]
    counts: dict[str, int] = {chain: 0 for chain in pocket_chains}
    radius_sq = 8.0 * 8.0
    for atom in _parse_protein_atoms(pdb_path):
        chain = atom["chain"]
        if chain not in counts:
            continue
        point = (atom["x"], atom["y"], atom["z"])
        if any(_distance_sq(point, site_point) <= radius_sq for site_point in points):
            counts[chain] += 1
    return max(
        pocket_chains,
        key=lambda chain: (counts.get(chain, 0), -pocket_chains.index(chain)),
    )


def _site_center_from_details(
    pdb_path: Path, chains: list[str] | None, site_details: dict[str, Any]
) -> tuple[float, float, float] | None:
    site_type = str(site_details.get("type") or "").lower()
    if site_type == "cocrystal":
        ligands = site_details.get("ligands") or []
        if ligands:
            centroid = ligands[0].get("centroid")
            if centroid and len(centroid) >= 3:
                return (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    if site_type == "p2rank":
        center = (site_details.get("pocket") or {}).get("center")
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
    radius_sq = radius * radius
    reference_atoms = _parse_ca_atoms(reference_pdb)
    conformer_atoms = _parse_ca_atoms(conformer_pdb)

    def _collect_rmsd(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> tuple[float | None, int]:
        sq_diffs: list[float] = []
        matched = 0
        for ref_atom, conf_atom in pairs:
            dx0 = ref_atom["x"] - center[0]
            dy0 = ref_atom["y"] - center[1]
            dz0 = ref_atom["z"] - center[2]
            if dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > radius_sq:
                continue
            matched += 1
            sq_diffs.append(
                (ref_atom["x"] - conf_atom["x"]) ** 2
                + (ref_atom["y"] - conf_atom["y"]) ** 2
                + (ref_atom["z"] - conf_atom["z"]) ** 2
            )
        if not sq_diffs:
            return None, matched
        return math.sqrt(sum(sq_diffs) / len(sq_diffs)), matched

    reference_by_key = {(a["chain"], a["resseq"]): a for a in reference_atoms}
    exact_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for atom in conformer_atoms:
        ref_atom = reference_by_key.get((atom["chain"], atom["resseq"]))
        if ref_atom is None or ref_atom.get("resname") != atom.get("resname"):
            continue
        exact_pairs.append((ref_atom, atom))
    rmsd, matched = _collect_rmsd(exact_pairs)
    if rmsd is not None or matched >= 4:
        return rmsd, matched

    reference_by_chain: dict[str, list[dict[str, Any]]] = {}
    for atom in reference_atoms:
        reference_by_chain.setdefault(atom["chain"] or "_", []).append(atom)
    conformer_by_chain: dict[str, list[dict[str, Any]]] = {}
    for atom in conformer_atoms:
        conformer_by_chain.setdefault(atom["chain"] or "_", []).append(atom)

    sequence_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    shared_chains = sorted(set(reference_by_chain) & set(conformer_by_chain))
    if shared_chains:
        for chain_id in shared_chains:
            sequence_pairs.extend(
                _shared_residue_pairs(
                    reference_by_chain[chain_id], conformer_by_chain[chain_id]
                )
            )
    elif len(reference_by_chain) == 1 and len(conformer_by_chain) == 1:
        sequence_pairs = _shared_residue_pairs(
            next(iter(reference_by_chain.values())),
            next(iter(conformer_by_chain.values())),
        )
    return _collect_rmsd(sequence_pairs)


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


def _emit(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or fallback


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sort_auto_conformers(
    conformers: list[EnsembleConformer],
) -> list[EnsembleConformer]:
    def _key(
        conformer: EnsembleConformer,
    ) -> tuple[float, ...] | tuple[int, int, float, str]:
        meta = conformer.metadata or {}
        qc_status = str(meta.get("qc_status") or "")
        status_rank = (
            0 if qc_status == "accepted" else 1 if qc_status == "skipped" else 2
        )
        matched = int(meta.get("qc_matched_ca") or 0)
        rmsd = meta.get("pocket_ca_rmsd")
        rmsd_penalty = abs(float(rmsd) - 1.8) if isinstance(rmsd, (int, float)) else 9.9
        return (status_rank, -matched, rmsd_penalty, conformer.name)

    return sorted(conformers, key=_key)


def _dedupe_conformers(
    conformers: list[EnsembleConformer],
    *,
    logger: Callable[[str], None] | None = None,
) -> list[EnsembleConformer]:
    unique: list[EnsembleConformer] = []
    seen: set[str] = set()
    for conformer in conformers:
        digest = _hash_file(conformer.pdbqt_path)
        if digest in seen:
            _emit(
                logger, f"ensemble: dedupe dropped {conformer.name} (same PDBQT hash)"
            )
            continue
        seen.add(digest)
        unique.append(
            EnsembleConformer(
                conformer.name,
                conformer.source,
                conformer.pdb_path,
                conformer.pdbqt_path,
                {**conformer.metadata, "pdbqt_sha256": digest},
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
    prep_chains: list[str] | None = None,
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
            chains=prep_chains if prep_chains is not None else chains,
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
    return EnsembleConformer(name, source, pdb_path, out_pdbqt, metadata or {})


def _prepare_qc_reference(
    pdb_path: Path,
    *,
    output_dir: Path,
    chains: list[str] | None,
    ph: float,
    keep_hetero: list[dict] | None,
    rec_prep_module: Any | None = None,
) -> Path:
    from ezscreen.prep import receptor as default_rec_prep

    rec_prep = rec_prep_module or default_rec_prep
    rec_result = rec_prep.prep_receptor(
        pdb_path=Path(pdb_path),
        chains=chains,
        output_dir=Path(output_dir),
        ph=ph,
        keep_hetero=keep_hetero,
        receptor_isomers="off",
    )
    return Path(
        rec_result.get("fixed_pdb_path")
        or rec_result.get("receptor_pdb_path")
        or pdb_path
    )


def prepare_manual_ensemble(
    paths: list[Path],
    *,
    output_dir: Path,
    chains: list[str] | None = None,
    ph: float = 7.4,
    keep_hetero: list[dict] | None = None,
    rec_prep_module: Any | None = None,
    logger: Callable[[str], None] | None = None,
) -> list[EnsembleConformer]:
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
                source=(
                    "manual_pdbqt" if path.suffix.lower() == ".pdbqt" else "manual_pdb"
                ),
                rec_prep_module=rec_prep_module,
                metadata={"input_path": str(path)},
            )
        )
    return _dedupe_conformers(conformers, logger=logger)


def _infer_query_pdb_id(pdb_path: Path) -> str | None:
    stem = Path(pdb_path).stem
    if len(stem) == 4 and stem.isalnum():
        return stem.upper()
    for line in Path(pdb_path).read_text(errors="ignore").splitlines()[:50]:
        if line.startswith("HEADER") and len(line) >= 66:
            code = line[62:66].strip().upper()
            if len(code) == 4 and code.isalnum():
                return code
    return None


def _infer_query_sequence(pdb_path: Path, chains: list[str]) -> str:
    if not chains:
        return ""
    sequences = _chain_sequences(pdb_path)
    parts = [sequences.get(chain, "") for chain in chains if sequences.get(chain)]
    return "".join(parts) if len(parts) == 1 else (parts[0] if parts else "")


def _extract_additional_models(
    pdb_path: Path, output_dir: Path, max_conformers: int
) -> list[Path]:
    text = Path(pdb_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    models: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("MODEL"):
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                models.append(current)
            current = None
            continue
        if current is not None:
            current.append(line)
    if len(models) <= 1:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    for idx, model_lines in enumerate(models[1 : 1 + max_conformers], 2):
        out = output_dir / f"input_model_{idx:03d}.pdb"
        out.write_text("\n".join(model_lines) + "\n", encoding="utf-8")
        candidates.append(out)
    return candidates


def _search_rcsb_sequence_candidates(
    sequence: str,
    *,
    exclude_ids: set[str] | None = None,
    rows: int = _AUTO_PDB_SEARCH_ROWS,
) -> list[str]:
    if len(sequence) < 20:
        return []
    exclude = {e.upper() for e in (exclude_ids or set())}
    try:
        import requests

        payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1.0,
                    "identity_cutoff": 0.9,
                    "target": "pdb_protein_sequence",
                    "value": sequence,
                },
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": rows},
                "results_verbosity": "compact",
            },
        }
        resp = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query", json=payload, timeout=20
        )
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        return []
    hits: list[str] = []
    for item in data.get("result_set") or []:
        if isinstance(item, dict):
            identifier = str(item.get("identifier") or "").upper()
        else:
            identifier = str(item or "").upper()
        if len(identifier) != 4 or identifier in exclude or identifier in hits:
            continue
        hits.append(identifier)
    return hits


def _download_rcsb_pdb(entry_id: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{entry_id.upper()}.pdb"
    if out.exists():
        return out
    try:
        import requests

        resp = requests.get(
            f"https://files.rcsb.org/download/{entry_id.upper()}.pdb", timeout=20
        )
        resp.raise_for_status()
        out.write_text(resp.text, encoding="utf-8")
        return out
    except Exception:
        return None


def _shared_residue_pairs(
    reference_residues: list[dict[str, Any]], candidate_residues: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    reference_codes = [
        _THREE_TO_ONE.get(str(residue.get("resname") or "").upper(), "X")
        for residue in reference_residues
    ]
    candidate_codes = [
        _THREE_TO_ONE.get(str(residue.get("resname") or "").upper(), "X")
        for residue in candidate_residues
    ]
    matcher = SequenceMatcher(a=reference_codes, b=candidate_codes, autojunk=False)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ref_start, cand_start, size in matcher.get_matching_blocks():
        if size <= 0:
            continue
        for offset in range(size):
            pairs.append(
                (
                    reference_residues[ref_start + offset],
                    candidate_residues[cand_start + offset],
                )
            )
    return pairs


def _best_candidate_chain(
    reference_pdb: Path, reference_chain: str, candidate_pdb: Path
) -> tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]] | None:
    reference_residues = _parse_ca_residues(reference_pdb, reference_chain)
    if not reference_residues:
        return None
    chain_ids = sorted(
        {atom["chain"] for atom in _parse_ca_atoms(candidate_pdb) if atom["chain"]}
    )
    best: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]] | None = None
    for chain_id in chain_ids:
        pairs = _shared_residue_pairs(
            reference_residues, _parse_ca_residues(candidate_pdb, chain_id)
        )
        if len(pairs) < _AUTO_PDB_MIN_ALIGNMENT_RESIDUES:
            continue
        if best is None or len(pairs) > len(best[1]):
            best = (chain_id, pairs)
    return best


def _apply_rigid_transform(
    point: tuple[float, float, float],
    rotation: list[list[float]],
    mobile_center: tuple[float, float, float],
    target_center: tuple[float, float, float],
) -> tuple[float, float, float]:
    shifted = (
        point[0] - mobile_center[0],
        point[1] - mobile_center[1],
        point[2] - mobile_center[2],
    )
    rotated = (
        rotation[0][0] * shifted[0]
        + rotation[0][1] * shifted[1]
        + rotation[0][2] * shifted[2],
        rotation[1][0] * shifted[0]
        + rotation[1][1] * shifted[1]
        + rotation[1][2] * shifted[2],
        rotation[2][0] * shifted[0]
        + rotation[2][1] * shifted[1]
        + rotation[2][2] * shifted[2],
    )
    return (
        rotated[0] + target_center[0],
        rotated[1] + target_center[1],
        rotated[2] + target_center[2],
    )


def _kabsch_rotation(
    mobile_points: list[tuple[float, float, float]],
    target_points: list[tuple[float, float, float]],
) -> (
    tuple[list[list[float]], tuple[float, float, float], tuple[float, float, float]]
    | None
):
    if len(mobile_points) != len(target_points) or len(mobile_points) < 3:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    mobile = np.array(mobile_points, dtype=float)
    target = np.array(target_points, dtype=float)
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _s, vt = np.linalg.svd(covariance)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    return (
        rot.tolist(),
        (float(mobile_center[0]), float(mobile_center[1]), float(mobile_center[2])),
        (float(target_center[0]), float(target_center[1]), float(target_center[2])),
    )


def _write_aligned_chain_pdb(
    candidate_pdb: Path,
    candidate_chain: str,
    reference_chain: str,
    output_pdb: Path,
    rotation: list[list[float]],
    mobile_center: tuple[float, float, float],
    target_center: tuple[float, float, float],
) -> None:
    out_lines: list[str] = []
    for raw_line in Path(candidate_pdb).read_text(errors="ignore").splitlines():
        if raw_line.startswith(("ATOM", "HETATM")):
            chain = raw_line[21:22].strip()
            if chain != candidate_chain:
                continue
            try:
                point = (
                    float(raw_line[30:38]),
                    float(raw_line[38:46]),
                    float(raw_line[46:54]),
                )
            except ValueError:
                continue
            x, y, z = _apply_rigid_transform(
                point, rotation, mobile_center, target_center
            )
            line = list(raw_line.ljust(80))
            line[21] = reference_chain[:1] if reference_chain else "A"
            line[30:38] = list(f"{x:8.3f}")
            line[38:46] = list(f"{y:8.3f}")
            line[46:54] = list(f"{z:8.3f}")
            out_lines.append("".join(line).rstrip())
    out_lines.extend(["TER", "END"])
    output_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _apply_ca_displacements_to_all_atoms(
    atoms: Any,
    ca_selection: Any,
    target_ca_coords: Any,
) -> Any:
    moved = atoms.copy()
    moved_coords = moved.getCoords()
    ca_coords = ca_selection.getCoords()
    deltas_by_resindex: dict[int, Any] = {}
    for resindex, source_coord, target_coord in zip(
        ca_selection.getResindices(), ca_coords, target_ca_coords
    ):
        deltas_by_resindex[int(resindex)] = target_coord - source_coord
    for atom_index, resindex in enumerate(moved.getResindices()):
        delta = deltas_by_resindex.get(int(resindex))
        if delta is None:
            continue
        moved_coords[atom_index] = moved_coords[atom_index] + delta
    moved.setCoords(moved_coords)
    return moved


def _align_pdb_candidate(
    reference_pdb: Path,
    candidate_pdb: Path,
    chains: list[str],
    output_dir: Path,
    site_details: dict[str, Any],
) -> _AlignedCandidate | None:
    if not chains:
        return None
    reference_chain = chains[0]
    best = _best_candidate_chain(reference_pdb, reference_chain, candidate_pdb)
    if best is None:
        return None
    candidate_chain, residue_pairs = best
    rotation_info = _kabsch_rotation(
        [
            (candidate["x"], candidate["y"], candidate["z"])
            for _, candidate in residue_pairs
        ],
        [
            (reference["x"], reference["y"], reference["z"])
            for reference, _ in residue_pairs
        ],
    )
    if rotation_info is None:
        return None
    rotation, mobile_center, target_center = rotation_info
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"aligned_{candidate_pdb.stem}_{candidate_chain}.pdb"
    _write_aligned_chain_pdb(
        candidate_pdb,
        candidate_chain,
        reference_chain,
        out,
        rotation,
        mobile_center,
        target_center,
    )
    center = _site_center_from_details(reference_pdb, chains, site_details)
    pocket_rmsd = (
        None if center is None else _pocket_ca_rmsd(reference_pdb, out, center)[0]
    )
    return _AlignedCandidate(
        out,
        "remote_pdb",
        pocket_rmsd,
        {
            "candidate_chain": candidate_chain,
            "reference_chain": reference_chain,
            "alignment_residues": len(residue_pairs),
        },
    )


def _select_diverse_aligned_candidates(
    reference_pdb: Path,
    aligned_candidates: list[_AlignedCandidate],
    center: tuple[float, float, float] | None,
    max_conformers: int,
) -> list[Path]:
    if not aligned_candidates:
        return []
    viable_min = _AUTO_QC_MIN_POCKET_CA_RMSD
    viable_max = _AUTO_QC_MAX_POCKET_CA_RMSD

    def _viability_rank(item: _AlignedCandidate) -> int:
        rmsd = item.pocket_ca_rmsd
        if rmsd is None:
            return 2
        if rmsd < viable_min:
            return 1
        if rmsd > viable_max:
            return 3
        return 0

    ranked = sorted(
        aligned_candidates,
        key=lambda item: (
            _viability_rank(item),
            -int(item.metadata.get("alignment_residues") or 0),
            -(item.pocket_ca_rmsd or 0.0),
            item.path.name,
        ),
    )
    selected: list[_AlignedCandidate] = []
    for candidate in ranked:
        if len(selected) >= max_conformers:
            break
        if center is None:
            selected.append(candidate)
            continue
        too_close = False
        for prior in selected:
            pair_rmsd, matched = _pocket_ca_rmsd(prior.path, candidate.path, center)
            if (
                pair_rmsd is not None
                and matched >= 4
                and pair_rmsd < _AUTO_PDB_MIN_DIVERSITY_RMSD
            ):
                too_close = True
                break
        if not too_close:
            selected.append(candidate)
    return [item.path for item in selected]


def _pdb_derived_candidates(
    pdb_path: Path,
    chains: list[str],
    site_details: dict[str, Any],
    output_dir: Path,
    max_conformers: int,
    *,
    logger: Callable[[str], None] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    model_candidates = _extract_additional_models(
        pdb_path, output_dir / "models", max_conformers
    )
    candidates.extend(model_candidates)
    if model_candidates:
        _emit(
            logger,
            f"ensemble: found {len(model_candidates)} extra MODEL conformer(s) in input PDB",
        )
    remaining = max_conformers - len(candidates)
    if remaining <= 0:
        return candidates[:max_conformers]
    pocket_chains = _infer_pocket_chains(pdb_path, chains, site_details)
    _emit(
        logger,
        f"ensemble: pocket_chains={','.join(pocket_chains) if pocket_chains else 'none'}",
    )
    if len(pocket_chains) != 1:
        _emit(
            logger,
            "ensemble: skipped remote PDB harvesting (interface-spanning or ambiguous pocket chains)",
        )
        return candidates[:max_conformers]
    anchor_chain = _anchor_query_chain(pdb_path, pocket_chains, site_details)
    if not anchor_chain:
        _emit(logger, "ensemble: skipped remote PDB harvesting (no anchor chain)")
        return candidates[:max_conformers]
    sequence = _infer_query_sequence(pdb_path, [anchor_chain])
    if not sequence:
        _emit(
            logger,
            f"ensemble: skipped remote PDB harvesting (empty sequence for chain {anchor_chain})",
        )
        return candidates[:max_conformers]
    exclude_ids = set()
    query_id = _infer_query_pdb_id(pdb_path)
    if query_id:
        exclude_ids.add(query_id)
    hit_ids = _search_rcsb_sequence_candidates(sequence, exclude_ids=exclude_ids)
    _emit(
        logger,
        f"ensemble: remote PDB search on chain {anchor_chain} returned {len(hit_ids)} hit(s)",
    )
    if not hit_ids:
        return candidates[:max_conformers]
    center = _site_center_from_details(pdb_path, chains, site_details)
    aligned: list[_AlignedCandidate] = []
    for entry_id in hit_ids:
        downloaded = _download_rcsb_pdb(entry_id, output_dir / "remote_raw")
        if downloaded is None:
            _emit(logger, f"ensemble: download failed for remote candidate {entry_id}")
            continue
        aligned_candidate = _align_pdb_candidate(
            pdb_path,
            downloaded,
            [anchor_chain],
            output_dir / "remote_aligned",
            site_details,
        )
        if aligned_candidate is not None:
            aligned.append(aligned_candidate)
        else:
            _emit(logger, f"ensemble: alignment rejected remote candidate {entry_id}")
    _emit(logger, f"ensemble: {len(aligned)} remote candidate(s) aligned successfully")
    selected_remote = _select_diverse_aligned_candidates(
        pdb_path, aligned, center, remaining
    )
    _emit(
        logger,
        f"ensemble: kept {len(selected_remote)} remote candidate(s) after diversity filter",
    )
    candidates.extend(selected_remote)
    return candidates[:max_conformers]


def _nma_fallback_candidates(
    pdb_path: Path,
    chains: list[str],
    site_details: dict[str, Any],
    output_dir: Path,
    max_conformers: int,
    *,
    logger: Callable[[str], None] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import prody  # type: ignore

        atoms = prody.parsePDB(str(pdb_path))
        selection = atoms.select("protein and name CA")
        if selection is None or len(selection) < 3:
            raise ValueError("not enough CA atoms for ANM")
        n_modes = max(1, min(3, max_conformers))
        anm = prody.ANM("ezscreen_auto_ensemble")
        anm.buildHessian(selection)
        anm.calcModes(n_modes=n_modes)
        candidates: list[Path] = []
        for mode_index, mode in enumerate(anm[:n_modes], 1):
            traversal = prody.traverseMode(mode, selection, rmsd=1.0, n_steps=3)
            coordsets_array = traversal.getCoordsets()
            if coordsets_array is None:
                continue
            coordsets = list(coordsets_array)
            if len(coordsets) < 2:
                continue
            for direction, coords in (("neg", coordsets[0]), ("pos", coordsets[-1])):
                moved = _apply_ca_displacements_to_all_atoms(atoms, selection, coords)
                out = output_dir / f"nma_mode_{mode_index:03d}_{direction}.pdb"
                prody.writePDB(str(out), moved)
                candidates.append(out)
                if len(candidates) >= max_conformers:
                    _emit(
                        logger, f"ensemble: NMA produced {len(candidates)} candidate(s)"
                    )
                    return candidates
        if candidates:
            _emit(logger, f"ensemble: NMA produced {len(candidates)} candidate(s)")
            return candidates
    except Exception as exc:
        _emit(logger, f"ensemble: NMA fallback unavailable ({exc})")
    out = output_dir / "nma_conformer_001.pdb"
    text = Path(pdb_path).read_text(encoding="utf-8", errors="replace")
    out.write_text(
        "REMARK EZSCREEN AUTO ENSEMBLE NMA FALLBACK\n" + text, encoding="utf-8"
    )
    _emit(logger, "ensemble: wrote marked-copy NMA fallback candidate")
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
    logger: Callable[[str], None] | None = None,
) -> list[EnsembleConformer]:
    from ezscreen.errors import ReceptorPrepError

    if max_conformers < 1:
        raise ValueError("max_conformers must be at least 1")
    max_conformers = min(int(max_conformers), 8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    prep_chains = _infer_pocket_chains(Path(pdb_path), chains, site_details)
    try:
        qc_reference_pdb = _prepare_qc_reference(
            Path(pdb_path),
            output_dir=output_dir / "qc_reference",
            chains=prep_chains if prep_chains else chains,
            ph=ph,
            keep_hetero=keep_hetero,
            rec_prep_module=rec_prep_module,
        )
    except ReceptorPrepError as exc:
        qc_reference_pdb = Path(pdb_path)
        _emit(
            logger,
            f"ensemble: QC reference prep failed; falling back to input receptor ({exc})",
        )
    _emit(
        logger,
        f"ensemble: using prep_chains={','.join(prep_chains) if prep_chains else 'none'}",
    )
    pdb_candidate_limit = max(max_conformers, min(12, max_conformers * 4))
    candidate_sources = [
        (
            "pdb",
            lambda: _pdb_derived_candidates(
                Path(pdb_path),
                chains,
                site_details,
                candidate_dir / "pdb",
                pdb_candidate_limit,
                logger=logger,
            ),
            max_conformers,
        ),
        (
            "nma",
            lambda: _nma_fallback_candidates(
                Path(pdb_path),
                chains,
                site_details,
                candidate_dir / "nma",
                max_conformers,
                logger=logger,
            ),
            max_conformers,
        ),
    ]
    for source, provider, keep_limit in candidate_sources:
        candidates = list(provider())
        _emit(
            logger, f"ensemble: source={source} yielded {len(candidates)} candidate(s)"
        )
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
                    prep_chains=prep_chains if source == "pdb" else chains,
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
                # QC the geometry on the aligned/raw candidate frame; prepared receptors may
                # be renumbered heavily enough to obscure pocket-residue correspondence.
                passes_qc, qc_meta = _passes_auto_qc(
                    qc_reference_pdb,
                    Path(candidate),
                    chains,
                    site_details,
                )
                if not passes_qc:
                    _emit(
                        logger,
                        f"ensemble: QC rejected {conformer.name} ({qc_meta.get('qc_reason', 'unknown')})",
                    )
                    continue
                _emit(logger, f"ensemble: QC accepted {conformer.name}")
                conformers.append(
                    EnsembleConformer(
                        conformer.name,
                        conformer.source,
                        conformer.pdb_path,
                        conformer.pdbqt_path,
                        {**conformer.metadata, **qc_meta},
                    )
                )
            except ReceptorPrepError as exc:
                _emit(
                    logger, f"ensemble: prep failed for {Path(candidate).name} ({exc})"
                )
                continue
        deduped = _sort_auto_conformers(_dedupe_conformers(conformers, logger=logger))
        _emit(
            logger,
            f"ensemble: source={source} kept {len(deduped)} conformer(s) after prep/QC/dedupe",
        )
        if deduped:
            return deduped[:keep_limit]
    _emit(logger, "ensemble: no auto conformers survived any source")
    return []
