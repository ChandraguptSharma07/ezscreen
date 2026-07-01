from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClusterResult:
    labels: list[int]           # cluster index per input molecule (-1 = no SMILES)
    centroid_indices: list[int] # index in rows[] for each cluster centroid
    sizes: list[int]            # molecules per cluster
    n_clusters: int


def cluster_hits(
    rows: list[dict],
    score_col: str,
    tanimoto_cutoff: float = 0.4,
) -> ClusterResult:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    # build fingerprints; track which rows had valid SMILES
    fps:       list = []
    valid_idx: list[int] = []
    for i, row in enumerate(rows):
        smi = row.get("smiles", "")
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            valid_idx.append(i)

    labels = [-1] * len(rows)

    if len(fps) < 2:
        # nothing to cluster
        for i, orig in enumerate(valid_idx):
            labels[orig] = i
        return ClusterResult(
            labels=labels,
            centroid_indices=valid_idx,
            sizes=[1] * len(valid_idx),
            n_clusters=len(valid_idx),
        )

    # upper-triangle distance matrix
    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, len(fps), 1.0 - tanimoto_cutoff, isDistData=True)

    centroid_indices: list[int] = []
    sizes: list[int] = []
    for cluster_id, members in enumerate(clusters):
        centroid_indices.append(valid_idx[members[0]])
        sizes.append(len(members))
        for fp_idx in members:
            labels[valid_idx[fp_idx]] = cluster_id

    return ClusterResult(
        labels=labels,
        centroid_indices=centroid_indices,
        sizes=sizes,
        n_clusters=len(clusters),
    )


@dataclass
class InteractionClusterResult:
    labels: list[int]                 # cluster index per entry (-1 = no interactions)
    representative_indices: list[int] # entry index of each cluster's representative
    sizes: list[int]
    n_clusters: int


def _interaction_fingerprint(entry: dict) -> frozenset:
    """A hit's contact set: (chain, residue_number, interaction_type) tuples."""
    feats = set()
    for ix in entry.get("interactions", []) or []:
        feats.add((
            str(ix.get("chain", "")),
            ix.get("residue_number", 0),
            str(ix.get("type", "")),
        ))
    return frozenset(feats)


def cluster_by_interactions(
    entries: list[dict],
    cutoff: float = 0.4,
) -> InteractionClusterResult:
    """Cluster hits by their PLIP interaction fingerprint (Jaccard + Butina).

    `entries` is the interactions_top_n.json list. Complements scaffold clustering:
    it groups hits that hit the same residues the same way, and picks the first
    (best-ranked) member of each cluster as the representative.
    """
    from rdkit.ML.Cluster import Butina

    fps       = [_interaction_fingerprint(e) for e in entries]
    valid_idx = [i for i, f in enumerate(fps) if f]
    labels    = [-1] * len(entries)

    if len(valid_idx) < 2:
        for cid, orig in enumerate(valid_idx):
            labels[orig] = cid
        return InteractionClusterResult(
            labels=labels,
            representative_indices=valid_idx,
            sizes=[1] * len(valid_idx),
            n_clusters=len(valid_idx),
        )

    vfps = [fps[i] for i in valid_idx]
    dists: list[float] = []
    for i in range(1, len(vfps)):
        a = vfps[i]
        for j in range(i):
            b = vfps[j]
            union = a | b
            sim = (len(a & b) / len(union)) if union else 1.0
            dists.append(1.0 - sim)

    clusters = Butina.ClusterData(dists, len(vfps), 1.0 - cutoff, isDistData=True)

    reps: list[int] = []
    sizes: list[int] = []
    for cid, members in enumerate(clusters):
        reps.append(valid_idx[members[0]])
        sizes.append(len(members))
        for m in members:
            labels[valid_idx[m]] = cid

    return InteractionClusterResult(
        labels=labels,
        representative_indices=reps,
        sizes=sizes,
        n_clusters=len(clusters),
    )


def export_centroids(
    rows: list[dict],
    result: ClusterResult,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx in result.centroid_indices:
        row = rows[idx]
        smi  = row.get("smiles", "").strip()
        name = row.get("name", str(idx)).strip()
        if smi:
            lines.append(f"{smi} {name}")
    output_path.write_text("\n".join(lines) + "\n")
    return len(lines)
