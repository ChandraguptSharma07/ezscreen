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
