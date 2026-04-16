from __future__ import annotations

import csv
from pathlib import Path

from ezscreen.benchmark.metrics import BenchmarkResult, compute_metrics


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _load_actives(path: Path) -> set[str]:
    canonical = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            c = _canonical(smi)
            if c:
                canonical.add(c)
    return canonical


def run_benchmark(
    actives_path: Path,
    scores_csv: Path,
    smiles_col: str = "smiles",
    score_col: str | None = None,
) -> BenchmarkResult:
    actives = _load_actives(actives_path)

    rows: list[dict] = []
    with scores_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {scores_csv}")

    headers = list(rows[0].keys())

    # auto-detect score column if not given
    if score_col is None:
        score_col = next(
            (h for h in headers if "score" in h.lower() or "affinity" in h.lower()),
            headers[-1],
        )

    scores: list[float] = []
    labels: list[int]   = []
    for row in rows:
        raw_score = row.get(score_col, "")
        try:
            s = float(raw_score)
        except (ValueError, TypeError):
            continue

        smi = row.get(smiles_col, "")
        c   = _canonical(smi) if smi else None
        label = 1 if (c and c in actives) else 0

        scores.append(s)
        labels.append(label)

    n_matched = sum(labels)
    if n_matched == 0:
        raise ValueError(
            f"None of the {len(actives)} known actives matched anything in {scores_csv.name}. "
            "Check that the SMILES in your actives file appear in the docking results."
        )

    return compute_metrics(scores, labels)
