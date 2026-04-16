from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    ef1: float
    ef5: float
    auc_roc: float
    n_actives: int
    n_decoys: int
    total_screened: int
    ranked_labels: list[int] = field(default_factory=list)  # 1=active, 0=decoy, best score first


def compute_metrics(
    scores: list[float],
    labels: list[int],
) -> BenchmarkResult:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")

    paired = sorted(zip(scores, labels), key=lambda x: x[0])  # ascending (lower score = better)
    ranked_labels = [lbl for _, lbl in paired]

    n_actives = sum(labels)
    n_decoys  = len(labels) - n_actives
    n_total   = len(labels)

    def enrichment_factor(cutoff_frac: float) -> float:
        n_top = max(1, int(n_total * cutoff_frac))
        actives_in_top = sum(ranked_labels[:n_top])
        if n_actives == 0:
            return 0.0
        return (actives_in_top / n_top) / (n_actives / n_total)

    ef1 = enrichment_factor(0.01)
    ef5 = enrichment_factor(0.05)
    auc = _auc_roc(ranked_labels, n_actives, n_decoys)

    return BenchmarkResult(
        ef1=round(ef1, 3),
        ef5=round(ef5, 3),
        auc_roc=round(auc, 4),
        n_actives=n_actives,
        n_decoys=n_decoys,
        total_screened=n_total,
        ranked_labels=ranked_labels,
    )


def _auc_roc(ranked_labels: list[int], n_actives: int, n_decoys: int) -> float:
    if n_actives == 0 or n_decoys == 0:
        return 0.0

    tpr_points = [0.0]
    fpr_points = [0.0]
    tp = fp = 0

    for label in ranked_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_actives)
        fpr_points.append(fp / n_decoys)

    # trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_points)):
        auc += (fpr_points[i] - fpr_points[i - 1]) * (tpr_points[i] + tpr_points[i - 1]) / 2
    return auc
