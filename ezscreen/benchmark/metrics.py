from __future__ import annotations

import math
from dataclasses import dataclass, field

BEDROC_ALPHA = 20.0


@dataclass
class BenchmarkResult:
    ef1: float
    ef5: float
    ef20: float
    auc_roc: float
    bedroc: float
    coverage: float
    first_active_rank: int | None
    n_actives: int
    n_decoys: int
    total_screened: int
    ranked_labels: list[int] = field(default_factory=list)
    label_diagnostics: dict = field(default_factory=dict)
    ranking_diagnostics: dict = field(default_factory=dict)


def _score_groups(paired: list[tuple[float, int]]) -> list[tuple[float, list[int]]]:
    groups: list[tuple[float, list[int]]] = []
    for score, label in paired:
        if groups and score == groups[-1][0]:
            groups[-1][1].append(label)
        else:
            groups.append((score, [label]))
    return groups


def _expected_actives_at_cutoff(
    groups: list[tuple[float, list[int]]], cutoff_count: int
) -> float:
    remaining = cutoff_count
    expected = 0.0
    for _score, group_labels in groups:
        if remaining <= 0:
            break
        take = min(remaining, len(group_labels))
        expected += take * (sum(group_labels) / len(group_labels))
        remaining -= take
    return expected


def _tie_aware_auc(
    groups: list[tuple[float, list[int]]], n_actives: int, n_decoys: int
) -> float:
    if n_actives == 0 or n_decoys == 0:
        return 0.0
    decoys_better = 0
    wins = 0.0
    for _score, group_labels in groups:
        group_actives = sum(group_labels)
        group_decoys = len(group_labels) - group_actives
        decoys_worse = n_decoys - decoys_better - group_decoys
        wins += group_actives * (decoys_worse + 0.5 * group_decoys)
        decoys_better += group_decoys
    return wins / (n_actives * n_decoys)


def _tie_aware_bedroc(
    groups: list[tuple[float, list[int]]],
    n_total: int,
    n_actives: int,
    alpha: float,
) -> float:
    if n_total == 0 or n_actives == 0:
        return 0.0
    if alpha <= 0:
        raise ValueError('BEDROC alpha must be greater than zero')
    rank = 1
    sum_exp = 0.0
    for _score, group_labels in groups:
        group_size = len(group_labels)
        group_actives = sum(group_labels)
        if group_actives:
            weight_sum = sum(
                math.exp(-(alpha * position) / n_total)
                for position in range(rank, rank + group_size)
            )
            sum_exp += (group_actives / group_size) * weight_sum
        rank += group_size
    denom = (1.0 / n_total) * (
        (1.0 - math.exp(-alpha)) / math.expm1(alpha / n_total)
    )
    rie = sum_exp / (n_actives * denom)
    active_ratio = n_actives / n_total
    rie_max = (1.0 - math.exp(-alpha * active_ratio)) / (
        active_ratio * (1.0 - math.exp(-alpha))
    )
    rie_min = math.expm1(alpha * active_ratio) / (
        active_ratio * math.expm1(alpha)
    )
    if math.isclose(rie_max, rie_min):
        return 1.0
    return (rie - rie_min) / (rie_max - rie_min)


def compute_metrics(
    scores: list[float],
    labels: list[int],
    *,
    scored: list[bool] | None = None,
    bedroc_alpha: float = BEDROC_ALPHA,
) -> BenchmarkResult:
    if len(scores) != len(labels):
        raise ValueError('scores and labels must be the same length')
    if scored is not None and len(scored) != len(scores):
        raise ValueError('scored and scores must be the same length')
    if any(label not in (0, 1) for label in labels):
        raise ValueError('labels must contain only 0 or 1')
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError('scores must be finite')
    paired = sorted(
        zip((float(score) for score in scores), labels), key=lambda item: item[0]
    )
    ranked_labels = [label for _score, label in paired]
    groups = _score_groups(paired)
    n_actives = sum(labels)
    n_decoys = len(labels) - n_actives
    n_total = len(labels)

    def enrichment_factor(cutoff_frac: float) -> float:
        if n_total == 0 or n_actives == 0:
            return 0.0
        n_top = max(1, math.ceil(n_total * cutoff_frac))
        expected_actives = _expected_actives_at_cutoff(groups, n_top)
        return (expected_actives / n_top) / (n_actives / n_total)

    first_active_rank: int | None = None
    rank = 1
    for _score, group_labels in groups:
        if any(group_labels):
            first_active_rank = rank
            break
        rank += len(group_labels)
    coverage = (
        sum(bool(value) for value in scored) / n_total
        if scored is not None and n_total
        else (1.0 if n_total else 0.0)
    )
    return BenchmarkResult(
        ef1=round(enrichment_factor(0.01), 3),
        ef5=round(enrichment_factor(0.05), 3),
        ef20=round(enrichment_factor(0.20), 3),
        auc_roc=round(_tie_aware_auc(groups, n_actives, n_decoys), 4),
        bedroc=round(_tie_aware_bedroc(groups, n_total, n_actives, bedroc_alpha), 4),
        coverage=round(coverage, 6),
        first_active_rank=first_active_rank,
        n_actives=n_actives,
        n_decoys=n_decoys,
        total_screened=n_total,
        ranked_labels=ranked_labels,
    )


def _auc_roc(ranked_labels: list[int], n_actives: int, n_decoys: int) -> float:
    groups = [(float(rank), [label]) for rank, label in enumerate(ranked_labels)]
    return _tie_aware_auc(groups, n_actives, n_decoys)
