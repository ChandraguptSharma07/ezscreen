from __future__ import annotations

import math

import pytest

from ezscreen.benchmark.metrics import compute_metrics


def test_auc_gives_half_credit_for_score_ties() -> None:
    result = compute_metrics([-8.0, -8.0, -7.0, -6.0], [1, 0, 0, 1])
    assert result.auc_roc == 0.375


def test_ef_boundary_tie_is_order_independent() -> None:
    scores = [-10.0] * 10 + [-9.0] * 10 + [-8.0] * 80
    labels_a = [0] * 10 + [1] * 5 + [0] * 5 + [0] * 80
    labels_b = [0] * 10 + [0] * 5 + [1] * 5 + [0] * 80
    result_a = compute_metrics(scores, labels_a)
    result_b = compute_metrics(scores, labels_b)
    assert result_a.ef20 == result_b.ef20 == 5.0


def test_reports_ef20_bedroc_coverage_and_first_active_rank() -> None:
    result = compute_metrics(
        [-10.0, -9.0, -8.0, -7.0, -6.0],
        [0, 1, 0, 0, 0],
        scored=[True, True, True, False, False],
    )
    assert result.ef20 == 0.0
    assert result.coverage == 0.6
    assert result.first_active_rank == 2
    assert 0.0 <= result.bedroc <= 1.0


def test_bedroc_matches_rdkit_without_ties() -> None:
    rdkit_scoring = pytest.importorskip('rdkit.ML.Scoring.Scoring')
    scores = [-9.0, -8.0, -7.0, -6.0, -5.0]
    labels = [1, 0, 1, 0, 0]
    result = compute_metrics(scores, labels)
    expected = rdkit_scoring.CalcBEDROC([[label] for label in labels], 0, 20.0)
    assert math.isclose(result.bedroc, expected, abs_tol=5e-5)


def test_tied_bedroc_is_order_independent() -> None:
    scores = [-8.0, -8.0, -7.0, -6.0]
    assert compute_metrics(scores, [1, 0, 0, 1]).bedroc == compute_metrics(
        scores, [0, 1, 0, 1]
    ).bedroc
