from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from ezscreen.results.clustering import (  # noqa: E402
    cluster_by_interactions,
)


def _entry(lig_id, contacts):
    return {
        "lig_id": lig_id,
        "name": lig_id,
        "interactions": [
            {"chain": "A", "residue_number": r, "type": t} for r, t in contacts
        ],
    }


def test_identical_profiles_cluster_together():
    entries = [
        _entry("a", [(42, "hbond"), (45, "hydrophobic")]),
        _entry("b", [(42, "hbond"), (45, "hydrophobic")]),   # same as a
        _entry("c", [(90, "pi_stack")]),                     # distinct
    ]
    result = cluster_by_interactions(entries)
    assert result.n_clusters == 2
    # a and b share a cluster; c is on its own
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] != result.labels[0]


def test_entries_without_interactions_are_unclustered():
    entries = [
        _entry("a", [(42, "hbond")]),
        {"lig_id": "b", "name": "b", "interactions": []},
    ]
    result = cluster_by_interactions(entries)
    assert result.labels[1] == -1          # no contacts → not clustered
    assert result.n_clusters == 1          # only 'a' clusters (single valid entry)
