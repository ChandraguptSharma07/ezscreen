from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"
OUTPUT_PATH = ROOT / "verification.json"


def enrichment_factor(labels: list[int], fraction: float) -> tuple[int, int, float]:
    cutoff = max(1, math.ceil(len(labels) * fraction))
    active_hits = sum(labels[:cutoff])
    prevalence = sum(labels) / len(labels)
    return cutoff, active_hits, (active_hits / cutoff) / prevalence


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    ranking_path = (ROOT / protocol["reference_ranking"]["path"]).resolve()
    ranking_bytes = ranking_path.read_bytes()
    ranking_sha256 = hashlib.sha256(ranking_bytes).hexdigest()
    expected_sha256 = protocol["reference_ranking"]["sha256"]
    if ranking_sha256 != expected_sha256:
        raise RuntimeError("reference ranking SHA-256 mismatch")

    rows = list(csv.DictReader(ranking_bytes.decode("utf-8").splitlines()))
    target = protocol["target"]
    if len(rows) != target["total_records"]:
        raise RuntimeError("reference ranking record count mismatch")
    if len({row["source_record_key"] for row in rows}) != len(rows):
        raise RuntimeError("source_record_key values are not unique")
    if {row["status"] for row in rows} != {"scored"}:
        raise RuntimeError("reference ranking contains unscored records")
    expected_prep = set(protocol["preparation"]["observed_ligand_prep_methods"])
    if {row["prep_method"] for row in rows} != expected_prep:
        raise RuntimeError("reference ranking preparation method mismatch")

    labels = [int(row["label"]) for row in rows]
    if sum(labels) != target["actives"] or len(labels) - sum(labels) != target["decoys"]:
        raise RuntimeError("active/decoy accounting mismatch")
    scores = [float(row["score"]) for row in rows]
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError("reference ranking contains non-finite scores")
    if scores != sorted(scores):
        raise RuntimeError("reference ranking is not raw-score ascending")

    metrics = {}
    for name, fraction in (("ef2", 0.02), ("ef5", 0.05)):
        cutoff, active_hits, value = enrichment_factor(labels, fraction)
        published = float(protocol["published_reference"][name])
        rounded_match = round(value, 2) == published
        metrics[name] = {
            "fraction": fraction,
            "cutoff": cutoff,
            "active_hits": active_hits,
            "pipeline": round(value, 6),
            "published": published,
            "rounded_match": rounded_match,
        }
        if not rounded_match:
            raise RuntimeError(f"{name} does not reproduce the published value")

    result = {
        "status": "passed",
        "target": target,
        "ranking_path": str(ranking_path),
        "ranking_sha256": ranking_sha256,
        "coverage": 1.0,
        "failures": 0,
        "prep_methods": sorted(expected_prep),
        "metrics": metrics,
        "interpretation": (
            "ACE is a positive-control reproduction, not an untouched validation target."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
