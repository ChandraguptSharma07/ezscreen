"""Validate, merge, rank, and score the 12-shard ACE AD4 GPU campaign."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

from ezscreen.benchmark.metrics import compute_metrics

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
FINAL = HERE / "final_results"
MAP_ZIP = HERE.parent / "map_bundle_output_020/ad4_map_bundle.zip"
TOTAL, ACTIVES, DECOYS = 1846, 49, 1797
MIN_COVERAGE = 0.98


def rows_from(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))


def json_from(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name))


def expected_map_manifest() -> dict:
    with zipfile.ZipFile(MAP_ZIP) as archive:
        return json.loads(archive.read("ad4_maps_manifest.json"))


def enrichment(labels: list[int], fraction: float) -> float:
    n = len(labels)
    n_active = sum(labels)
    if not n or not n_active:
        return 0.0
    top = max(1, math.ceil(n * fraction))
    return round((sum(labels[:top]) / top) / (n_active / n), 6)


def pr_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    positives = sum(labels)
    if not positives:
        return {"average_precision": 0.0, "pr_auc_trapezoid": 0.0}
    groups = []
    for score, label in sorted(zip(scores, labels), key=lambda item: item[0]):
        if groups and score == groups[-1][0]:
            groups[-1][1].append(label)
        else:
            groups.append((score, [label]))
    tp = fp = 0
    previous_recall, previous_precision = 0.0, 1.0
    ap = area = 0.0
    for _score, group_labels in groups:
        tp += sum(group_labels)
        fp += len(group_labels) - sum(group_labels)
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        area += (recall - previous_recall) * (precision + previous_precision) / 2
        previous_recall, previous_precision = recall, precision
    return {"average_precision": round(ap, 6), "pr_auc_trapezoid": round(area, 6)}


def bayes_ef_proxy(scored: list[dict], fraction: float) -> float:
    active_scores = [row["score"] for row in scored if row["label"] == 1]
    decoy_scores = sorted(row["score"] for row in scored if row["label"] == 0)
    if not active_scores or not decoy_scores:
        return 0.0
    cutoff = decoy_scores[max(0, math.ceil(len(decoy_scores) * fraction) - 1)]
    active_fraction = sum(score <= cutoff for score in active_scores) / len(
        active_scores
    )
    random_fraction = sum(score <= cutoff for score in decoy_scores) / len(decoy_scores)
    return round(active_fraction / random_fraction, 6) if random_fraction else 0.0


def bayes_ef_max_proxy(scored: list[dict]) -> dict:
    active_scores = [row["score"] for row in scored if row["label"] == 1]
    decoy_scores = sorted(row["score"] for row in scored if row["label"] == 0)
    best = {"value": 0.0, "threshold": None, "decoy_fraction": None}
    for threshold in sorted(set(decoy_scores)):
        a = sum(score <= threshold for score in active_scores) / len(active_scores)
        d = sum(score <= threshold for score in decoy_scores) / len(decoy_scores)
        value = a / d if d else 0.0
        if value > best["value"]:
            best = {"value": value, "threshold": threshold, "decoy_fraction": d}
    return {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in best.items()
    }


def main() -> None:
    expected_manifest = expected_map_manifest()
    plan = json.loads((HERE / "shard_plan.json").read_text())
    if plan["total"] != TOTAL or plan["shards"] != 12:
        raise RuntimeError("shard plan mismatch")
    combined: dict[int, dict] = {}
    shard_audit, runtimes = [], []

    for bound in plan["bounds"]:
        shard = int(bound["shard"])
        run = WORK / f"ace-1o86-ad4-full-s{shard:02d}" / "output"
        bundle = run / "output.zip"
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        with zipfile.ZipFile(bundle) as archive:
            required = {
                "scores.csv",
                "prepared_manifest.csv",
                "prep_failures.csv",
                "fallbacks.csv",
                "failed_docking.csv",
                "ad4_map_runtime.json",
                "runtime.json",
                "docked_pdbqt.zip",
            }
            missing = required - set(archive.namelist())
            if missing:
                raise RuntimeError(f"shard {shard}: missing {sorted(missing)}")
            runtime = json_from(archive, "runtime.json")
            map_runtime = json_from(archive, "ad4_map_runtime.json")
            if runtime["range"] != [bound["start"], bound["end"]]:
                raise RuntimeError(f"shard {shard}: range mismatch")
            if runtime["intended"] != bound["count"]:
                raise RuntimeError(f"shard {shard}: intended mismatch")
            if runtime.get("lower_is_better") is not True:
                raise RuntimeError(f"shard {shard}: ordering flag")
            if map_runtime.get("manifest") != expected_manifest:
                raise RuntimeError(f"shard {shard}: map manifest mismatch")
            if (
                map_runtime.get("fld_sha256")
                != expected_manifest["map_files"][
                    f"{expected_manifest['maps_prefix']}.maps.fld"
                ]
            ):
                raise RuntimeError(f"shard {shard}: fld hash mismatch")
            score_rows = rows_from(archive, "scores.csv")
            prep_failed = rows_from(archive, "prep_failures.csv")
            dock_failed = rows_from(archive, "failed_docking.csv")
            fallbacks = rows_from(archive, "fallbacks.csv")
            local = []
            for row in score_rows:
                score = float(row["score"])
                if not math.isfinite(score):
                    raise RuntimeError(f"shard {shard}: nonfinite score")
                local.append(
                    {
                        "sample_index": int(row["sample_index"]),
                        "source_id": row["source_id"],
                        "label": int(row["label"]),
                        "score": score,
                        "status": "scored",
                        "prep_method": row["prep_method"],
                        "failure_reason": "",
                    }
                )
            for row in prep_failed:
                local.append(
                    {
                        "sample_index": int(row["sample_index"]),
                        "source_id": row["source_id"],
                        "label": int(row["label"]),
                        "score": None,
                        "status": "prep_failed",
                        "prep_method": "",
                        "failure_reason": row["reason"],
                    }
                )
            for row in dock_failed:
                local.append(
                    {
                        "sample_index": int(row["sample_index"]),
                        "source_id": row["source_id"],
                        "label": int(row["label"]),
                        "score": None,
                        "status": "dock_failed",
                        "prep_method": "",
                        "failure_reason": row["reason"],
                    }
                )
            expected_indices = set(range(bound["start"], bound["end"]))
            got_indices = [row["sample_index"] for row in local]
            if (
                len(got_indices) != len(set(got_indices))
                or set(got_indices) != expected_indices
            ):
                raise RuntimeError(f"shard {shard}: index accounting mismatch")
            for row in local:
                if row["sample_index"] in combined:
                    raise RuntimeError("duplicate global index")
                combined[row["sample_index"]] = row
            runtimes.append(float(runtime["total_seconds"]))
            shard_audit.append(
                {
                    "shard": shard,
                    "intended": bound["count"],
                    "scored": len(score_rows),
                    "prep_failures": len(prep_failed),
                    "dock_failures": len(dock_failed),
                    "fallbacks": len(fallbacks),
                    "runtime_seconds": runtime["total_seconds"],
                }
            )

    if set(combined) != set(range(TOTAL)):
        raise RuntimeError("global accounting mismatch")
    source_ids = [row["source_id"] for row in combined.values()]
    source_id_counts = Counter(source_ids)
    duplicate_source_ids = {
        source_id: count
        for source_id, count in source_id_counts.items()
        if count > 1
    }
    for sample_index, row in combined.items():
        # DUD2 contains separate records sharing a ZINC ID. The archive ordinal is
        # therefore part of record identity and must not be deduplicated.
        row["source_record_key"] = f"{row['source_id']}@{sample_index}"
    labels = [row["label"] for row in combined.values()]
    if (sum(labels), len(labels) - sum(labels)) != (ACTIVES, DECOYS):
        raise RuntimeError("label count mismatch")

    scored = sorted(
        (row for row in combined.values() if row["score"] is not None),
        key=lambda row: (row["score"], row["sample_index"]),
    )
    failed = sorted(
        (row for row in combined.values() if row["score"] is None),
        key=lambda row: row["sample_index"],
    )
    ranking = scored + failed
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    ranked_labels = [row["label"] for row in ranking]
    coverage = len(scored) / TOTAL
    worst = (max(row["score"] for row in scored) + 1.0) if scored else 0.0
    complete_scores = [
        row["score"] if row["score"] is not None else worst for row in ranking
    ]
    metric = compute_metrics(
        complete_scores,
        ranked_labels,
        scored=[row["status"] == "scored" for row in ranking],
    )
    valid_metric = (
        compute_metrics(
            [row["score"] for row in scored], [row["label"] for row in scored]
        )
        if scored
        else None
    )
    pr = pr_metrics(complete_scores, ranked_labels)
    metrics = {
        "protocol": "UniDock AD4 accelerated port; not AutoDock Vina-equivalent; 0.52 A map spacing",
        "total": TOTAL,
        "actives": ACTIVES,
        "decoys": DECOYS,
        "scored": len(scored),
        "coverage": round(coverage, 6),
        "coverage_gate": MIN_COVERAGE,
        "coverage_gate_passed": coverage >= MIN_COVERAGE,
        "failures": len(failed),
        "filtered_rows": len(failed),
        "unique_source_ids": len(source_id_counts),
        "duplicate_source_id_groups": len(duplicate_source_ids),
        "repeated_source_id_records": sum(
            count - 1 for count in duplicate_source_ids.values()
        ),
        "ef1": metric.ef1,
        "ef5": metric.ef5,
        "ef10": enrichment(ranked_labels, 0.10),
        "ef20": metric.ef20,
        "roc_auc": metric.auc_roc,
        "bedroc20": metric.bedroc,
        **pr,
        "bayes_ef_decoy_proxy_1pct": bayes_ef_proxy(scored, 0.01),
        "bayes_ef_decoy_proxy_5pct": bayes_ef_proxy(scored, 0.05),
        "bayes_ef_max_decoy_proxy": bayes_ef_max_proxy(scored),
        "bayes_ef_caveat": "DUD2 decoys are not random compounds; values are decoy-distribution proxies, not canonical Bayes EF.",
        "valid_only_sensitivity": None
        if valid_metric is None
        else {
            "roc_auc": valid_metric.auc_roc,
            "bedroc20": valid_metric.bedroc,
            "ef1": valid_metric.ef1,
            "ef5": valid_metric.ef5,
            "ef20": valid_metric.ef20,
        },
        "runtime_seconds_sum": round(sum(runtimes), 3),
        "runtime_seconds_max_wall_proxy": round(max(runtimes), 3),
    }
    FINAL.mkdir(parents=True, exist_ok=True)
    with (FINAL / "ranking_1846.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank",
            "sample_index",
            "source_record_key",
            "source_id",
            "label",
            "score",
            "status",
            "prep_method",
            "failure_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranking)
    with (FINAL / "filtered_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "rank",
            "sample_index",
            "source_record_key",
            "source_id",
            "label",
            "score",
            "status",
            "prep_method",
            "failure_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failed)
    (FINAL / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (FINAL / "shard_audit.json").write_text(json.dumps(shard_audit, indent=2) + "\n")
    (FINAL / "provenance.json").write_text(
        json.dumps(
            {
                "target": "ACE",
                "pdb_id": "1O86",
                "dataset": "DUD Release 2",
                "map_manifest": expected_manifest,
                "shards": shard_audit,
                "ranking_policy": "finite scores ascending; failed compounds retained after scored compounds",
                "record_identity": "source_record_key is source_id@sample_index because DUD2 may repeat a ZINC ID across separate archive records",
                "duplicate_source_id_counts": duplicate_source_ids,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
