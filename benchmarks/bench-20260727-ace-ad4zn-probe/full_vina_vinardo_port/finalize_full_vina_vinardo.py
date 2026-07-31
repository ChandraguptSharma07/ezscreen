"""Validate and rank both ACE UniDock Vina/Vinardo GPU port campaigns."""

from __future__ import annotations

import csv
import hashlib
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
AD4_WORK = HERE.parent / "full_ad4_port" / "work"
TOTAL, ACTIVES, DECOYS = 1846, 49, 1797
MIN_COVERAGE = 0.98
MODES = ("vina", "vinardo")
PUBLISHED = {
    "vina": {"roc_auc": 0.431, "bedroc20": 0.159},
    "vinardo": {"roc_auc": 0.431, "bedroc20": 0.165},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))


def json_from(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name))


def enrichment(labels: list[int], fraction: float) -> float:
    if not labels or not sum(labels):
        return 0.0
    top = max(1, math.ceil(len(labels) * fraction))
    return round((sum(labels[:top]) / top) / (sum(labels) / len(labels)), 6)


def pr_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    positives = sum(labels)
    if not positives:
        return {"average_precision": 0.0, "pr_auc_trapezoid": 0.0}
    groups: list[tuple[float, list[int]]] = []
    for score, label in sorted(zip(scores, labels), key=lambda item: item[0]):
        if groups and score == groups[-1][0]:
            groups[-1][1].append(label)
        else:
            groups.append((score, [label]))
    tp = fp = 0
    prior_recall, prior_precision = 0.0, 1.0
    average_precision = area = 0.0
    for _score, group_labels in groups:
        tp += sum(group_labels)
        fp += len(group_labels) - sum(group_labels)
        recall, precision = tp / positives, tp / (tp + fp)
        average_precision += (recall - prior_recall) * precision
        area += (recall - prior_recall) * (precision + prior_precision) / 2
        prior_recall, prior_precision = recall, precision
    return {
        "average_precision": round(average_precision, 6),
        "pr_auc_trapezoid": round(area, 6),
    }


def bayes_ef_proxy(scored: list[dict], fraction: float) -> float:
    active = [row["score"] for row in scored if row["label"] == 1]
    decoy = sorted(row["score"] for row in scored if row["label"] == 0)
    if not active or not decoy:
        return 0.0
    cutoff = decoy[max(0, math.ceil(len(decoy) * fraction) - 1)]
    active_fraction = sum(score <= cutoff for score in active) / len(active)
    decoy_fraction = sum(score <= cutoff for score in decoy) / len(decoy)
    return round(active_fraction / decoy_fraction, 6) if decoy_fraction else 0.0


def bayes_ef_max_proxy(scored: list[dict]) -> dict:
    active = [row["score"] for row in scored if row["label"] == 1]
    decoy = sorted(row["score"] for row in scored if row["label"] == 0)
    best = {"value": 0.0, "threshold": None, "decoy_fraction": None}
    for threshold in sorted(set(decoy)):
        active_fraction = sum(score <= threshold for score in active) / len(active)
        decoy_fraction = sum(score <= threshold for score in decoy) / len(decoy)
        value = active_fraction / decoy_fraction if decoy_fraction else 0.0
        if value > best["value"]:
            best = {
                "value": value,
                "threshold": threshold,
                "decoy_fraction": decoy_fraction,
            }
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in best.items()
    }


def expected_prepared(shard: int) -> list[dict[str, str]]:
    bundle = AD4_WORK / f"ace-1o86-ad4-full-s{shard:02d}" / "output" / "output.zip"
    with zipfile.ZipFile(bundle) as archive:
        return rows_from(archive, "prepared_manifest.csv")


def validate_command(command: list[str], mode: str) -> None:
    expected = {
        "--center_x": "40.7317",
        "--center_y": "32.8822",
        "--center_z": "47.3635",
        "--size_x": "36.8194",
        "--size_y": "41.835",
        "--size_z": "43.7048",
        "--spacing": "0.375",
        "--scoring": mode,
        "--exhaustiveness": "32",
    }
    if "--receptor" not in command or "--maps" in command:
        raise RuntimeError(f"{mode}: receptor command mismatch")
    for option, value in expected.items():
        if option not in command or command[command.index(option) + 1] != value:
            raise RuntimeError(f"{mode}: command mismatch for {option}")


def write_ranking(mode: str, combined: dict[int, dict], runtimes: list[float]) -> dict:
    source_counts = Counter(row["source_id"] for row in combined.values())
    duplicate_counts = {
        source_id: count for source_id, count in source_counts.items() if count > 1
    }
    for sample_index, row in combined.items():
        row["source_record_key"] = f"{row['source_id']}@{sample_index}"
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
    labels = [row["label"] for row in ranking]
    coverage = len(scored) / TOTAL
    worst = max((row["score"] for row in scored), default=-1.0) + 1.0
    complete_scores = [
        row["score"] if row["score"] is not None else worst for row in ranking
    ]
    metric = compute_metrics(
        complete_scores,
        labels,
        scored=[row["status"] == "scored" for row in ranking],
    )
    valid_metric = (
        compute_metrics(
            [row["score"] for row in scored], [row["label"] for row in scored]
        )
        if scored
        else None
    )
    metrics = {
        "protocol": f"UniDock {mode.capitalize()} accelerated port; not Vina-equivalent",
        "total": TOTAL,
        "actives": ACTIVES,
        "decoys": DECOYS,
        "scored": len(scored),
        "coverage": round(coverage, 6),
        "coverage_gate": MIN_COVERAGE,
        "coverage_gate_passed": coverage >= MIN_COVERAGE,
        "failures": len(failed),
        "filtered_rows": len(failed),
        "unique_source_ids": len(source_counts),
        "duplicate_source_id_groups": len(duplicate_counts),
        "repeated_source_id_records": sum(
            count - 1 for count in duplicate_counts.values()
        ),
        "ef1": metric.ef1,
        "ef5": metric.ef5,
        "ef10": enrichment(labels, 0.10),
        "ef20": metric.ef20,
        "roc_auc": metric.auc_roc,
        "bedroc20": metric.bedroc,
        **pr_metrics(complete_scores, labels),
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
        "published_context_only": PUBLISHED[mode],
        "published_delta_context_only": {
            "roc_auc": round(metric.auc_roc - PUBLISHED[mode]["roc_auc"], 6),
            "bedroc20": round(metric.bedroc - PUBLISHED[mode]["bedroc20"], 6),
        },
    }
    out = FINAL / mode
    out.mkdir(parents=True, exist_ok=True)
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
    for name, rows in (("ranking_1846.csv", ranking), ("filtered_rows.csv", failed)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    plan = json.loads((HERE / "shard_plan.json").read_text())
    if plan["total"] != TOTAL or plan["shards"] != 12 or plan["modes"] != list(MODES):
        raise RuntimeError("shard plan mismatch")
    combined = {mode: {} for mode in MODES}
    mode_runtimes = {mode: [] for mode in MODES}
    shard_audit = []
    identity_fields = (
        "sample_index",
        "source_id",
        "label",
        "prep_method",
        "source_sha256",
        "prepared_sha256",
    )

    for bound in plan["bounds"]:
        shard = int(bound["shard"])
        bundle = WORK / f"ace-1o86-vina-vinardo-s{shard:02d}" / "output" / "output.zip"
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        with zipfile.ZipFile(bundle) as archive:
            required = {
                "prepared_manifest.csv",
                "prep_failures.csv",
                "fallbacks.csv",
                "prepared_identity_check.json",
                "runtime.json",
            }
            for mode in MODES:
                required |= {
                    f"scores_{mode}.csv",
                    f"failed_{mode}.csv",
                    f"unidock_{mode}.log",
                    f"docked_{mode}.zip",
                }
            missing = required - set(archive.namelist())
            if missing:
                raise RuntimeError(f"shard {shard}: missing {sorted(missing)}")
            runtime = json_from(archive, "runtime.json")
            identity = json_from(archive, "prepared_identity_check.json")
            if runtime["range"] != [bound["start"], bound["end"]]:
                raise RuntimeError(f"shard {shard}: range mismatch")
            if runtime["intended"] != bound["count"] or runtime["lower_is_better"] is not True:
                raise RuntimeError(f"shard {shard}: protocol accounting mismatch")
            if runtime.get("grid_spacing") != 0.375 or runtime.get("grid_point_cap") != 2_097_152:
                raise RuntimeError(f"shard {shard}: GPU grid configuration mismatch")
            if runtime.get("prepared_identity_passed") is not True or identity.get("passed") is not True:
                raise RuntimeError(f"shard {shard}: prepared identity gate failed")
            prepared = rows_from(archive, "prepared_manifest.csv")
            expected = expected_prepared(shard)
            actual_identity = [tuple(row[field] for field in identity_fields) for row in prepared]
            expected_identity = [tuple(row[field] for field in identity_fields) for row in expected]
            if actual_identity != expected_identity:
                raise RuntimeError(f"shard {shard}: AD4 prepared manifest mismatch")
            prep_failed = rows_from(archive, "prep_failures.csv")
            fallbacks = rows_from(archive, "fallbacks.csv")
            audit = {
                "shard": shard,
                "output_zip_sha256": sha256(bundle),
                "intended": bound["count"],
                "prepared": len(prepared),
                "prep_failures": len(prep_failed),
                "fallbacks": len(fallbacks),
                "prepared_identity_passed": True,
                "modes": {},
            }
            for mode in MODES:
                mode_run = runtime["modes"][mode]
                validate_command(mode_run["command"], mode)
                score_rows = rows_from(archive, f"scores_{mode}.csv")
                dock_failed = rows_from(archive, f"failed_{mode}.csv")
                local = []
                for row in score_rows:
                    score = float(row["score"])
                    if not math.isfinite(score):
                        raise RuntimeError(f"shard {shard} {mode}: nonfinite score")
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
                            "prep_method": "autodocktools",
                            "failure_reason": row["reason"],
                        }
                    )
                expected_indices = set(range(bound["start"], bound["end"]))
                indices = [row["sample_index"] for row in local]
                if len(indices) != len(set(indices)) or set(indices) != expected_indices:
                    raise RuntimeError(f"shard {shard} {mode}: index accounting mismatch")
                for row in local:
                    if row["sample_index"] in combined[mode]:
                        raise RuntimeError(f"{mode}: duplicate global index")
                    combined[mode][row["sample_index"]] = row
                mode_runtimes[mode].append(float(mode_run["runtime_seconds"]))
                audit["modes"][mode] = {
                    "scored": len(score_rows),
                    "failures": len(dock_failed),
                    "runtime_seconds": mode_run["runtime_seconds"],
                }
            shard_audit.append(audit)

    metrics = {}
    for mode in MODES:
        if set(combined[mode]) != set(range(TOTAL)):
            raise RuntimeError(f"{mode}: global accounting mismatch")
        labels = [row["label"] for row in combined[mode].values()]
        if (sum(labels), len(labels) - sum(labels)) != (ACTIVES, DECOYS):
            raise RuntimeError(f"{mode}: label count mismatch")
        metrics[mode] = write_ranking(mode, combined[mode], mode_runtimes[mode])
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "shard_audit.json").write_text(json.dumps(shard_audit, indent=2) + "\n")
    (FINAL / "provenance.json").write_text(
        json.dumps(
            {
                "target": "ACE",
                "pdb_id": "1O86",
                "dataset": "DUD Release 2",
                "prepared_identity": "byte hashes match completed AD4 attempt 04 manifests",
                "receptor_sha256": "7c3e148c6c094ffc5bb5d10ddb7184983f0248590712a1ed7c82f3a752375585",
                "box_center": [40.7317, 32.8822, 47.3635],
                "box_size": [36.8194, 41.835, 43.7048],
                "exhaustiveness": 32,
                "grid_spacing": 0.375,
                "grid_point_cap": 2_097_152,
                "unidock_commit": "ebde3b4a22c8b2c7b26e523cc15109aab815be21",
                "protocol_label": "UniDock accelerated ports; not AutoDock Vina-equivalent",
                "ranking_policy": "finite scores ascending; failed records retained after scored records",
                "shards": shard_audit,
            },
            indent=2,
        )
        + "\n"
    )
    artifact_hashes = {
        str(path.relative_to(FINAL)).replace("\\", "/"): sha256(path)
        for path in sorted(FINAL.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    (FINAL / "artifact_manifest.json").write_text(
        json.dumps({"sha256": artifact_hashes}, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
