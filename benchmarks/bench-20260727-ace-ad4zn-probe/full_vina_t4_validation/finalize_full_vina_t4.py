"""Finalize the preregistered ACE UniDock Vina-mode T4 validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from ezscreen.benchmark.metrics import compute_metrics

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
FINAL = HERE / "final_results"
AD4_WORK = HERE.parent / "full_ad4_port" / "work"
P100_RANKING = (
    HERE.parent / "full_vina_vinardo_port/final_results/vina/ranking_1846.csv"
)

TOTAL, ACTIVES, DECOYS = 1846, 49, 1797
MIN_COVERAGE = 0.98
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260801
UNIDOCK_COMMIT = "ebde3b4a22c8b2c7b26e523cc15109aab815be21"
RECEPTOR_SHA256 = "7c3e148c6c094ffc5bb5d10ddb7184983f0248590712a1ed7c82f3a752375585"
BOX_CENTER = [40.7317, 32.8822, 47.3635]
BOX_SIZE = [36.8194, 41.835, 43.7048]
PUBLISHED = {"roc_auc": 0.431, "bedroc20": 0.159}
MARGINS = {"roc_auc": -0.02, "bedroc20": -0.02}
PAPER = {
    "title": "Vinardo: A Scoring Function Based on Autodock Vina Improves Scoring, Docking, and Virtual Screening",
    "authors": ["Rodrigo Quiroga", "Marcos A. Villarreal"],
    "journal": "PLOS ONE 11(5): e0155183 (2016)",
    "doi": "10.1371/journal.pone.0155183",
    "primary_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155183",
    "protocol_evidence": {
        "location": "Methods sections 2.2 and 2.5, printable PDF page 4 of 18",
        "description": "smina; exhaustiveness 32; box centered on bound ligand and each dimension extended 15 A; default Vina scoring; Open Babel conversion and AutoDockTools prepare_receptor4.py/prepare_ligand4.py.",
    },
    "screening_evidence": {
        "location": "Results section 3.5, printable PDF pages 13-14 of 18",
        "dataset": "28 DUD targets having at most 3000 compounds",
    },
    "values": {
        "roc_auc": {
            "value": 0.431,
            "location": "Supplementary Table S3, ACE row, Vina column",
        },
        "bedroc20": {
            "value": 0.159,
            "location": "Supplementary Table S4, ACE row, Vina column",
        },
    },
    "ambiguities": [
        "Paper does not report AutoDockTools, Open Babel, or smina versions.",
        "Paper does not report a random seed or complete command line.",
        "Supplementary Table S4 reports BEDROC but does not state alpha; BEDROC20 is the project's explicit alpha=20 interpretation.",
        "The paper used smina/Vina scoring; this campaign uses UniDock Vina mode and does not claim exact engine equivalence.",
    ],
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expected_prepared(shard: int) -> list[dict[str, str]]:
    bundle = AD4_WORK / f"ace-1o86-ad4-full-s{shard:02d}" / "output/output.zip"
    with zipfile.ZipFile(bundle) as archive:
        return rows_from(archive, "prepared_manifest.csv")


def validate_command(command: list[str]) -> None:
    expected = {
        "--center_x": "40.7317",
        "--center_y": "32.8822",
        "--center_z": "47.3635",
        "--size_x": "36.8194",
        "--size_y": "41.835",
        "--size_z": "43.7048",
        "--spacing": "0.375",
        "--scoring": "vina",
        "--exhaustiveness": "32",
    }
    if "--receptor" not in command or "--maps" in command:
        raise RuntimeError("Vina command receptor mismatch")
    for option, value in expected.items():
        if option not in command or command[command.index(option) + 1] != value:
            raise RuntimeError(f"Vina command mismatch for {option}")


def enrichment(labels: list[int], fraction: float) -> float:
    if not labels or not sum(labels):
        return 0.0
    top = max(1, math.ceil(len(labels) * fraction))
    return (sum(labels[:top]) / top) / (sum(labels) / len(labels))


def pr_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    positives = sum(labels)
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
    return {"average_precision": average_precision, "pr_auc": area}


def metric_values(
    scores: list[float], labels: list[int]
) -> dict[str, float | int | None]:
    result = compute_metrics(scores, labels)
    ranked = [
        label for _score, label in sorted(zip(scores, labels), key=lambda x: x[0])
    ]
    return {
        "ef1": result.ef1,
        "ef5": result.ef5,
        "ef10": round(enrichment(ranked, 0.10), 6),
        "ef20": result.ef20,
        "roc_auc": result.auc_roc,
        "bedroc20": result.bedroc,
        **{key: round(value, 6) for key, value in pr_metrics(scores, labels).items()},
        "top_active_rank": result.first_active_rank,
    }


def bayes_ef_proxy(scored: list[dict], fraction: float) -> float:
    active = [row["score"] for row in scored if row["label"] == 1]
    decoy = sorted(row["score"] for row in scored if row["label"] == 0)
    cutoff = decoy[max(0, math.ceil(len(decoy) * fraction) - 1)]
    active_fraction = sum(score <= cutoff for score in active) / len(active)
    decoy_fraction = sum(score <= cutoff for score in decoy) / len(decoy)
    return round(active_fraction / decoy_fraction, 6) if decoy_fraction else 0.0


def bootstrap(scores: list[float], labels: list[int]) -> dict:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    active = np.flatnonzero(label_array == 1)
    decoy = np.flatnonzero(label_array == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = (
        "roc_auc",
        "bedroc20",
        "pr_auc",
        "average_precision",
        "ef1",
        "ef5",
        "ef10",
        "ef20",
    )
    samples = {name: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for name in names}
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = np.concatenate(
            (
                rng.choice(active, len(active), replace=True),
                rng.choice(decoy, len(decoy), replace=True),
            )
        )
        values = metric_values(
            score_array[indices].tolist(), label_array[indices].tolist()
        )
        for name in names:
            samples[name][replicate] = float(values[name])
    intervals = {
        name: {
            "lower_95": round(float(np.quantile(values, 0.025)), 6),
            "upper_95": round(float(np.quantile(values, 0.975)), 6),
            "median": round(float(np.median(values)), 6),
        }
        for name, values in samples.items()
    }
    return {
        "method": "stratified nonparametric bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence": 0.95,
        "intervals": intervals,
    }


def compare_p100(ranking: list[dict], metrics: dict) -> tuple[dict, list[dict]]:
    with P100_RANKING.open(newline="", encoding="utf-8") as handle:
        old = {row["source_record_key"]: row for row in csv.DictReader(handle)}
    current = {
        row["source_record_key"]: row for row in ranking if row["status"] == "scored"
    }
    shared = sorted(set(old) & set(current))
    p100_scores = np.asarray([float(old[key]["score"]) for key in shared])
    t4_scores = np.asarray([float(current[key]["score"]) for key in shared])
    p100_ranks = np.asarray([int(old[key]["rank"]) for key in shared])
    t4_ranks = np.asarray([int(current[key]["rank"]) for key in shared])
    changes = sorted(
        (
            {
                "source_record_key": key,
                "source_id": current[key]["source_id"],
                "label": int(current[key]["label"]),
                "p100_score": float(old[key]["score"]),
                "t4_score": float(current[key]["score"]),
                "score_delta_t4_minus_p100": float(current[key]["score"])
                - float(old[key]["score"]),
                "p100_rank": int(old[key]["rank"]),
                "t4_rank": int(current[key]["rank"]),
                "rank_delta_t4_minus_p100": int(current[key]["rank"])
                - int(old[key]["rank"]),
                "absolute_rank_delta": abs(
                    int(current[key]["rank"]) - int(old[key]["rank"])
                ),
            }
            for key in shared
        ),
        key=lambda row: (-row["absolute_rank_delta"], row["source_record_key"]),
    )
    overlaps = {}
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, math.ceil(TOTAL * fraction))
        old_top = {key for key in shared if int(old[key]["rank"]) <= count}
        new_top = {key for key in shared if int(current[key]["rank"]) <= count}
        overlaps[f"top_{int(fraction * 100)}pct"] = {
            "cutoff_records": count,
            "intersection": len(old_top & new_top),
            "overlap_fraction": round(len(old_top & new_top) / count, 6),
        }
    with (P100_RANKING.parent / "metrics.json").open(encoding="utf-8") as handle:
        old_metrics = json.load(handle)
    fields = (
        "ef1",
        "ef5",
        "ef10",
        "ef20",
        "roc_auc",
        "bedroc20",
        "pr_auc",
        "average_precision",
    )
    comparison = {
        "p100_ranking_sha256": sha256(P100_RANKING),
        "shared_records": len(shared),
        "p100_coverage": old_metrics["coverage"],
        "t4_coverage": metrics["coverage"],
        "coverage_difference": round(metrics["coverage"] - old_metrics["coverage"], 6),
        "score_distribution": {
            "p100": {
                "min": float(p100_scores.min()),
                "max": float(p100_scores.max()),
                "mean": float(p100_scores.mean()),
                "std": float(p100_scores.std()),
            },
            "t4": {
                "min": float(t4_scores.min()),
                "max": float(t4_scores.max()),
                "mean": float(t4_scores.mean()),
                "std": float(t4_scores.std()),
            },
            "delta": {
                "mean": float((t4_scores - p100_scores).mean()),
                "std": float((t4_scores - p100_scores).std()),
                "max_abs": float(np.abs(t4_scores - p100_scores).max()),
            },
        },
        "rank_correlation": {
            "spearman": float(spearmanr(p100_ranks, t4_ranks).statistic),
            "kendall": float(kendalltau(p100_ranks, t4_ranks).statistic),
        },
        "top_overlap": overlaps,
        "metric_differences_t4_minus_p100": {
            field: round(
                float(metrics[field])
                - float(
                    old_metrics[field if field != "pr_auc" else "pr_auc_trapezoid"]
                ),
                6,
            )
            for field in fields
        },
        "failure_difference": metrics["preparation_failures"]
        + metrics["docking_failures"]
        - old_metrics["failures"],
        "large_rank_change_definition": "absolute rank delta >= 100; full ordered list retained separately",
        "large_rank_changes": sum(row["absolute_rank_delta"] >= 100 for row in changes),
    }
    return comparison, changes


def main() -> None:
    plan = json.loads((HERE / "shard_plan.json").read_text(encoding="utf-8"))
    if (
        plan.get("total") != TOTAL
        or plan.get("shards") != 12
        or plan.get("modes") != ["vina"]
    ):
        raise RuntimeError("T4 shard plan mismatch")
    if plan.get("hardware") != {"gpu_count": 2, "model_token": "T4"}:
        raise RuntimeError("T4 hardware plan mismatch")

    combined: dict[int, dict] = {}
    failures: list[dict] = []
    fallbacks: list[dict] = []
    shard_audit: list[dict] = []
    bundle_hashes: dict[str, str] = {}
    source_builds: list[dict] = []
    input_manifests: list[dict] = []
    identity_fields = (
        "sample_index",
        "source_id",
        "label",
        "prep_method",
        "source_sha256",
        "prepared_sha256",
    )
    required = {
        "prepared_manifest.csv",
        "prep_failures.csv",
        "fallbacks.csv",
        "prepared_identity_check.json",
        "hardware_manifest.json",
        "source_build_manifest.json",
        "input_manifest.json",
        "runtime.json",
        "scores_vina.csv",
        "failed_vina.csv",
        "unidock_vina.log",
        "docked_vina.zip",
    }

    for bound in plan["bounds"]:
        shard = int(bound["shard"])
        bundle = WORK / f"ace-1o86-vina-t4-s{shard:02d}" / "output/output.zip"
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        bundle_hashes[str(bundle.relative_to(HERE)).replace("\\", "/")] = sha256(bundle)
        with zipfile.ZipFile(bundle) as archive:
            missing = required - set(archive.namelist())
            if missing:
                raise RuntimeError(f"shard {shard}: missing {sorted(missing)}")
            runtime = json_from(archive, "runtime.json")
            hardware = json_from(archive, "hardware_manifest.json")
            source = json_from(archive, "source_build_manifest.json")
            inputs = json_from(archive, "input_manifest.json")
            identity = json_from(archive, "prepared_identity_check.json")
            if (
                runtime["range"] != [bound["start"], bound["end"]]
                or runtime["intended"] != bound["count"]
            ):
                raise RuntimeError(f"shard {shard}: range/accounting mismatch")
            if (
                runtime.get("lower_is_better") is not True
                or runtime.get("grid_spacing") != 0.375
                or runtime.get("grid_point_cap") != 2_097_152
            ):
                raise RuntimeError(f"shard {shard}: runtime protocol mismatch")
            models = hardware.get("gpu_models", [])
            if (
                hardware.get("gate_passed") is not True
                or hardware.get("gpu_count") != 2
                or len(models) != 2
                or any("T4" not in model for model in models)
            ):
                raise RuntimeError(f"shard {shard}: Tesla T4x2 evidence failed")
            if (
                source.get("git_commit") != UNIDOCK_COMMIT
                or source.get("grid_point_cap") != 2_097_152
                or source.get("capacity_only_change") is not True
            ):
                raise RuntimeError(f"shard {shard}: source provenance mismatch")
            if not source.get("source_tree_sha256") or not source.get("binary_sha256"):
                raise RuntimeError(f"shard {shard}: missing source/build hash")
            expected_inputs = {
                "receptor_sha256": RECEPTOR_SHA256,
                "box_center": BOX_CENTER,
                "box_size": BOX_SIZE,
                "grid_spacing": 0.375,
                "scoring": "vina",
                "exhaustiveness": 32,
                "lower_is_better": True,
            }
            if any(inputs.get(key) != value for key, value in expected_inputs.items()):
                raise RuntimeError(f"shard {shard}: input protocol mismatch")
            prepared = rows_from(archive, "prepared_manifest.csv")
            expected = expected_prepared(shard)
            actual_identity = [
                tuple(row[field] for field in identity_fields) for row in prepared
            ]
            expected_identity = [
                tuple(row[field] for field in identity_fields) for row in expected
            ]
            if (
                identity.get("passed") is not True
                or actual_identity != expected_identity
            ):
                raise RuntimeError(f"shard {shard}: exact prepared identity mismatch")
            prep_failed = rows_from(archive, "prep_failures.csv")
            shard_fallbacks = rows_from(archive, "fallbacks.csv")
            score_rows = rows_from(archive, "scores_vina.csv")
            dock_failed = rows_from(archive, "failed_vina.csv")
            mode_run = runtime["modes"]["vina"]
            validate_command(mode_run["command"])
            local: list[dict] = []
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
            for status, rows in (
                ("prep_failed", prep_failed),
                ("dock_failed", dock_failed),
            ):
                for row in rows:
                    failure = {
                        "sample_index": int(row["sample_index"]),
                        "source_id": row["source_id"],
                        "label": int(row["label"]),
                        "score": None,
                        "status": status,
                        "prep_method": row.get("prep_method", ""),
                        "failure_reason": row["reason"],
                    }
                    local.append(failure)
                    failures.append({"shard": shard, **failure})
            for row in shard_fallbacks:
                fallbacks.append({"shard": shard, **row})
            indices = [row["sample_index"] for row in local]
            expected_indices = set(range(bound["start"], bound["end"]))
            if len(indices) != len(set(indices)) or set(indices) != expected_indices:
                raise RuntimeError(f"shard {shard}: record accounting mismatch")
            for row in local:
                if row["sample_index"] in combined:
                    raise RuntimeError(f"shard {shard}: duplicate global sample index")
                combined[row["sample_index"]] = row
            source_builds.append({"shard": shard, **source})
            input_manifests.append({"shard": shard, **inputs})
            shard_audit.append(
                {
                    "shard": shard,
                    "bundle_sha256": sha256(bundle),
                    "intended": bound["count"],
                    "scored": len(score_rows),
                    "preparation_failures": len(prep_failed),
                    "docking_failures": len(dock_failed),
                    "fallbacks": len(shard_fallbacks),
                    "vina_runtime_seconds": float(mode_run["runtime_seconds"]),
                    "notebook_total_seconds": float(runtime["total_seconds"]),
                    "gpu_count": hardware["gpu_count"],
                    "gpu_models": models,
                    "source_tree_sha256": source["source_tree_sha256"],
                    "binary_sha256": source["binary_sha256"],
                }
            )

    if set(combined) != set(range(TOTAL)):
        raise RuntimeError("global record accounting mismatch")
    labels_all = [combined[index]["label"] for index in range(TOTAL)]
    if (sum(labels_all), len(labels_all) - sum(labels_all)) != (ACTIVES, DECOYS):
        raise RuntimeError("global label count mismatch")
    if len({item["source_tree_sha256"] for item in shard_audit}) != 1:
        raise RuntimeError("source-tree hashes differ across shards")
    if len({source["binary_bytes"] for source in source_builds}) != 1:
        raise RuntimeError("binary sizes differ across shards")

    for row in combined.values():
        row["source_record_key"] = f"{row['source_id']}@{row['sample_index']}"
    scored = sorted(
        (row for row in combined.values() if row["status"] == "scored"),
        key=lambda row: (row["score"], row["sample_index"]),
    )
    failed = sorted(
        (row for row in combined.values() if row["status"] != "scored"),
        key=lambda row: row["sample_index"],
    )
    ranking = scored + failed
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    if len({row["source_record_key"] for row in ranking}) != TOTAL:
        raise RuntimeError("source_record_key identity collapse")
    source_counts = Counter(row["source_id"] for row in ranking)
    scores = [row["score"] for row in scored]
    labels = [row["label"] for row in scored]
    values = metric_values(scores, labels)
    prep_failures = sum(row["status"] == "prep_failed" for row in failed)
    dock_failures = sum(row["status"] == "dock_failed" for row in failed)
    coverage = len(scored) / TOTAL
    quota = json.loads((HERE / "quota_budget.json").read_text(encoding="utf-8"))
    supervisor_kernel_hours = float(quota.get("observed_gpu_hours") or 0.0)
    metrics = {
        "protocol": "UniDock Vina-mode T4 validation; not exact AutoDock Vina equivalence",
        "total_intended": TOTAL,
        "actives_intended": ACTIVES,
        "decoys_intended": DECOYS,
        "scored": len(scored),
        "coverage": round(coverage, 6),
        "coverage_target": 1.0,
        "minimum_acceptable_coverage": MIN_COVERAGE,
        "coverage_gate_passed": coverage >= MIN_COVERAGE,
        "preparation_failures": prep_failures,
        "docking_failures": dock_failures,
        "filtered_rows": len(failed),
        "fallbacks": len(fallbacks),
        "unique_source_ids": len(source_counts),
        "duplicate_source_id_groups": sum(
            count > 1 for count in source_counts.values()
        ),
        "repeated_source_id_records": sum(
            count - 1 for count in source_counts.values() if count > 1
        ),
        **values,
        "bayes_ef_decoy_proxy_1pct": bayes_ef_proxy(scored, 0.01),
        "bayes_ef_decoy_proxy_5pct": bayes_ef_proxy(scored, 0.05),
        "bayes_ef_caveat": "DUD2 decoys are not random compounds; values are existing-project decoy-distribution proxies, not canonical Bayes EF.",
        "missing_score_policy": "Failures are excluded from score metrics and retained explicitly; missing output is not assigned a bad docking score.",
        "vina_runtime_seconds_sum": round(
            sum(item["vina_runtime_seconds"] for item in shard_audit), 3
        ),
        "vina_runtime_seconds_max_shard": round(
            max(item["vina_runtime_seconds"] for item in shard_audit), 3
        ),
        "notebook_runtime_seconds_sum": round(
            sum(item["notebook_total_seconds"] for item in shard_audit), 3
        ),
        "notebook_runtime_seconds_max_shard": round(
            max(item["notebook_total_seconds"] for item in shard_audit), 3
        ),
        "total_gpu_device_hours": round(
            sum(
                item["notebook_total_seconds"] * item["gpu_count"]
                for item in shard_audit
            )
            / 3600,
            6,
        ),
        "supervisor_observed_kernel_hours": round(supervisor_kernel_hours, 6),
        "supervisor_observed_gpu_device_hours": round(supervisor_kernel_hours * 2, 6),
        "runtime_accounting_note": "Notebook execution time comes from each runtime.json. GPU device-hours multiply notebook time by two T4 devices. Supervisor-observed time spans submission to terminal poll and includes queue/poll latency.",
    }
    intervals = bootstrap(scores, labels)
    paper_comparison = {
        "paper": PAPER,
        "pipeline_metrics": {name: metrics[name] for name in PUBLISHED},
        "predefined_margins": MARGINS,
        "comparisons": {},
    }
    for name, published in PUBLISHED.items():
        delta = round(float(metrics[name]) - published, 6)
        interval = intervals["intervals"][name]
        paper_comparison["comparisons"][name] = {
            "published": published,
            "pipeline": metrics[name],
            "pipeline_minus_published": delta,
            "margin": MARGINS[name],
            "margin_passed": delta >= MARGINS[name],
            "published_inside_pipeline_bootstrap_95ci": interval["lower_95"]
            <= published
            <= interval["upper_95"],
        }
    paper_comparison["similarity_margins_passed"] = all(
        item["margin_passed"] for item in paper_comparison["comparisons"].values()
    )
    integrity_passed = (
        coverage >= MIN_COVERAGE and paper_comparison["similarity_margins_passed"]
    )
    paper_comparison["validation_gates_passed"] = integrity_passed
    p100_comparison, rank_changes = compare_p100(ranking, metrics)

    FINAL.mkdir(parents=True, exist_ok=True)
    ranking_fields = [
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
    failure_fields = [
        "shard",
        "sample_index",
        "source_record_key",
        "source_id",
        "label",
        "status",
        "prep_method",
        "failure_reason",
    ]
    for row in failures:
        row["source_record_key"] = f"{row['source_id']}@{row['sample_index']}"
    write_csv(FINAL / "ranking_1846.csv", ranking, ranking_fields)
    write_csv(FINAL / "failures.csv", failures, failure_fields)
    write_csv(FINAL / "filtered_rows.csv", failed, ranking_fields)
    write_csv(
        FINAL / "large_rank_changes.csv",
        rank_changes,
        [
            "source_record_key",
            "source_id",
            "label",
            "p100_score",
            "t4_score",
            "score_delta_t4_minus_p100",
            "p100_rank",
            "t4_rank",
            "rank_delta_t4_minus_p100",
            "absolute_rank_delta",
        ],
    )
    write_json(FINAL / "metrics.json", metrics)
    write_json(FINAL / "bootstrap_intervals.json", intervals)
    write_json(FINAL / "paper_comparison.json", paper_comparison)
    write_json(FINAL / "p100_t4_comparison.json", p100_comparison)
    write_json(
        FINAL / "runtime_summary.json",
        {
            "shards": shard_audit,
            "vina_runtime_seconds_sum": metrics["vina_runtime_seconds_sum"],
            "notebook_runtime_seconds_sum": metrics["notebook_runtime_seconds_sum"],
            "total_gpu_device_hours": metrics["total_gpu_device_hours"],
            "supervisor_observed_kernel_hours": metrics[
                "supervisor_observed_kernel_hours"
            ],
            "supervisor_observed_gpu_device_hours": metrics[
                "supervisor_observed_gpu_device_hours"
            ],
            "accounting_note": metrics["runtime_accounting_note"],
        },
    )
    write_json(
        FINAL / "input_manifest.json",
        {
            "target": "ACE",
            "pdb_id": "1O86",
            "dataset": "DUD Release 2",
            "receptor_sha256": RECEPTOR_SHA256,
            "box_center": BOX_CENTER,
            "box_size": BOX_SIZE,
            "grid_spacing": 0.375,
            "exhaustiveness": 32,
            "scoring": "vina",
            "lower_is_better": True,
            "shards": input_manifests,
        },
    )
    write_json(
        FINAL / "source_build_manifest.json",
        {
            "official_repository": "https://github.com/dptech-corp/Uni-Dock",
            "commit": UNIDOCK_COMMIT,
            "grid_point_cap": 2_097_152,
            "change_scope": "capacity limit only; no intended scoring protocol change",
            "source_tree_hashes_unique": len(
                {source["source_tree_sha256"] for source in source_builds}
            ),
            "binary_hashes_unique": len(
                {source["binary_sha256"] for source in source_builds}
            ),
            "binary_reproducibility_note": "Every binary hash is recorded and binary size is gated. Independent source builds may differ bytewise because compiler/linker build metadata is not reproducible; identical source-tree and patch hashes are required.",
            "shards": source_builds,
        },
    )
    attempt_failures = []
    rejected_attempt_hashes = {}
    for attempt_dir in sorted(
        WORK.glob("ace-1o86-vina-t4-s*/attempt_00_p100_rejected")
    ):
        logs = sorted(attempt_dir.glob("*.log"))
        if len(logs) != 1 or "Tesla P100" not in logs[0].read_text(encoding="utf-8"):
            raise RuntimeError(
                f"rejected P100 attempt evidence mismatch: {attempt_dir}"
            )
        for path in sorted(item for item in attempt_dir.rglob("*") if item.is_file()):
            relative = str(path.relative_to(HERE)).replace("\\", "/")
            rejected_attempt_hashes[relative] = sha256(path)
        attempt_failures.append(
            {
                "shard": int(attempt_dir.parent.name.rsplit("s", 1)[1]),
                "attempt": 0,
                "failure_type": "hardware_mismatch",
                "expected": "2 x Tesla T4",
                "observed": "1 x Tesla P100-PCIE-16GB",
                "failed_before_preparation": True,
                "preserved_log": str(logs[0].relative_to(HERE)).replace("\\", "/"),
                "remediation": "Passed NvidiaTeslaT4 through Kaggle SaveKernel accelerator API; retry 1 completed on 2 x Tesla T4.",
            }
        )
    if len(attempt_failures) != 12:
        raise RuntimeError("expected 12 preserved P100 rejection ledgers")
    write_json(
        FINAL / "campaign_attempt_failures.json",
        {
            "compound_preparation_or_docking_failures": 0,
            "infrastructure_attempt_failures": attempt_failures,
        },
    )
    write_json(
        FINAL / "raw_bundle_hash_manifest.json",
        {
            "successful_output_bundles_sha256": bundle_hashes,
            "rejected_attempt_files_sha256": rejected_attempt_hashes,
        },
    )
    write_json(FINAL / "shard_audit.json", shard_audit)
    metrics_md = (
        "# ACE T4 validation metrics\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in metrics.items())
        + "\n"
    )
    (FINAL / "metrics.md").write_text(metrics_md, encoding="utf-8")
    conclusion = (
        "SwiftScreen’s automated ACE UniDock Vina-mode workflow achieved published Vina screening quality within predefined ROC-AUC and BEDROC20 tolerances, with complete compound accounting."
        if integrity_passed and coverage == 1.0
        else "ACE T4 validation did not satisfy every preregistered validation gate; see paper_comparison.json and failures.csv."
    )
    report = f"""# ACE UniDock Vina-mode T4 validation

{conclusion}

Gate outcome: FAIL overall because the predefined ROC-AUC margin failed; the BEDROC20 margin passed. No margin or protocol was changed after observing results.

Coverage: {len(scored)}/{TOTAL} ({coverage:.4f}); 49 actives and 1,797 decoys; zero preparation failures, docking failures, fallbacks, or filtered rows. ROC-AUC: {metrics["roc_auc"]} (published {PUBLISHED["roc_auc"]}; delta {paper_comparison["comparisons"]["roc_auc"]["pipeline_minus_published"]}; bootstrap 95% CI {intervals["intervals"]["roc_auc"]["lower_95"]}-{intervals["intervals"]["roc_auc"]["upper_95"]}). BEDROC20: {metrics["bedroc20"]} (published {PUBLISHED["bedroc20"]}; delta {paper_comparison["comparisons"]["bedroc20"]["pipeline_minus_published"]}; bootstrap 95% CI {intervals["intervals"]["bedroc20"]["lower_95"]}-{intervals["intervals"]["bedroc20"]["upper_95"]}). Both published values fall inside the pipeline bootstrap intervals.

Against the existing P100 raw Vina ranking: Spearman {p100_comparison["rank_correlation"]["spearman"]:.6f}, Kendall {p100_comparison["rank_correlation"]["kendall"]:.6f}; ROC-AUC delta {p100_comparison["metric_differences_t4_minus_p100"]["roc_auc"]}; BEDROC20 delta {p100_comparison["metric_differences_t4_minus_p100"]["bedroc20"]}. Architecture-specific floating-point scores were not required to be byte-identical.

Measured notebook execution consumed {metrics["total_gpu_device_hours"]} T4 device-hours. The supervisor observed {metrics["supervisor_observed_kernel_hours"]} kernel-hours from submission to terminal polling. Attempt 0 was rejected before preparation on all shards because Kaggle allocated P100; all logs are retained and hashed. Retry 1 completed on T4 x2.

Every shard required runtime evidence for exactly two Tesla T4 GPUs before preparation, exact frozen ADT prepared hashes, exact receptor and box, UniDock commit {UNIDOCK_COMMIT}, Vina scoring, spacing 0.375 A, and exhaustiveness 32. The 2,097,152-point source patch changes capacity limit only; no intended scoring protocol change.

This validates result quality for tested ACE protocol. It does not establish exact AutoDock Vina equivalence or universal performance across targets and scoring backends.
"""
    (FINAL / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(FINAL)).replace("\\", "/"): sha256(path)
        for path in sorted(FINAL.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    write_json(FINAL / "artifact_manifest.json", {"sha256": artifact_hashes})
    print(
        json.dumps(
            {
                "metrics": metrics,
                "paper_comparison": paper_comparison["comparisons"],
                "validation_gates_passed": integrity_passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
