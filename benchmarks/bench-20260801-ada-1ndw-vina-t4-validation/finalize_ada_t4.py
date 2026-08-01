"""Finalize the untouched ADA/1NDW UniDock Vina-mode T4 validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from ezscreen.benchmark.metrics import compute_metrics

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
FINAL = HERE / "final_results"
PREPARED_INPUT = HERE / "prepared_input_manifest.json"

TOTAL, ACTIVES, DECOYS, SHARDS = 966, 39, 927, 12
MIN_COVERAGE = 0.98
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260801
UNIDOCK_COMMIT = "ebde3b4a22c8b2c7b26e523cc15109aab815be21"
BOX_CENTER = [42.2931, 17.3816, 49.5368]
BOX_SIZE = [36.9618, 39.689, 38.3826]
PUBLISHED = {"roc_auc": 0.451, "bedroc20": 0.005}
MARGINS = {"roc_auc": -0.02, "bedroc20": -0.02}
PAPER = {
    "title": "Vinardo: A Scoring Function Based on Autodock Vina Improves Scoring, Docking, and Virtual Screening",
    "authors": ["Rodrigo Quiroga", "Marcos A. Villarreal"],
    "journal": "PLOS ONE 11(5): e0155183 (2016)",
    "doi": "10.1371/journal.pone.0155183",
    "primary_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155183",
    "protocol_evidence": {
        "location": "Methods sections 2.2 and 2.5",
        "description": "DUD Release 2; smina; exhaustiveness 32; bound-ligand-centered box extended 15 A; Vina scoring; Open Babel conversion and AutoDockTools preparation.",
    },
    "screening_evidence": {
        "location": "Results section 3.5 and Supplementary Tables S2-S4",
        "s2_ada": ["ada", "1ndw", 39, 927],
        "s3_vina_roc_auc": 0.451,
        "s4_vina_bedroc": 0.005,
    },
    "ambiguities": [
        "Paper does not report AutoDockTools, Open Babel, or smina versions.",
        "Paper does not report a random seed or complete command line.",
        "Supplementary Table S4 reports BEDROC without alpha; BEDROC20 is the project's alpha=20 inference.",
        "UniDock Vina mode is not claimed to be engine-equivalent to paper smina/Vina.",
        "ADA required one explicit blank Zn element-column annotation before AutoDockTools; no atom or coordinate changed.",
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


def execution_failure_audit() -> tuple[list[dict], list[dict], float]:
    specs = [
        (
            "preparation",
            "ada-1ndw-prepare-t4",
            WORK / "ada-1ndw-prepare-t4" / "attempt_00_adt_zinc_element_failure",
            "ADT rejected the source receptor because the existing Zn atom had blank PDB element columns; no prepared campaign input was accepted.",
        )
    ]
    specs.extend(
        (
            "docking",
            f"ada-1ndw-vina-t4-s{shard:02d}",
            WORK / f"ada-1ndw-vina-t4-s{shard:02d}" / "attempt_00_kaggle_zip_auto_expansion",
            "Kaggle auto-expanded prepared_bundle.zip, so the exact bundle hash gate failed before docking and no score was emitted.",
        )
        for shard in range(SHARDS)
    )
    attempts, raw_files, elapsed_sum = [], [], 0.0
    for stage, job_id, directory, reason in specs:
        if not directory.is_dir():
            raise RuntimeError(f"missing preserved failed attempt: {directory}")
        hardware = json.loads((directory / "hardware_manifest.json").read_text())
        if hardware.get("gpu_models") != ["Tesla T4", "Tesla T4"]:
            raise RuntimeError(f"failed attempt hardware mismatch: {job_id}")
        logs = list(directory.glob("*.log"))
        if len(logs) != 1:
            raise RuntimeError(f"expected one raw failed-attempt log: {job_id}")
        log_text = logs[0].read_text(errors="replace")
        elapsed_values = [float(value) for value in re.findall(r'"time":\s*([0-9]+(?:\.[0-9]+)?)', log_text)]
        if not elapsed_values:
            raise RuntimeError(f"missing raw Kaggle elapsed timestamps: {job_id}")
        elapsed_seconds = max(elapsed_values)
        elapsed_sum += elapsed_seconds
        attempts.append({
            "attempt": 0,
            "stage": stage,
            "job_id": job_id,
            "reason": reason,
            "gpu_models": "Tesla T4|Tesla T4",
            "elapsed_seconds_from_raw_log": round(elapsed_seconds, 6),
            "compound_scores_emitted": 0,
            "preserved_directory": directory.relative_to(HERE).as_posix(),
        })
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            raw_files.append({
                "attempt": 0,
                "stage": stage,
                "job_id": job_id,
                "path": path.relative_to(HERE).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            })
    return attempts, raw_files, elapsed_sum


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


def metric_values(scores: list[float], labels: list[int]) -> dict:
    result = compute_metrics(scores, labels)
    ranked = [
        label
        for _score, label in sorted(
            zip(scores, labels), key=lambda item: item[0]
        )
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
    names = ("roc_auc", "bedroc20", "pr_auc", "average_precision", "ef1", "ef5", "ef10", "ef20")
    samples = {name: np.empty(BOOTSTRAP_REPLICATES) for name in names}
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = np.concatenate((rng.choice(active, len(active), replace=True), rng.choice(decoy, len(decoy), replace=True)))
        values = metric_values(score_array[indices].tolist(), label_array[indices].tolist())
        for name in names:
            samples[name][replicate] = float(values[name])
    return {
        "method": "stratified nonparametric bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence": 0.95,
        "intervals": {
            name: {
                "lower_95": round(float(np.quantile(values, 0.025)), 6),
                "upper_95": round(float(np.quantile(values, 0.975)), 6),
                "median": round(float(np.median(values)), 6),
            }
            for name, values in samples.items()
        },
    }


def validate_command(log_text: str) -> None:
    expected = {
        "--center_x": "42.2931", "--center_y": "17.3816", "--center_z": "49.5368",
        "--size_x": "36.9618", "--size_y": "39.689", "--size_z": "38.3826",
        "--spacing": "0.375", "--scoring": "vina", "--exhaustiveness": "32",
    }
    first_line = log_text.splitlines()[0] if log_text.splitlines() else ""
    if "--receptor" not in first_line or "--maps" in first_line:
        raise RuntimeError("UniDock receptor command mismatch")
    for option, value in expected.items():
        if f"{option} {value}" not in first_line:
            raise RuntimeError(f"UniDock command mismatch for {option}")


def main() -> None:
    frozen = json.loads(PREPARED_INPUT.read_text(encoding="utf-8"))
    expected_counts = {
        "intended": TOTAL, "actives": ACTIVES, "decoys": DECOYS,
        "prepared": TOTAL, "preparation_failures": 0, "fallbacks": 0,
    }
    if frozen["records"] != expected_counts:
        raise RuntimeError("frozen prepared-input counts drifted")
    prep_bundle = WORK / "ada-1ndw-prepare-t4/output/output.zip"
    if not prep_bundle.is_file() or sha256(prep_bundle) != frozen["prep_output_zip_sha256"]:
        raise RuntimeError("preparation output bundle hash mismatch")

    scored: list[dict] = []
    docking_failures: list[dict] = []
    assigned_prep_failures: list[dict] = []
    shard_audit, runtimes = [], []
    raw_hashes = [{
        "stage": "preparation", "job_id": "ada-1ndw-prepare-t4",
        "path": prep_bundle.relative_to(HERE).as_posix(), "sha256": sha256(prep_bundle),
        "bytes": prep_bundle.stat().st_size,
    }]
    source_tree_hashes, binary_hashes, patched_file_hashes = set(), set(), set()
    intended_indices: list[int] = []
    for shard in range(SHARDS):
        output = WORK / f"ada-1ndw-vina-t4-s{shard:02d}/output/output.zip"
        if not output.is_file():
            raise FileNotFoundError(output)
        raw_hashes.append({
            "stage": "docking", "job_id": f"ada-1ndw-vina-t4-s{shard:02d}",
            "path": output.relative_to(HERE).as_posix(), "sha256": sha256(output),
            "bytes": output.stat().st_size,
        })
        with zipfile.ZipFile(output) as archive:
            required = {
                "scores_vina.csv", "failed_vina.csv", "prep_failures_assigned.csv",
                "unidock_vina.log", "runtime.json", "hardware_manifest.json",
                "source_build_manifest.json", "input_manifest.json",
            }
            missing = required - set(archive.namelist())
            if missing:
                raise RuntimeError(f"shard {shard}: missing {sorted(missing)}")
            hardware = json_from(archive, "hardware_manifest.json")
            source = json_from(archive, "source_build_manifest.json")
            input_manifest = json_from(archive, "input_manifest.json")
            runtime = json_from(archive, "runtime.json")
            scores = rows_from(archive, "scores_vina.csv")
            failed = rows_from(archive, "failed_vina.csv")
            prep_failed = rows_from(archive, "prep_failures_assigned.csv")
            validate_command(archive.read("unidock_vina.log").decode(errors="replace"))
        if not hardware.get("gate_passed") or hardware.get("gpu_count") != 2 or hardware.get("gpu_models") != ["Tesla T4", "Tesla T4"]:
            raise RuntimeError(f"shard {shard}: T4x2 hardware proof failed")
        if source.get("git_commit") != UNIDOCK_COMMIT:
            raise RuntimeError(f"shard {shard}: UniDock commit mismatch")
        if source.get("grid_point_cap") != 2_097_152 or not source.get("capacity_only_change"):
            raise RuntimeError(f"shard {shard}: grid-cap provenance mismatch")
        for key in ("prepared_bundle_sha256", "prepared_manifest_sha256", "receptor_pdbqt_sha256"):
            if input_manifest.get(key) != frozen[key]:
                raise RuntimeError(f"shard {shard}: frozen {key} mismatch")
        start, end = shard * TOTAL // SHARDS, (shard + 1) * TOTAL // SHARDS
        if input_manifest.get("range") != [start, end] or runtime.get("range") != [start, end]:
            raise RuntimeError(f"shard {shard}: range mismatch")
        if runtime.get("scoring") != "vina" or runtime.get("exhaustiveness") != 32 or runtime.get("grid_spacing") != 0.375:
            raise RuntimeError(f"shard {shard}: scoring protocol mismatch")
        if runtime.get("box_center") != BOX_CENTER or runtime.get("box_size") != BOX_SIZE:
            raise RuntimeError(f"shard {shard}: box mismatch")
        if runtime.get("intended") != end - start or len(scores) + len(failed) + len(prep_failed) != end - start:
            raise RuntimeError(f"shard {shard}: record accounting mismatch")
        intended_indices.extend(range(start, end))
        source_tree_hashes.add(source["source_tree_sha256"])
        binary_hashes.add(source["binary_sha256"])
        patched_file_hashes.update(source["patched_files"].values())
        for row in scores:
            score, index = float(row["score"]), int(row["sample_index"])
            if not math.isfinite(score):
                raise RuntimeError(f"shard {shard}: non-finite score")
            if row["record_id"] != f"{row['source_id']}@{index}":
                raise RuntimeError(f"shard {shard}: record identity mismatch")
            scored.append({**row, "sample_index": index, "label": int(row["label"]), "score": score, "shard": shard})
        docking_failures.extend({**row, "stage": "docking", "shard": shard} for row in failed)
        assigned_prep_failures.extend({**row, "stage": "preparation", "shard": shard} for row in prep_failed)
        runtimes.append(runtime)
        shard_audit.append({
            "shard": shard, "range": [start, end], "intended": end - start,
            "scored": len(scores), "docking_failures": len(failed),
            "preparation_failures": len(prep_failed), "gpu_models": hardware["gpu_models"],
            "source_tree_sha256": source["source_tree_sha256"],
            "binary_sha256": source["binary_sha256"], "raw_bundle_sha256": sha256(output),
        })

    if intended_indices != list(range(TOTAL)):
        raise RuntimeError("global shard ranges do not cover every ADA index")
    all_failures = assigned_prep_failures + docking_failures
    if len(scored) + len(all_failures) != TOTAL:
        raise RuntimeError("global score/failure accounting mismatch")
    score_indices = [row["sample_index"] for row in scored]
    failure_indices = [int(row["sample_index"]) for row in all_failures]
    if sorted(score_indices + failure_indices) != list(range(TOTAL)) or len(set(score_indices + failure_indices)) != TOTAL:
        raise RuntimeError("global index identity mismatch")
    labels = Counter(row["label"] for row in scored)
    labels.update(int(row["label"]) for row in all_failures)
    if labels != Counter({0: DECOYS, 1: ACTIVES}):
        raise RuntimeError(f"global label count mismatch: {labels}")
    coverage = len(scored) / TOTAL
    if coverage < MIN_COVERAGE:
        raise RuntimeError(f"coverage below {MIN_COVERAGE}: {coverage}")

    scored.sort(key=lambda row: (row["score"], row["sample_index"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    scores = [row["score"] for row in scored]
    score_labels = [row["label"] for row in scored]
    metrics = metric_values(scores, score_labels)
    metrics.update({
        "target": "ADA", "pdb_id": "1NDW", "intended_records": TOTAL,
        "scored_records": len(scored), "coverage": coverage,
        "preparation_failures": len(assigned_prep_failures),
        "docking_failures": len(docking_failures), "fallbacks": frozen["records"]["fallbacks"],
        "filtered_rows": 0, "bayes_ef_decoy_proxy_1pct": bayes_ef_proxy(scored, 0.01),
        "bayes_ef_decoy_proxy_5pct": bayes_ef_proxy(scored, 0.05),
        "bayes_ef_note": "DUD2 decoys are property-matched, not random background; values are project decoy-threshold proxies.",
    })
    intervals = bootstrap(scores, score_labels)
    comparisons = {}
    for name, published in PUBLISHED.items():
        observed = float(metrics[name])
        difference = observed - published
        interval = intervals["intervals"][name]
        comparisons[name] = {
            "pipeline": observed, "published": published,
            "pipeline_minus_published": round(difference, 6),
            "predefined_margin": MARGINS[name], "margin_pass": difference >= MARGINS[name],
            "published_inside_pipeline_bootstrap_95ci": interval["lower_95"] <= published <= interval["upper_95"],
        }
    overall_pass = coverage == 1.0 and not all_failures and all(item["margin_pass"] for item in comparisons.values())

    paper_comparison = {
        "paper": PAPER, "published_values": PUBLISHED,
        "predefined_similarity_margins": MARGINS, "comparisons": comparisons,
        "integrity_gates": {
            "complete_accounting": len(scored) + len(all_failures) == TOTAL,
            "full_coverage": coverage == 1.0,
            "no_preparation_failures": not assigned_prep_failures,
            "no_docking_failures": not docking_failures,
            "no_fallbacks": frozen["records"]["fallbacks"] == 0,
            "all_shards_t4x2": all(audit["gpu_models"] == ["Tesla T4", "Tesla T4"] for audit in shard_audit),
            "exact_frozen_prepared_bundle": True, "exact_ada_box": True,
            "overall_validation_pass": overall_pass,
        },
        "allowed_success_conclusion": (
            "SwiftScreen's automated ADA UniDock Vina-mode workflow achieved published Vina screening quality within predefined ROC-AUC and BEDROC20 tolerances, with complete compound accounting."
            if overall_pass else None
        ),
        "required_caveat": "This validates result quality for tested ADA protocol. It does not establish exact AutoDock Vina equivalence or universal performance across targets and scoring backends.",
    }
    execution_failures, failed_attempt_raw_hashes, failed_attempt_seconds = execution_failure_audit()
    prep_seconds = float(frozen["prep_runtime"]["total_seconds"])
    docking_seconds = sum(float(item["docking_seconds"]) for item in runtimes)
    notebook_seconds = sum(float(item["total_seconds"]) for item in runtimes)
    successful_device_hours = (prep_seconds + notebook_seconds) * 2 / 3600
    failed_attempt_device_hours = failed_attempt_seconds * 2 / 3600
    runtime_summary = {
        "preparation_seconds": prep_seconds, "docking_seconds_sum": docking_seconds,
        "docking_notebook_seconds_sum": notebook_seconds,
        "total_notebook_seconds_including_preparation": prep_seconds + notebook_seconds,
        "successful_t4_device_hours": round(successful_device_hours, 6),
        "failed_attempt_notebook_seconds_sum": round(failed_attempt_seconds, 6),
        "failed_attempt_t4_device_hours": round(failed_attempt_device_hours, 6),
        "total_observed_t4_device_hours": round(successful_device_hours + failed_attempt_device_hours, 6),
        "measured_t4_device_hours": round(successful_device_hours + failed_attempt_device_hours, 6),
        "gpu_count_per_job": 2,
        "shards": [{
            "shard": item["shard_index"], "docking_seconds": item["docking_seconds"],
            "total_seconds": item["total_seconds"],
        } for item in runtimes],
        "accounting_note": "Total observed device-hours include successful notebooks plus all 13 preserved failed attempts, multiplying raw notebook elapsed time by two observed T4 devices; excludes queue and download time.",
    }

    FINAL.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL / "ranking_966.csv", scored, [
        "rank", "record_id", "sample_index", "source_id", "label", "score", "prep_method", "shard",
    ])
    write_csv(FINAL / "failures.csv", all_failures, [
        "record_id", "sample_index", "source_id", "label", "stage", "shard", "raw_score", "reason",
    ])
    write_csv(FINAL / "filtered_rows.csv", [], ["record_id", "reason"])
    write_json(FINAL / "metrics.json", metrics)
    write_json(FINAL / "bootstrap_intervals.json", intervals)
    write_json(FINAL / "paper_comparison.json", paper_comparison)
    write_json(FINAL / "runtime_summary.json", runtime_summary)
    write_json(FINAL / "shard_audit.json", shard_audit)
    write_json(FINAL / "input_manifest.json", {
        "frozen_prepared_input": frozen, "box_center": BOX_CENTER, "box_size": BOX_SIZE,
        "grid_spacing": 0.375, "scoring": "vina", "exhaustiveness": 32,
        "record_identity": "source_id@sample_index",
    })
    write_json(FINAL / "source_build_manifest.json", {
        "official_repository": "https://github.com/dptech-corp/Uni-Dock",
        "commit": UNIDOCK_COMMIT, "grid_point_cap": 2_097_152,
        "capacity_only_change": True, "source_tree_sha256": sorted(source_tree_hashes),
        "binary_sha256": sorted(binary_hashes), "patched_file_sha256": sorted(patched_file_hashes),
        "binary_hash_note": "Independent source builds may contain nondeterministic build metadata; every hash is retained.",
    })
    write_json(FINAL / "raw_bundle_hash_manifest.json", raw_hashes)
    write_csv(FINAL / "execution_attempt_failures.csv", execution_failures, [
        "attempt", "stage", "job_id", "reason", "gpu_models",
        "elapsed_seconds_from_raw_log", "compound_scores_emitted", "preserved_directory",
    ])
    write_json(FINAL / "failed_attempt_raw_hash_manifest.json", failed_attempt_raw_hashes)

    metrics_md = f"""# ADA/1NDW T4 validation metrics

- Coverage: {len(scored)}/{TOTAL} ({coverage:.6f})
- EF1 / EF5 / EF10 / EF20: {metrics['ef1']:.6f} / {metrics['ef5']:.6f} / {metrics['ef10']:.6f} / {metrics['ef20']:.6f}
- ROC-AUC: {metrics['roc_auc']:.6f}
- BEDROC20: {metrics['bedroc20']:.6f}
- PR-AUC: {metrics['pr_auc']:.6f}
- Average precision: {metrics['average_precision']:.6f}
- Top-active rank: {metrics['top_active_rank']}
- Preparation failures / docking failures / fallbacks / filtered: {len(assigned_prep_failures)} / {len(docking_failures)} / {frozen['records']['fallbacks']} / 0
"""
    (FINAL / "metrics.md").write_text(metrics_md, encoding="utf-8")
    disposition = (
        "PASS: the untouched ADA T4 workflow satisfied both predefined similarity margins and every integrity gate."
        if overall_pass
        else "FAIL: the untouched ADA T4 workflow did not satisfy every predefined similarity margin and integrity gate."
    )
    report = f"""# ADA/1NDW fresh-target T4 validation

## Outcome

{disposition}

## Accounting

- Intended: {TOTAL} ({ACTIVES} actives, {DECOYS} decoys)
- Scored: {len(scored)}; coverage: {coverage:.6f}
- Preparation failures / docking failures / fallbacks / filtered: {len(assigned_prep_failures)} / {len(docking_failures)} / {frozen['records']['fallbacks']} / 0

## Published comparison

- ROC-AUC: pipeline {metrics['roc_auc']:.6f}; published 0.451; difference {comparisons['roc_auc']['pipeline_minus_published']:.6f}; margin pass {comparisons['roc_auc']['margin_pass']}.
- BEDROC20: pipeline {metrics['bedroc20']:.6f}; published 0.005; difference {comparisons['bedroc20']['pipeline_minus_published']:.6f}; margin pass {comparisons['bedroc20']['margin_pass']}.
- The paper reports BEDROC without alpha; alpha=20 remains an explicit project inference.

## Protocol and provenance

- AutoDockTools preparation first; exact frozen prepared bundle `{frozen['prepared_bundle_sha256']}`.
- One blank Zn element field annotated; no atom or coordinate changed.
- Box center `{BOX_CENTER}`; size `{BOX_SIZE}`; spacing 0.375 A; exhaustiveness 32; raw score ascending.
- All 12 docking shards proved 2x Tesla T4.
- UniDock commit `{UNIDOCK_COMMIT}`; 2,097,152-point capacity patch only.
- Measured T4 device-hours: {runtime_summary['measured_t4_device_hours']:.6f}.
- This total includes {len(execution_failures)} preserved failed execution attempts; each emitted zero compound scores. See `execution_attempt_failures.csv`.

## Scope caveat

This validates result quality for tested ADA protocol. It does not establish exact AutoDock Vina equivalence or universal performance across targets and scoring backends.
"""
    (FINAL / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    artifact_rows = [{
        "filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
    } for path in sorted(FINAL.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"]
    write_json(FINAL / "artifact_manifest.json", artifact_rows)
    print(json.dumps({
        "overall_validation_pass": overall_pass, "coverage": coverage,
        "metrics": metrics, "final_results": str(FINAL),
    }, indent=2))


if __name__ == "__main__":
    main()
