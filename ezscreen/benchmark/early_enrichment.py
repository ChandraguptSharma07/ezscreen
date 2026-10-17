from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ezscreen.benchmark.metrics import compute_metrics

ARM_IDS = (
    "B0_frozen_baseline",
    "L1_meeko_single_state",
    "L2_bounded_state_ensemble",
    "S1_budget_matched_replicates",
    "C1_states_plus_replicates",
)
_SOURCE_KEY_RE = re.compile(r"^.+@\d+$")


class ProtocolContractError(ValueError):
    """Raised when the preregistered early-enrichment contract is violated."""


def load_protocol_freeze(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol_freeze(document)
    return document


def _arms_by_id(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    arms = document.get("arms")
    if not isinstance(arms, list):
        raise ProtocolContractError("protocol arms must be a list")
    by_id = {str(arm.get("id")): arm for arm in arms if isinstance(arm, Mapping)}
    if tuple(by_id) != ARM_IDS:
        raise ProtocolContractError(
            f"protocol arm IDs/order differ from freeze: {tuple(by_id)!r}"
        )
    return by_id


def sampling_runs(
    document: Mapping[str, Any], arm_id: str
) -> tuple[dict[str, int], ...]:
    arm = _arms_by_id(document).get(arm_id)
    if arm is None:
        raise ProtocolContractError(f"unknown protocol arm: {arm_id}")
    raw = arm.get("searches", arm.get("searches_per_state"))
    if not isinstance(raw, list) or not raw:
        raise ProtocolContractError(f"{arm_id} has no frozen searches")
    runs: list[dict[str, int]] = []
    for index, run in enumerate(raw):
        if not isinstance(run, Mapping):
            raise ProtocolContractError(f"{arm_id} search {index} is not an object")
        try:
            seed = int(run["seed"])
            exhaustiveness = int(run["exhaustiveness"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolContractError(
                f"{arm_id} search {index} lacks integer seed/exhaustiveness"
            ) from exc
        if seed <= 0 or exhaustiveness <= 0:
            raise ProtocolContractError(
                f"{arm_id} search {index} seed/exhaustiveness must be positive"
            )
        runs.append({"seed": seed, "exhaustiveness": exhaustiveness})
    return tuple(runs)


def validate_protocol_freeze(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ProtocolContractError("unsupported protocol schema")
    if document.get("execution", {}).get("launch_authorized") is not False:
        raise ProtocolContractError("design file must not authorize execution")

    engine = document.get("engine", {})
    if engine.get("name") != "Uni-Dock" or engine.get("scoring") != "vina":
        raise ProtocolContractError("engine/scoring must be Uni-Dock/canonical Vina")
    if engine.get("score_direction") != "lower is better":
        raise ProtocolContractError("raw Vina score direction must be lower-is-better")
    validity_policy = str(engine.get("score_validity_policy", "")).lower()
    if "every finite raw vina affinity" not in validity_policy:
        raise ProtocolContractError("freeze must retain every finite raw Vina affinity")

    common = engine.get("common_arguments", {})
    required_common = {
        "spacing",
        "cpu",
        "num_modes",
        "energy_range",
        "min_rmsd",
        "max_evals",
        "max_step",
        "max_gpu_memory",
        "verbosity",
    }
    missing = required_common - set(common)
    if missing:
        raise ProtocolContractError(f"missing frozen Uni-Dock arguments: {sorted(missing)}")

    arms = _arms_by_id(document)
    for arm_id in ARM_IDS:
        sampling_runs(document, arm_id)

    if arms["L1_meeko_single_state"].get("no_fallback") is not True:
        raise ProtocolContractError("L1 must be fail-closed")
    if arms["L2_bounded_state_ensemble"].get("no_fallback") is not True:
        raise ProtocolContractError("L2 must be fail-closed")
    l1_prep = arms["L1_meeko_single_state"].get("ligand_preparation", {})
    if (
        not isinstance(l1_prep, Mapping)
        or l1_prep.get("coordinate_tolerance_angstrom") != 0.01
        or "symmetry" not in str(l1_prep.get("coordinate_mapping", "")).lower()
    ):
        raise ProtocolContractError("L1 coordinate-preservation contract differs from freeze")
    l2_prep = arms["L2_bounded_state_ensemble"].get("ligand_preparation", {})
    if (
        not isinstance(l2_prep, Mapping)
        or l2_prep.get("mmff_convergence_required") is not True
        or l2_prep.get("accepted_mmff_return_codes") != [0]
    ):
        raise ProtocolContractError("L2 MMFF94 convergence contract differs from freeze")
    if arms["C1_states_plus_replicates"].get("no_component_substitution") is not True:
        raise ProtocolContractError("C1 component substitution must be prohibited")
    provenance = document.get("provenance_contract", {})
    required_preparation = {
        "pip_freeze.txt",
        "environment_manifest.json",
        "executable_manifest.json",
        "command_ledger.json",
        "artifact_manifest.json",
        "archive_manifest.json",
    }
    required_docking = {
        "pip_freeze.txt",
        "environment_manifest.json",
        "docking_executable_manifest.json",
        "docking_inputs_manifest.json",
        "docking_command_ledger.json",
        "docking_artifact_manifest.json",
    }
    if (
        provenance.get("log_tail_limit") != 2000
        or set(provenance.get("required_preparation_outputs", []))
        != required_preparation
        or set(provenance.get("required_docking_outputs", [])) != required_docking
        or "Never serialize the full environment"
        not in str(provenance.get("credential_policy", ""))
    ):
        raise ProtocolContractError("provenance contract differs from freeze")
    metrics = document.get("metric_contract", {})
    bootstrap = metrics.get("within_target_uncertainty", {})
    if (
        metrics.get("cutoff_size") != "ceil(N * cutoff_fraction)"
        or "expected active count" not in str(metrics.get("tie_handling", ""))
        or int(bootstrap.get("replicates", 0)) != 10_000
        or int(bootstrap.get("seed", 0)) != 20_260_801
    ):
        raise ProtocolContractError("metric/bootstrap contract differs from freeze")


def preparation_contract(
    document: Mapping[str, Any], arm_id: str
) -> Mapping[str, Any] | str:
    arm = _arms_by_id(document).get(arm_id)
    if arm is None:
        raise ProtocolContractError(f"unknown protocol arm: {arm_id}")
    prep = arm.get("ligand_preparation")
    if not isinstance(prep, (str, Mapping)):
        raise ProtocolContractError(f"{arm_id} lacks a ligand-preparation contract")
    return prep


def render_matrix(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic render specifications; never submits work."""
    validate_protocol_freeze(document)
    common = document["engine"]["common_arguments"]
    specs: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        for search_index, run in enumerate(sampling_runs(document, arm_id)):
            specs.append(
                {
                    "arm_id": arm_id,
                    "search_index": search_index,
                    "render_kwargs": {
                        "engine": "unidock",
                        "scoring": "vina",
                        "prep_on_kaggle": False,
                        "search_mode": "balance",
                        "exhaustiveness": run["exhaustiveness"],
                        "seed": run["seed"],
                        "num_modes": int(common["num_modes"]),
                        "energy_range": float(common["energy_range"]),
                        "min_rmsd": float(common["min_rmsd"]),
                        "max_evals": int(common["max_evals"]),
                        "max_step": int(common["max_step"]),
                        "cpu": int(common["cpu"]),
                        "max_gpu_memory": int(common["max_gpu_memory"]),
                        "verbosity": int(common["verbosity"]),
                        "retain_all_finite_scores": True,
                    },
                }
            )
    return specs


def _finite_score(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def aggregate_raw_vina_scores(
    raw_rows: Iterable[Mapping[str, Any]],
    intended_source_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Collapse states/searches only after retaining strict constituent provenance.

    `source_key` must be `source_id@sample_index`; therefore two records sharing a
    source ID remain distinct. Each source is ranked by its minimum finite raw Vina
    affinity. Missing/non-finite constituents remain recorded as failure reasons.
    """
    intended = [str(key) for key in intended_source_keys]
    if len(set(intended)) != len(intended):
        raise ProtocolContractError("intended source keys must be unique")
    if any(not _SOURCE_KEY_RE.match(key) for key in intended):
        raise ProtocolContractError(
            "every intended source key must use source_id@sample_index"
        )
    intended_set = set(intended)

    successes: dict[str, list[dict[str, Any]]] = {key: [] for key in intended}
    failures: dict[str, list[str]] = {key: [] for key in intended}
    seen_constituents: set[tuple[str, int, int]] = set()

    for row_number, row in enumerate(raw_rows):
        source_key = str(row.get("source_key", ""))
        if source_key not in intended_set:
            raise ProtocolContractError(
                f"raw row {row_number} has unknown source key: {source_key!r}"
            )
        if str(row.get("engine", "")).lower() != "unidock":
            raise ProtocolContractError(f"raw row {row_number} mixes docking engines")
        if str(row.get("scoring", "")).lower() != "vina":
            raise ProtocolContractError(f"raw row {row_number} mixes scoring functions")
        try:
            state_index = int(row.get("state_index", 0))
            search_index = int(row.get("search_index", 0))
        except (TypeError, ValueError) as exc:
            raise ProtocolContractError(
                f"raw row {row_number} has invalid state/search index"
            ) from exc
        constituent = (source_key, state_index, search_index)
        if constituent in seen_constituents:
            raise ProtocolContractError(f"duplicate constituent row: {constituent!r}")
        seen_constituents.add(constituent)

        score = _finite_score(row.get("score"))
        if score is None:
            failures[source_key].append(str(row.get("reason") or "non_finite_or_missing"))
            continue
        successes[source_key].append(
            {
                "score": score,
                "state_index": state_index,
                "search_index": search_index,
            }
        )

    aggregated: list[dict[str, Any]] = []
    for source_key in intended:
        candidates = successes[source_key]
        if candidates:
            selected = min(
                candidates,
                key=lambda row: (
                    row["score"],
                    row["state_index"],
                    row["search_index"],
                ),
            )
            aggregated.append(
                {
                    "source_key": source_key,
                    "score": selected["score"],
                    "status": "scored",
                    "selected_state_index": selected["state_index"],
                    "selected_search_index": selected["search_index"],
                    "finite_constituent_count": len(candidates),
                    "failure_reasons": sorted(failures[source_key]),
                }
            )
        else:
            reasons = failures[source_key] or ["missing_all_protocol_rows"]
            aggregated.append(
                {
                    "source_key": source_key,
                    "score": None,
                    "status": "failed",
                    "selected_state_index": None,
                    "selected_search_index": None,
                    "finite_constituent_count": 0,
                    "failure_reasons": sorted(reasons),
                }
            )
    return aggregated


def assemble_constituent_rows(
    prepared_rows: Iterable[Mapping[str, Any]],
    score_rows: Iterable[Mapping[str, Any]],
    *,
    search_index: int,
) -> list[dict[str, Any]]:
    """Join one frozen preparation manifest to one Uni-Dock/Vina search output."""
    if search_index < 0:
        raise ProtocolContractError("search index must be non-negative")

    scores_by_ligand: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(score_rows):
        ligand = str(row.get("ligand", "")).strip()
        if not ligand:
            raise ProtocolContractError(f"score row {row_number} lacks ligand")
        if ligand in scores_by_ligand:
            raise ProtocolContractError(f"duplicate score row for ligand: {ligand}")
        scores_by_ligand[ligand] = row

    assembled: list[dict[str, Any]] = []
    seen_states: set[tuple[str, int]] = set()
    consumed_ligands: set[str] = set()
    for row_number, prepared in enumerate(prepared_rows):
        source_key = str(prepared.get("source_key", "")).strip()
        if not _SOURCE_KEY_RE.match(source_key):
            raise ProtocolContractError(
                f"prepared row {row_number} lacks source_id@sample_index"
            )
        try:
            state_index = int(prepared.get("state_index", 0))
        except (TypeError, ValueError) as exc:
            raise ProtocolContractError(
                f"prepared row {row_number} has invalid state index"
            ) from exc
        if state_index < 0:
            raise ProtocolContractError("state index must be non-negative")
        state_key = (source_key, state_index)
        if state_key in seen_states:
            raise ProtocolContractError(f"duplicate prepared state: {state_key!r}")
        seen_states.add(state_key)

        status = str(prepared.get("preparation_status", "prepared")).lower()
        ligand = str(prepared.get("ligand", "")).strip()
        output = {
            "source_key": source_key,
            "state_index": state_index,
            "search_index": search_index,
            "engine": "unidock",
            "scoring": "vina",
            "ligand": ligand,
            "prepared_sha256": str(prepared.get("prepared_sha256", "")),
        }
        if status != "prepared":
            output.update(
                {
                    "score": None,
                    "reason": str(
                        prepared.get("failure_reason") or "preparation_failed"
                    ),
                }
            )
            assembled.append(output)
            continue
        if not ligand:
            raise ProtocolContractError(
                f"prepared row {row_number} is prepared but lacks ligand"
            )
        scored = scores_by_ligand.get(ligand)
        if scored is None:
            output.update({"score": None, "reason": "docking_missing_score"})
        else:
            consumed_ligands.add(ligand)
            output.update(
                {
                    "score": scored.get("score"),
                    "reason": str(scored.get("reason", "")),
                }
            )
        assembled.append(output)

    unknown = sorted(set(scores_by_ligand) - consumed_ligands)
    if unknown:
        raise ProtocolContractError(
            f"score output contains ligands absent from preparation manifest: {unknown}"
        )
    return assembled


def _pr_metrics(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    positives = sum(labels)
    if not labels or not positives:
        return {"average_precision": 0.0, "pr_auc": 0.0}
    groups: list[tuple[float, list[int]]] = []
    for score, label in sorted(zip(scores, labels), key=lambda item: item[0]):
        if groups and score == groups[-1][0]:
            groups[-1][1].append(label)
        else:
            groups.append((score, [label]))
    true_positives = false_positives = 0
    previous_recall, previous_precision = 0.0, 1.0
    average_precision = area = 0.0
    for _score, group_labels in groups:
        true_positives += sum(group_labels)
        false_positives += len(group_labels) - sum(group_labels)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        area += (recall - previous_recall) * (
            precision + previous_precision
        ) / 2
        previous_recall, previous_precision = recall, precision
    return {"average_precision": average_precision, "pr_auc": area}


def _tie_aware_enrichment(
    scores: Sequence[float], labels: Sequence[int], fraction: float
) -> float:
    if not labels or not sum(labels):
        return 0.0
    cutoff_count = max(1, math.ceil(len(labels) * fraction))
    remaining = cutoff_count
    expected_actives = 0.0
    groups: list[tuple[float, list[int]]] = []
    for score, label in sorted(zip(scores, labels), key=lambda item: item[0]):
        if groups and score == groups[-1][0]:
            groups[-1][1].append(label)
        else:
            groups.append((score, [label]))
    for _score, group_labels in groups:
        if remaining <= 0:
            break
        take = min(remaining, len(group_labels))
        expected_actives += take * (sum(group_labels) / len(group_labels))
        remaining -= take
    return (expected_actives / cutoff_count) / (sum(labels) / len(labels))


def _metric_values(
    scores: list[float], labels: list[int], scored: list[bool]
) -> dict[str, Any]:
    result = compute_metrics(scores, labels, scored=scored)
    pr = _pr_metrics(scores, labels)
    return {
        "ef1": result.ef1,
        "ef5": result.ef5,
        "ef10": round(_tie_aware_enrichment(scores, labels, 0.10), 6),
        "ef20": result.ef20,
        "roc_auc": result.auc_roc,
        "pr_auc": round(pr["pr_auc"], 6),
        "average_precision": round(pr["average_precision"], 6),
        "bedroc20": result.bedroc,
        "coverage": result.coverage,
        "first_active_rank": result.first_active_rank,
        "n_actives": result.n_actives,
        "n_decoys": result.n_decoys,
        "total_intended": result.total_screened,
        "failure_count": len(scored) - sum(scored),
    }


def metric_input_arrays(
    aggregated_rows: Sequence[Mapping[str, Any]],
    labels_by_source_key: Mapping[str, int],
) -> tuple[list[float], list[int], list[bool]]:
    """Create full-dataset arrays, placing tied failures strictly after scores."""
    keys = [str(row.get("source_key", "")) for row in aggregated_rows]
    if len(set(keys)) != len(keys):
        raise ProtocolContractError("aggregated source keys must be unique")
    if set(keys) != set(labels_by_source_key):
        raise ProtocolContractError("aggregated rows and labels must have identical keys")
    labels = [int(labels_by_source_key[key]) for key in keys]
    if any(label not in (0, 1) for label in labels):
        raise ProtocolContractError("labels must contain only 0 or 1")
    scored = [
        row.get("status") == "scored" and _finite_score(row.get("score")) is not None
        for row in aggregated_rows
    ]
    finite_scores = [
        float(row["score"]) for row, ok in zip(aggregated_rows, scored) if ok
    ]
    if finite_scores:
        failure_score = math.nextafter(max(finite_scores), math.inf)
        if not math.isfinite(failure_score):
            raise ProtocolContractError("cannot represent a finite worst-rank score")
    else:
        failure_score = 0.0
    scores = [
        float(row["score"]) if ok else failure_score
        for row, ok in zip(aggregated_rows, scored)
    ]
    return scores, labels, scored


def compute_early_enrichment_metrics(
    aggregated_rows: Sequence[Mapping[str, Any]],
    labels_by_source_key: Mapping[str, int],
) -> dict[str, Any]:
    scores, labels, scored = metric_input_arrays(
        aggregated_rows, labels_by_source_key
    )
    return _metric_values(scores, labels, scored)


def stratified_bootstrap_intervals(
    aggregated_rows: Sequence[Mapping[str, Any]],
    labels_by_source_key: Mapping[str, int],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_801,
) -> dict[str, Any]:
    """Return frozen stratified percentile intervals for target-level metrics."""
    if replicates <= 0:
        raise ProtocolContractError("bootstrap replicates must be positive")
    import numpy as np

    scores, labels, scored = metric_input_arrays(
        aggregated_rows, labels_by_source_key
    )
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    scored_array = np.asarray(scored, dtype=bool)
    active = np.flatnonzero(label_array == 1)
    decoy = np.flatnonzero(label_array == 0)
    if not len(active) or not len(decoy):
        raise ProtocolContractError("bootstrap requires active and decoy records")
    names = (
        "roc_auc",
        "bedroc20",
        "pr_auc",
        "average_precision",
        "ef1",
        "ef5",
        "ef10",
        "ef20",
        "coverage",
    )
    samples = {name: np.empty(replicates) for name in names}
    generator = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = np.concatenate(
            (
                generator.choice(active, len(active), replace=True),
                generator.choice(decoy, len(decoy), replace=True),
            )
        )
        values = _metric_values(
            score_array[indices].tolist(),
            label_array[indices].tolist(),
            scored_array[indices].tolist(),
        )
        for name in names:
            samples[name][replicate] = float(values[name])
    return {
        "method": "stratified nonparametric percentile bootstrap",
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "intervals": {
            name: {
                "lower_95": round(float(np.quantile(values, 0.025)), 6),
                "median": round(float(np.median(values)), 6),
                "upper_95": round(float(np.quantile(values, 0.975)), 6),
            }
            for name, values in samples.items()
        },
    }
