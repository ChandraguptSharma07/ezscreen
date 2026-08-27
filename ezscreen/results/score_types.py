from __future__ import annotations

import json
from pathlib import Path

# Native score type per engine. The merger writes the active one into
# results_meta.json; the engine selector (Phase 32) sets it per backend.
DEFAULT_SCORE_TYPE = "vina_kcal_mol"

_SCORE_TYPES: dict[str, dict[str, str]] = {
    "vina_kcal_mol": {
        "label": "Vina score (kcal/mol)",
        "unit":  "kcal/mol",
        "desc":  "more negative = stronger predicted binding",
    },
    "cnn_affinity": {
        "label": "GNINA CNNaffinity (pK)",
        "unit":  "pK",
        "desc":  "higher = stronger predicted binding",
    },
    "diffdock_confidence": {
        "label": "DiffDock confidence",
        "unit":  "",
        "desc":  "higher = more confident pose",
    },
    "consensus_rank": {
        "label": "Consensus normalized rank",
        "unit":  "",
        "desc":  "lower = better mean rank across engines",
    },
}


def info(score_type: str | None) -> dict[str, str]:
    return _SCORE_TYPES.get(score_type or DEFAULT_SCORE_TYPE, _SCORE_TYPES[DEFAULT_SCORE_TYPE])


def label(score_type: str | None) -> str:
    return info(score_type)["label"]


def unit(score_type: str | None) -> str:
    return info(score_type)["unit"]


def describe(score_type: str | None) -> str:
    return info(score_type)["desc"]


def read_results_meta(output_dir: Path) -> dict:
    """Read a results metadata sidecar, returning an empty mapping if invalid."""
    meta = output_dir / "results_meta.json"
    if meta.exists():
        try:
            value = json.loads(meta.read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            pass
    return {}


def read_score_type(output_dir: Path) -> str:
    """Read score_type from a run's results_meta.json sidecar; default when absent."""
    return str(read_results_meta(output_dir).get("score_type") or DEFAULT_SCORE_TYPE)


def read_score_col(output_dir: Path, headers: list[str]) -> str:
    """Resolve the primary score column, preferring the merger's sidecar contract."""
    configured = str(read_results_meta(output_dir).get("score_col") or "")
    if configured in headers:
        return configured
    if "consensus_norm_rank" in headers:
        return "consensus_norm_rank"
    return next(
        (
            header
            for header in headers
            if any(token in header.lower() for token in ("score", "affinity", "confidence"))
        ),
        headers[-1] if headers else "",
    )


def higher_is_better(score_type: str | None) -> bool:
    return score_type in {"cnn_affinity", "diffdock_confidence"}


def ranking_integrity_warning(
    rows: list[dict], score_col: str, score_type: str | None
) -> str | None:
    """Warn when a legacy Vina-family ranking retains failure-like top scores."""
    if higher_is_better(score_type) or score_type == "consensus_rank":
        return None
    invalid = 0
    for row in rows:
        try:
            if float(row.get(score_col, "")) <= -20.0:
                invalid += 1
        except (TypeError, ValueError):
            continue
    if not invalid:
        return None
    noun = "row" if invalid == 1 else "rows"
    return (
        f"Ranking integrity warning: {invalid} {noun} contain Vina-family scores "
        "at or below -20 and may be legacy docking failures ranked as top hits. "
        "Treat this archived ranking as quarantined and re-merge from raw outputs."
    )
