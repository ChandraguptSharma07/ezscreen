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
}


def info(score_type: str | None) -> dict[str, str]:
    return _SCORE_TYPES.get(score_type or DEFAULT_SCORE_TYPE, _SCORE_TYPES[DEFAULT_SCORE_TYPE])


def label(score_type: str | None) -> str:
    return info(score_type)["label"]


def unit(score_type: str | None) -> str:
    return info(score_type)["unit"]


def describe(score_type: str | None) -> str:
    return info(score_type)["desc"]


def read_score_type(output_dir: Path) -> str:
    """Read score_type from a run's results_meta.json sidecar; default when absent."""
    meta = output_dir / "results_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("score_type", DEFAULT_SCORE_TYPE)
        except Exception:
            pass
    return DEFAULT_SCORE_TYPE
