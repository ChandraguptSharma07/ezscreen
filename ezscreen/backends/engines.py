from __future__ import annotations

from dataclasses import dataclass

# Docking-engine profile registry.
#
# Two dimensions the wizard exposes: the *engine* (what searches for poses) and
# the *scoring function* (the energy model). A scoring function is not owned by a
# single engine — the Vina family (UniDock, GNINA, smina, Vina) shares vina /
# vinardo / ad4 — so each engine declares the subset it supports and the wizard
# shows only those. CNN is the special case: GNINA produces it natively during
# docking AND it can rescore any engine's poses afterwards (see gnina_runner).
#
# Engines marked implemented=False are declared placeholders so smina / DiffDock /
# AutoDock-GPU slot in later (Phase 33) as one new entry + their runner, without
# rewiring the wizard.

# Scoring functions shared across the Vina-family engines.
_VINA_FAMILY = ("vina", "vinardo", "ad4")

DEFAULT_ENGINE = "unidock"


@dataclass(frozen=True)
class EngineProfile:
    key: str
    label: str
    requires_gpu: bool
    requires_box: bool
    scoring_functions: tuple[str, ...]
    default_scoring: str
    supports_cnn: bool
    native_score_type: str  # key into results.score_types
    implemented: bool
    note: str = ""
    # Kaggle notebook template that drives this engine's docking, or None when no
    # docking path exists yet. The wizard only offers engines that have one.
    kaggle_template: str | None = None


_ENGINES: dict[str, EngineProfile] = {
    "unidock": EngineProfile(
        key="unidock",
        label="UniDock (GPU)",
        requires_gpu=True,
        requires_box=True,
        scoring_functions=_VINA_FAMILY,
        default_scoring="vina",
        supports_cnn=False,
        native_score_type="vina_kcal_mol",
        implemented=True,
        note="Fast GPU Vina-family docking; the default.",
        kaggle_template="vina_shard.ipynb.j2",
    ),
    "unidock-pro": EngineProfile(
        key="unidock-pro",
        label="UniDock-Pro (GPU)",
        requires_gpu=True,
        requires_box=True,
        scoring_functions=_VINA_FAMILY,
        default_scoring="vina",
        supports_cnn=False,
        native_score_type="vina_kcal_mol",
        implemented=False,
        note="UniDock-Pro's real value is ligand-based + hybrid screening (a separate "
             "fork). Its structure-based docking is the same engine as UniDock, so it is "
             "not offered as a docking engine; the LBVS/hybrid modes are a future feature.",
    ),
    "gnina": EngineProfile(
        key="gnina",
        label="GNINA (GPU, CNN)",
        requires_gpu=True,
        requires_box=True,
        scoring_functions=_VINA_FAMILY,
        default_scoring="vina",
        supports_cnn=True,
        # Ranks by the Vina-style affinity (kcal/mol) so the merger's floor/ceiling/
        # sort logic is unchanged; CNNscore/CNNaffinity ride along as extra columns.
        native_score_type="vina_kcal_mol",
        implemented=True,
        note="Vina-family docking with CNN scores added per pose; slower per ligand than UniDock.",
        kaggle_template="vina_shard.ipynb.j2",
    ),
    # ---- declared, not yet implemented (Phase 33 slots) ----
    "vina-local": EngineProfile(
        key="vina-local",
        label="AutoDock Vina (local CPU)",
        requires_gpu=False,
        requires_box=True,
        scoring_functions=("vina", "vinardo"),
        default_scoring="vina",
        supports_cnn=False,
        native_score_type="vina_kcal_mol",
        implemented=False,
        note="Local CPU baseline; no Kaggle account needed.",
    ),
    "smina": EngineProfile(
        key="smina",
        label="smina (local CPU)",
        requires_gpu=False,
        requires_box=True,
        scoring_functions=("vina", "vinardo", "ad4", "dkoes_scoring"),
        default_scoring="vinardo",
        supports_cnn=False,
        native_score_type="vina_kcal_mol",
        implemented=False,
        note="Vina fork with custom atom-typed scoring; only worth adding for its custom functions, since vinardo/ad4 already ship via UniDock & GNINA.",
    ),
    "diffdock": EngineProfile(
        key="diffdock",
        label="DiffDock-L (GPU, box-free)",
        requires_gpu=True,
        requires_box=False,
        scoring_functions=(),
        default_scoring="",
        supports_cnn=False,
        native_score_type="diffdock_confidence",
        implemented=False,
        note="Diffusion model; blind docking, no box required.",
    ),
    "autodock-gpu": EngineProfile(
        key="autodock-gpu",
        label="AutoDock-GPU",
        requires_gpu=True,
        requires_box=True,
        scoring_functions=("ad4",),
        default_scoring="ad4",
        supports_cnn=False,
        native_score_type="vina_kcal_mol",
        implemented=False,
        note="Lamarckian GA; Kaggle template to be added.",
    ),
}


def get(key: str) -> EngineProfile:
    """Return the profile for an engine, falling back to the default engine."""
    return _ENGINES.get(key, _ENGINES[DEFAULT_ENGINE])


def all_engines() -> list[EngineProfile]:
    return list(_ENGINES.values())


def implemented_engines() -> list[EngineProfile]:
    """Engines that are implemented in some form (docking and/or rescore)."""
    return [e for e in _ENGINES.values() if e.implemented]


def dockable_engines() -> list[EngineProfile]:
    """Engines the run wizard offers — those with a working docking template."""
    return [e for e in _ENGINES.values() if e.implemented and e.kaggle_template]


def scoring_functions(key: str) -> tuple[str, ...]:
    return get(key).scoring_functions


def default_scoring(key: str) -> str:
    return get(key).default_scoring


def native_score_type(key: str) -> str:
    return get(key).native_score_type


def supports_scoring(key: str, scoring: str) -> bool:
    return scoring in get(key).scoring_functions
