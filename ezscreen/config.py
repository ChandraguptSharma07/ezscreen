from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

CONFIG_DIR: Path = Path.home() / ".ezscreen"
CONFIG_PATH: Path = CONFIG_DIR / "config.toml"

DEFAULTS: dict[str, Any] = {
    "run": {
        "auto_resume_threshold": 10,
        "shard_retry_limit": 3,
        "default_search_depth": "Balanced",
        "default_ph": 7.4,
        "admet_pre_filter": True,
    },
    "kaggle": {
        "default_dataset": "",
    },
    "defaults": {
        "box_padding": 5.0,
        "enumerate_tautomers": False,
    },
    "local": {
        "enable_score_floor": True,
        "score_floor": -15.0,
        "score_ceiling": 0.0,
        "exhaustiveness": 4,
        "cpu_cores": 0,
    },
    "prep": {
        "enable_gpu_size_filter": True,
        "max_heavy_atoms": 70,
        "max_mw": 700.0,
        "max_rotatable_bonds": 20,
        "mmff_max_iters": 0,
    },
}


def load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save(DEFAULTS)
        return _deep_copy(DEFAULTS)
    with CONFIG_PATH.open("rb") as f:
        data = tomllib.load(f)
    return _deep_merge(DEFAULTS, data)


def save(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("wb") as f:
        tomli_w.dump(config, f)


def get(key_path: str, config: dict[str, Any] | None = None) -> Any:
    if config is None:
        config = load()
    node: Any = config
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config key not found: '{key_path}'")
        node = node[part]
    return node


def set_value(key_path: str, value: Any) -> None:
    config = load()
    parts = key_path.split(".")
    node: Any = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    save(config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_copy(d: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(d, {})
