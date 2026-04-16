from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import ezscreen.config as cfg

# ---------------------------------------------------------------------------
# Helpers — redirect CONFIG_PATH to tmp_path so tests never touch ~/.ezscreen
# ---------------------------------------------------------------------------

def _patch(monkeypatch, tmp_path: Path) -> Path:
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", config_file)
    return config_file


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------

def test_load_creates_file_when_missing(monkeypatch, tmp_path):
    config_file = _patch(monkeypatch, tmp_path)
    assert not config_file.exists()
    data = cfg.load()
    assert config_file.exists()
    assert data["run"]["default_ph"] == pytest.approx(7.4)


def test_save_and_reload(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    custom = {"run": {"default_ph": 6.5, "admet_pre_filter": False}}
    cfg.save(custom)
    loaded = cfg.load()
    assert loaded["run"]["default_ph"] == pytest.approx(6.5)
    assert loaded["run"]["admet_pre_filter"] is False


def test_round_trip_preserves_all_keys(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    original = cfg.load()
    cfg.save(original)
    reloaded = cfg.load()
    assert reloaded == original


def test_save_writes_valid_toml(monkeypatch, tmp_path):
    config_file = _patch(monkeypatch, tmp_path)
    cfg.save(cfg.DEFAULTS)
    with config_file.open("rb") as f:
        parsed = tomllib.load(f)
    assert "run" in parsed


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------

def test_deep_merge_fills_missing_defaults(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    # write a partial config — only override one key
    cfg.save({"run": {"default_ph": 5.0}})
    loaded = cfg.load()
    # missing keys should be filled from DEFAULTS
    assert loaded["run"]["shard_retry_limit"] == 3
    assert loaded["run"]["default_ph"] == pytest.approx(5.0)


def test_deep_merge_override_wins(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    cfg.save({"run": {"default_search_depth": "Exhaustive"}})
    loaded = cfg.load()
    assert loaded["run"]["default_search_depth"] == "Exhaustive"


def test_deep_merge_nested_dict_merged_not_replaced(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    cfg.save({"defaults": {"box_padding": 10.0}})
    loaded = cfg.load()
    assert loaded["defaults"]["box_padding"] == pytest.approx(10.0)
    assert loaded["defaults"]["enumerate_tautomers"] is False  # from DEFAULTS


# ---------------------------------------------------------------------------
# get / set_value
# ---------------------------------------------------------------------------

def test_get_returns_leaf_value(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    assert cfg.get("run.default_ph") == pytest.approx(7.4)


def test_get_raises_for_missing_key(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="config key not found"):
        cfg.get("run.nonexistent_key")


def test_set_value_persists(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    cfg.set_value("run.default_ph", 8.0)
    assert cfg.get("run.default_ph") == pytest.approx(8.0)


def test_set_value_creates_nested_key(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    cfg.set_value("brand_new.section.key", "hello")
    assert cfg.get("brand_new.section.key") == "hello"
