from __future__ import annotations

import pytest

import ezscreen.checkpoint as cp

# ---------------------------------------------------------------------------
# Redirect DB to a fresh tmp_path SQLite file per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    db_file = tmp_path / "checkpoints.db"
    monkeypatch.setattr(cp, "DB_DIR", tmp_path)
    monkeypatch.setattr(cp, "DB_PATH", db_file)
    cp.init_db()
    yield


# ---------------------------------------------------------------------------
# create_run / get_run
# ---------------------------------------------------------------------------

def test_create_and_get_run():
    cp.create_run("run-001", {"receptor": "1hsg"}, total_compounds=500)
    run = cp.get_run("run-001")
    assert run is not None
    assert run["run_id"] == "run-001"
    assert run["status"] == "running"
    assert run["total_compounds"] == 500


def test_get_run_returns_none_for_unknown():
    assert cp.get_run("does-not-exist") is None


def test_create_run_stores_config_json():
    cp.create_run("run-002", {"depth": "balanced", "ph": 7.4}, total_compounds=10)
    run = cp.get_run("run-002")
    import json
    config = json.loads(run["config_json"])
    assert config["depth"] == "balanced"


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_list_runs_empty():
    assert cp.list_runs() == []


def test_list_runs_returns_all():
    cp.create_run("a", {}, 100)
    cp.create_run("b", {}, 200)
    runs = cp.list_runs()
    assert len(runs) == 2
    ids = {r["run_id"] for r in runs}
    assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# mark_complete / mark_failed
# ---------------------------------------------------------------------------

def test_mark_run_complete():
    cp.create_run("run-003", {}, 50)
    cp.mark_run_complete("run-003")
    assert cp.get_run("run-003")["status"] == "complete"


def test_mark_run_failed():
    cp.create_run("run-004", {}, 50)
    cp.mark_run_failed("run-004")
    assert cp.get_run("run-004")["status"] == "failed"


# ---------------------------------------------------------------------------
# shard CRUD
# ---------------------------------------------------------------------------

def test_add_and_get_shard():
    cp.create_run("run-s", {}, 100)
    cp.add_shard("run-s", shard_index=0, compounds=50)
    cp.add_shard("run-s", shard_index=1, compounds=50)
    pending = cp.get_incomplete_shards("run-s")
    assert len(pending) == 2
    assert pending[0]["shard_index"] == 0


def test_update_shard_to_done():
    cp.create_run("run-s2", {}, 100)
    cp.add_shard("run-s2", 0, 50)
    cp.update_shard("run-s2", 0, "done")
    incomplete = cp.get_incomplete_shards("run-s2")
    assert incomplete == []


def test_update_shard_to_failed_with_error():
    cp.create_run("run-s3", {}, 100)
    cp.add_shard("run-s3", 0, 50)
    cp.update_shard("run-s3", 0, "failed", error="OOM")
    failed = cp.get_failed_shards("run-s3")
    assert len(failed) == 1
    assert failed[0]["error_message"] == "OOM"


def test_add_shard_idempotent():
    cp.create_run("run-s4", {}, 100)
    cp.add_shard("run-s4", 0, 50)
    cp.add_shard("run-s4", 0, 50)  # INSERT OR IGNORE — should not raise
    assert len(cp.get_incomplete_shards("run-s4")) == 1


# ---------------------------------------------------------------------------
# increment_retry
# ---------------------------------------------------------------------------

def test_increment_shard_retry():
    cp.create_run("run-r", {}, 100)
    cp.add_shard("run-r", 0, 50)
    count1 = cp.increment_shard_retry("run-r", 0)
    count2 = cp.increment_shard_retry("run-r", 0)
    assert count1 == 1
    assert count2 == 2
