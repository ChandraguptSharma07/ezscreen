from __future__ import annotations

import sqlite3
from contextlib import closing

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


def test_mark_run_partial():
    cp.create_run("run-003p", {}, 50)
    cp.mark_run_partial("run-003p")
    assert cp.get_run("run-003p")["status"] == "partial"


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


# ---------------------------------------------------------------------------
# annotations — flags + notes
# ---------------------------------------------------------------------------

def test_get_annotations_empty_for_unannotated_run():
    cp.create_run("run-a", {}, 10)
    assert cp.get_annotations("run-a") == {}


def test_set_and_get_annotation_round_trip():
    cp.create_run("run-a2", {}, 10)
    cp.set_annotation("run-a2", "ligX", flag="green", note="promising")
    anns = cp.get_annotations("run-a2")
    assert anns == {"ligX": {"flag": "green", "note": "promising"}}


def test_set_annotation_upsert_overwrites():
    cp.create_run("run-a3", {}, 10)
    cp.set_annotation("run-a3", "ligY", flag="green", note="first")
    cp.set_annotation("run-a3", "ligY", flag="red", note="second")
    anns = cp.get_annotations("run-a3")
    assert anns["ligY"] == {"flag": "red", "note": "second"}
    assert len(anns) == 1


def test_annotations_isolated_per_run():
    cp.create_run("run-a4", {}, 10)
    cp.create_run("run-a5", {}, 10)
    cp.set_annotation("run-a4", "lig1", flag="yellow")
    assert cp.get_annotations("run-a5") == {}



def test_mark_run_complete_marks_non_failed_shards_done():
    cp.create_run("run-variants", {}, 100)
    cp.add_shard("run-variants", 0, 50)
    cp.add_shard("run-variants", 1, 50)
    cp.mark_run_complete("run-variants")

    assert cp.get_run("run-variants")["completed_compounds"] == 100
    assert cp.get_incomplete_shards("run-variants") == []

def test_create_and_list_md_runs():
    cp.create_run("run-md-parent", {}, 10)
    cp.create_md_run(
        md_run_id="run-md-parent-md-lig1",
        run_id="run-md-parent",
        ligand_id="lig1",
        preset="smoke",
        pose_source="unidock",
        accelerator="nvidiaTeslaT4",
    )

    rows = cp.list_md_runs("run-md-parent")
    assert len(rows) == 1
    assert rows[0]["md_run_id"] == "run-md-parent-md-lig1"
    assert rows[0]["status"] == "running"


def test_mark_md_run_complete_persists_summary_and_output(tmp_path):
    cp.create_run("run-md-parent2", {}, 10)
    out_dir = tmp_path / "md-out"
    summary = {"platform": "CUDA", "simulated_ps": 10.0}
    cp.create_md_run(
        md_run_id="run-md-parent2-md-lig2",
        run_id="run-md-parent2",
        ligand_id="lig2",
        preset="quick",
        pose_source="gnina",
        accelerator="nvidiaTeslaT4",
        output_dir=out_dir,
    )
    cp.mark_md_run_complete("run-md-parent2-md-lig2", out_dir, summary=summary)

    row = cp.get_md_run("run-md-parent2-md-lig2")
    assert row is not None
    assert row["status"] == "complete"
    assert row["output_dir"] == str(out_dir)
    import json
    assert json.loads(row["summary_json"])["platform"] == "CUDA"


def test_mark_md_run_failed_records_error(tmp_path):
    cp.create_run("run-md-parent3", {}, 10)
    out_dir = tmp_path / "md-failed"
    cp.create_md_run(
        md_run_id="run-md-parent3-md-lig3",
        run_id="run-md-parent3",
        ligand_id="lig3",
        preset="standard",
        pose_source="unidock",
        accelerator="nvidiaTeslaP100",
    )
    cp.mark_md_run_failed("run-md-parent3-md-lig3", "kernel failed", out_dir)

    row = cp.get_md_run("run-md-parent3-md-lig3")
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_message"] == "kernel failed"
    assert row["output_dir"] == str(out_dir)


def test_init_db_migrates_legacy_schema_and_preserves_rows():
    cp.DB_PATH.unlink()
    with closing(sqlite3.connect(cp.DB_PATH)) as conn:
        conn.executescript(
            """
            PRAGMA user_version = 0;
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE shards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                UNIQUE(run_id, shard_index)
            );
            INSERT INTO runs (run_id, created_at) VALUES ('legacy-run', '2026-01-01T00:00:00+00:00');
            INSERT INTO shards (run_id, shard_index) VALUES ('legacy-run', 0);
            """
        )

    cp.init_db()

    run = cp.get_run('legacy-run')
    assert run is not None
    assert run['status'] == 'running'
    assert run['total_compounds'] == 0
    assert run['completed_compounds'] == 0
    assert cp.get_incomplete_shards('legacy-run')[0]['status'] == 'pending'

    with closing(sqlite3.connect(cp.DB_PATH)) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert version == cp.SCHEMA_VERSION
    assert {'runs', 'shards', 'annotations', 'md_runs'} <= tables


def test_init_db_upgrades_from_annotations_only_schema():
    cp.DB_PATH.unlink()
    with closing(sqlite3.connect(cp.DB_PATH)) as conn:
        conn.executescript(
            """
            PRAGMA user_version = 2;
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                config_json TEXT,
                total_compounds INTEGER DEFAULT 0,
                completed_compounds INTEGER DEFAULT 0
            );
            CREATE TABLE shards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                compounds INTEGER DEFAULT 0,
                error_message TEXT,
                updated_at TEXT,
                UNIQUE(run_id, shard_index)
            );
            CREATE TABLE annotations (
                run_id TEXT NOT NULL,
                compound_id TEXT NOT NULL,
                flag TEXT DEFAULT '',
                note TEXT DEFAULT '',
                updated_at TEXT,
                PRIMARY KEY (run_id, compound_id)
            );
            """
        )

    cp.init_db()

    with closing(sqlite3.connect(cp.DB_PATH)) as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        md_runs_cols = {row[1] for row in conn.execute('PRAGMA table_info(md_runs)')}

    assert version == cp.SCHEMA_VERSION
    assert {'md_run_id', 'run_id', 'ligand_id', 'preset', 'pose_source', 'accelerator', 'status', 'created_at'} <= md_runs_cols
