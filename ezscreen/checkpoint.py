from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from ezscreen.errors import CheckpointError

DB_DIR: Path = Path.home() / ".ezscreen"
DB_PATH: Path = DB_DIR / "checkpoints.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'running',
    config_json         TEXT,
    total_compounds     INTEGER DEFAULT 0,
    completed_compounds INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shards (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL REFERENCES runs(run_id),
    shard_index   INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    retry_count   INTEGER DEFAULT 0,
    compounds     INTEGER DEFAULT 0,
    error_message TEXT,
    updated_at    TEXT,
    UNIQUE(run_id, shard_index)
);
"""


@contextmanager
def _connection() -> Generator[sqlite3.Connection, None, None]:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        raise CheckpointError(str(exc)) from exc
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _PathEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    with _connection() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Run operations
# ---------------------------------------------------------------------------

def create_run(run_id: str, config: dict[str, Any], total_compounds: int) -> None:
    with _connection() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, created_at, config_json, total_compounds) "
            "VALUES (?, ?, ?, ?)",
            (run_id, _now(), json.dumps(config, cls=_PathEncoder), total_compounds),
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_runs() -> list[dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def mark_run_complete(run_id: str) -> None:
    with _connection() as conn:
        conn.execute(
            "UPDATE runs SET status = 'complete' WHERE run_id = ?", (run_id,)
        )


def mark_run_failed(run_id: str) -> None:
    with _connection() as conn:
        conn.execute(
            "UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,)
        )


def increment_completed(run_id: str, count: int) -> None:
    with _connection() as conn:
        conn.execute(
            "UPDATE runs SET completed_compounds = completed_compounds + ? "
            "WHERE run_id = ?",
            (count, run_id),
        )


# ---------------------------------------------------------------------------
# Shard operations
# ---------------------------------------------------------------------------

def add_shard(run_id: str, shard_index: int, compounds: int) -> None:
    with _connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO shards (run_id, shard_index, compounds, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, shard_index, compounds, _now()),
        )


def update_shard(
    run_id: str,
    shard_index: int,
    status: str,
    error: str | None = None,
) -> None:
    with _connection() as conn:
        conn.execute(
            "UPDATE shards SET status = ?, error_message = ?, updated_at = ? "
            "WHERE run_id = ? AND shard_index = ?",
            (status, error, _now(), run_id, shard_index),
        )


def increment_shard_retry(run_id: str, shard_index: int) -> int:
    with _connection() as conn:
        conn.execute(
            "UPDATE shards SET retry_count = retry_count + 1, updated_at = ? "
            "WHERE run_id = ? AND shard_index = ?",
            (_now(), run_id, shard_index),
        )
        row = conn.execute(
            "SELECT retry_count FROM shards "
            "WHERE run_id = ? AND shard_index = ?",
            (run_id, shard_index),
        ).fetchone()
    return int(row["retry_count"]) if row else 0


def get_incomplete_shards(run_id: str) -> list[dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM shards "
            "WHERE run_id = ? AND status NOT IN ('done') "
            "ORDER BY shard_index",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_failed_shards(run_id: str) -> list[dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM shards WHERE run_id = ? AND status = 'failed' "
            "ORDER BY shard_index",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]
