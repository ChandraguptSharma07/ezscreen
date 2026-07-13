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
SCHEMA_VERSION = 3

_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'running',
    config_json         TEXT,
    total_compounds     INTEGER DEFAULT 0,
    completed_compounds INTEGER DEFAULT 0
)
"""

_SHARDS_TABLE = """
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
)
"""

_ANNOTATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS annotations (
    run_id      TEXT NOT NULL,
    compound_id TEXT NOT NULL,
    flag        TEXT DEFAULT '',
    note        TEXT DEFAULT '',
    updated_at  TEXT,
    PRIMARY KEY (run_id, compound_id)
)
"""

_MD_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS md_runs (
    md_run_id      TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(run_id),
    ligand_id      TEXT NOT NULL,
    preset         TEXT NOT NULL,
    pose_source    TEXT NOT NULL,
    accelerator    TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',
    created_at     TEXT NOT NULL,
    completed_at   TEXT,
    output_dir     TEXT,
    summary_json   TEXT,
    error_message  TEXT
)
"""

_SCHEMA_COLUMNS: dict[str, dict[str, str]] = {
    "runs": {
        "status": "TEXT NOT NULL DEFAULT 'running'",
        "config_json": "TEXT",
        "total_compounds": "INTEGER DEFAULT 0",
        "completed_compounds": "INTEGER DEFAULT 0",
    },
    "shards": {
        "status": "TEXT NOT NULL DEFAULT 'pending'",
        "retry_count": "INTEGER DEFAULT 0",
        "compounds": "INTEGER DEFAULT 0",
        "error_message": "TEXT",
        "updated_at": "TEXT",
    },
    "annotations": {
        "flag": "TEXT DEFAULT ''",
        "note": "TEXT DEFAULT ''",
        "updated_at": "TEXT",
    },
    "md_runs": {
        "completed_at": "TEXT",
        "output_dir": "TEXT",
        "summary_json": "TEXT",
        "error_message": "TEXT",
    },
}


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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = _table_columns(conn, table)
    for name, ddl in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _migration_1(conn: sqlite3.Connection) -> None:
    conn.execute(_RUNS_TABLE)
    conn.execute(_SHARDS_TABLE)
    _ensure_columns(conn, "runs", _SCHEMA_COLUMNS["runs"])
    _ensure_columns(conn, "shards", _SCHEMA_COLUMNS["shards"])


def _migration_2(conn: sqlite3.Connection) -> None:
    conn.execute(_ANNOTATIONS_TABLE)
    _ensure_columns(conn, "annotations", _SCHEMA_COLUMNS["annotations"])


def _migration_3(conn: sqlite3.Connection) -> None:
    conn.execute(_MD_RUNS_TABLE)
    _ensure_columns(conn, "md_runs", _SCHEMA_COLUMNS["md_runs"])


_MIGRATIONS = {
    1: _migration_1,
    2: _migration_2,
    3: _migration_3,
}


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    with _connection() as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        for next_version in range(version + 1, SCHEMA_VERSION + 1):
            _MIGRATIONS[next_version](conn)
            conn.execute(f"PRAGMA user_version = {next_version}")


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
            """
            UPDATE runs
            SET status = 'complete',
                completed_compounds = CASE
                    WHEN total_compounds > completed_compounds THEN total_compounds
                    ELSE completed_compounds
                END
            WHERE run_id = ?
            """,
            (run_id,),
        )
        conn.execute(
            """
            UPDATE shards
            SET status = 'done', error_message = NULL, updated_at = ?
            WHERE run_id = ? AND status != 'failed'
            """,
            (_now(), run_id),
        )


def mark_run_failed(run_id: str) -> None:
    with _connection() as conn:
        conn.execute(
            "UPDATE runs SET status = 'failed' WHERE run_id = ?", (run_id,)
        )


def mark_run_partial(run_id: str) -> None:
    with _connection() as conn:
        conn.execute(
            "UPDATE runs SET status = 'partial' WHERE run_id = ?", (run_id,)
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


# ---------------------------------------------------------------------------
# MD runs
# ---------------------------------------------------------------------------

def create_md_run(
    md_run_id: str,
    run_id: str,
    ligand_id: str,
    preset: str,
    pose_source: str,
    accelerator: str,
    output_dir: Path | str | None = None,
) -> None:
    with _connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO md_runs (
                md_run_id, run_id, ligand_id, preset, pose_source, accelerator,
                status, created_at, completed_at, output_dir, summary_json, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
            """,
            (
                md_run_id,
                run_id,
                ligand_id,
                preset,
                pose_source,
                accelerator,
                "running",
                _now(),
                str(output_dir) if output_dir else None,
            ),
        )


def get_md_run(md_run_id: str) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM md_runs WHERE md_run_id = ?",
            (md_run_id,),
        ).fetchone()
    return dict(row) if row else None


def list_md_runs(run_id: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM md_runs"
    params: tuple[Any, ...] = ()
    if run_id:
        query += " WHERE run_id = ?"
        params = (run_id,)
    query += " ORDER BY created_at DESC"
    with _connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def mark_md_run_complete(
    md_run_id: str,
    output_dir: Path | str,
    summary: dict[str, Any] | None = None,
) -> None:
    with _connection() as conn:
        conn.execute(
            """
            UPDATE md_runs
            SET status = 'complete',
                completed_at = ?,
                output_dir = ?,
                summary_json = ?,
                error_message = NULL
            WHERE md_run_id = ?
            """,
            (
                _now(),
                str(output_dir),
                json.dumps(summary, cls=_PathEncoder) if summary is not None else None,
                md_run_id,
            ),
        )


def mark_md_run_failed(
    md_run_id: str,
    error_message: str,
    output_dir: Path | str | None = None,
) -> None:
    with _connection() as conn:
        conn.execute(
            """
            UPDATE md_runs
            SET status = 'failed',
                completed_at = ?,
                output_dir = COALESCE(?, output_dir),
                error_message = ?
            WHERE md_run_id = ?
            """,
            (_now(), str(output_dir) if output_dir else None, error_message, md_run_id),
        )


# ---------------------------------------------------------------------------
# Hit annotations - flags + notes
# ---------------------------------------------------------------------------

def set_annotation(run_id: str, compound_id: str, flag: str = "", note: str = "") -> None:
    """Upsert a flag (one of '', 'green', 'yellow', 'red') and note for a compound."""
    with _connection() as conn:
        conn.execute(
            "INSERT INTO annotations (run_id, compound_id, flag, note, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id, compound_id) DO UPDATE SET "
            "flag = excluded.flag, note = excluded.note, updated_at = excluded.updated_at",
            (run_id, compound_id, flag, note, _now()),
        )


def get_annotations(run_id: str) -> dict[str, dict[str, str]]:
    """Return {compound_id: {'flag': ..., 'note': ...}} for a run; {} when none."""
    with _connection() as conn:
        rows = conn.execute(
            "SELECT compound_id, flag, note FROM annotations WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    return {r["compound_id"]: {"flag": r["flag"] or "", "note": r["note"] or ""} for r in rows}
