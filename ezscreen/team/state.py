from __future__ import annotations

import calendar
import hashlib
import json
import re
import secrets
import sqlite3
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

KAGGLE_SECRET_FIELD_RE = re.compile(
    r"(^|_)(kaggle_key|kaggle_json|kaggle_credentials|kaggle_secret|key)$|^KAGGLE_KEY$",
    re.IGNORECASE,
)
PRE_RUN_STALE_STATUSES = {"needs_upload_tickets", "uploads_ready", "queued_run"}
PRE_RUN_STALE_TIMEOUT_S = 30 * 60


class TeamStateError(ValueError):
    """Raised when a team state transition is invalid."""


def default_db_path() -> Path:
    return Path.home() / ".ezscreen" / "team" / "team.db"


def assert_no_kaggle_secrets(payload: Any) -> None:
    """Reject raw Kaggle credentials before they can enter server state."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if KAGGLE_SECRET_FIELD_RE.search(str(key)):
                raise TeamStateError(f"Kaggle secret field is not accepted: {key}")
            assert_no_kaggle_secrets(value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_kaggle_secrets(item)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _parse_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class TeamStore:
    """SQLite-backed v1 team runner state.

    The store enforces one active team per user, enabled/paused workers, one active
    job per worker, upload-ticket ownership, and server-side Kaggle secret rejection.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    active_team_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    invite_code TEXT NOT NULL UNIQUE,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS members (
                    member_id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    github_repo TEXT,
                    github_installation_id INTEGER,
                    kaggle_username TEXT,
                    workflow_file TEXT NOT NULL DEFAULT 'team-worker.yml',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    paused INTEGER NOT NULL DEFAULT 0,
                    token_hash TEXT NOT NULL,
                    active_worker_job_id TEXT,
                    joined_at TEXT NOT NULL,
                    left_at TEXT,
                    FOREIGN KEY(team_id) REFERENCES teams(team_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
                CREATE TABLE IF NOT EXISTS invites (
                    invite_code TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS team_jobs (
                    team_job_id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS worker_jobs (
                    worker_job_id TEXT PRIMARY KEY,
                    team_job_id TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_payload_json TEXT NOT NULL DEFAULT '{}',
                    required_files_json TEXT NOT NULL,
                    result_zip_path TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(team_job_id) REFERENCES team_jobs(team_job_id),
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
                CREATE TABLE IF NOT EXISTS upload_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    team_job_id TEXT NOT NULL,
                    worker_job_id TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    upload_url TEXT NOT NULL,
                    blob_token TEXT NOT NULL,
                    expires_at TEXT,
                    uploaded_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(worker_job_id) REFERENCES worker_jobs(worker_job_id)
                );
                CREATE TABLE IF NOT EXISTS callbacks (
                    callback_id TEXT PRIMARY KEY,
                    team_job_id TEXT,
                    worker_job_id TEXT,
                    member_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    provider_login TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
                """
            )
            conn.execute(
                """
                UPDATE members
                SET active_worker_job_id = NULL
                WHERE active_worker_job_id IS NOT NULL
                  AND (
                    left_at IS NOT NULL
                    OR enabled = 0
                    OR github_repo IS NULL
                    OR TRIM(github_repo) = ''
                  )
                """
            )
            self._ensure_worker_job_columns(conn)
            self._ensure_member_columns(conn)
            self._release_stale_worker_slots(conn)


    def _ensure_worker_job_columns(self, conn: sqlite3.Connection) -> None:
        columns = {row['name'] for row in conn.execute("PRAGMA table_info(worker_jobs)").fetchall()}
        if 'status_payload_json' not in columns:
            conn.execute("ALTER TABLE worker_jobs ADD COLUMN status_payload_json TEXT NOT NULL DEFAULT '{}'")

    def _ensure_member_columns(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(members)").fetchall()}
        if "github_installation_id" not in columns:
            conn.execute("ALTER TABLE members ADD COLUMN github_installation_id INTEGER")
        if "kaggle_username" not in columns:
            conn.execute("ALTER TABLE members ADD COLUMN kaggle_username TEXT")

    def _refresh_team_job_status(self, conn: sqlite3.Connection, team_job_id: str) -> None:
        rows = conn.execute(
            "SELECT status, result_zip_path, completed_at FROM worker_jobs WHERE team_job_id = ?",
            (team_job_id,),
        ).fetchall()
        if not rows:
            return
        statuses = {row['status'] for row in rows}
        active_statuses = {'needs_upload_tickets', 'uploads_ready', 'queued_run', 'running', 'collecting'}
        terminal_statuses = {'complete', 'failed', 'timeout'}
        completed_at = None
        if statuses <= {'complete'}:
            next_status = 'complete'
            completed_at = _now()
        elif statuses <= terminal_statuses:
            next_status = 'failed'
            completed_at = _now()
        elif statuses & {'running', 'collecting'}:
            next_status = 'running'
        elif statuses & active_statuses:
            next_status = 'preparing_uploads'
        else:
            next_status = 'preparing_uploads'
        conn.execute(
            "UPDATE team_jobs SET status = ?, completed_at = ? WHERE team_job_id = ?",
            (next_status, completed_at, team_job_id),
        )

    def _release_stale_worker_slots(self, conn: sqlite3.Connection, *, now_ts: int | None = None) -> int:
        now_ts = int(time.time()) if now_ts is None else int(now_ts)
        released = 0
        rows = conn.execute(
            """
            SELECT worker_job_id, member_id, team_job_id, status, created_at
            FROM worker_jobs
            WHERE completed_at IS NULL
              AND status IN ('needs_upload_tickets', 'uploads_ready', 'queued_run')
            """
        ).fetchall()
        for row in rows:
            created_ts = _parse_timestamp(row["created_at"])
            if created_ts is None or (now_ts - created_ts) < PRE_RUN_STALE_TIMEOUT_S:
                continue
            timeout_at = _now()
            conn.execute(
                "UPDATE worker_jobs SET status = 'timeout', status_payload_json = ?, completed_at = ? WHERE worker_job_id = ?",
                (_json({"reason": "stale_pre_run_timeout"}), timeout_at, row["worker_job_id"]),
            )
            conn.execute(
                "UPDATE members SET active_worker_job_id = NULL WHERE member_id = ? AND active_worker_job_id = ?",
                (row["member_id"], row["worker_job_id"]),
            )
            conn.execute(
                """
                INSERT INTO callbacks(callback_id, team_job_id, worker_job_id, member_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _id("cb"),
                    row["team_job_id"],
                    row["worker_job_id"],
                    row["member_id"],
                    'timeout',
                    _json({"reason": "stale_pre_run_timeout"}),
                    timeout_at,
                ),
            )
            self._refresh_team_job_status(conn, row["team_job_id"])
            released += 1
        return released

    def ensure_user(self, display_name: str, user_id: str | None = None) -> dict[str, Any]:
        display_name = display_name.strip()
        if not display_name:
            raise TeamStateError("display_name is required")
        user_id = user_id or _id("usr")
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                if row["display_name"] != display_name:
                    conn.execute("UPDATE users SET display_name = ? WHERE user_id = ?", (display_name, user_id))
                    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                return dict(row)
            conn.execute(
                "INSERT INTO users(user_id, display_name, created_at) VALUES (?, ?, ?)",
                (user_id, display_name, _now()),
            )
            return dict(conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())

    def ensure_github_user(self, *, github_id: str | int, login: str, name: str | None = None) -> dict[str, Any]:
        github_id = str(github_id).strip()
        login = login.strip()
        if not github_id or not login:
            raise TeamStateError("github identity is incomplete")
        display_name = (name or login).strip()
        return self.ensure_user(display_name=display_name, user_id=f"gh_{github_id}")

    def create_session(
        self,
        *,
        user_id: str,
        provider: str,
        provider_login: str,
        ttl_seconds: int = 60 * 60 * 24 * 30,
    ) -> dict[str, Any]:
        token = secrets.token_urlsafe(40)
        session_id = _id("sess")
        expires_at = int(time.time()) + ttl_seconds
        with self.connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if not user:
                raise TeamStateError("session user does not exist")
            conn.execute(
                """
                INSERT INTO auth_sessions(session_id, user_id, token_hash, provider, provider_login, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, _hash_token(token), provider, provider_login, _now(), expires_at),
            )
        return {
            "session_token": token,
            "session_id": session_id,
            "user_id": user_id,
            "provider": provider,
            "provider_login": provider_login,
            "expires_at": expires_at,
        }

    def session_from_token(self, token: str) -> dict[str, Any]:
        if not token:
            raise TeamStateError("session token is required")
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT s.*, u.display_name, u.active_team_id
                FROM auth_sessions s JOIN users u ON u.user_id = s.user_id
                WHERE s.token_hash = ? AND s.revoked_at IS NULL
                """,
                (_hash_token(token),),
            ).fetchone()
        if not row:
            raise TeamStateError("invalid session token")
        session = dict(row)
        if int(session["expires_at"]) < int(time.time()):
            raise TeamStateError("session token expired")
        return session

    def revoke_session(self, token: str) -> dict[str, Any]:
        session = self.session_from_token(token)
        revoked_at = _now()
        with self.connection() as conn:
            conn.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE session_id = ? AND revoked_at IS NULL",
                (revoked_at, session["session_id"]),
            )
        session["revoked_at"] = revoked_at
        return {"ok": True, "session_id": session["session_id"], "user_id": session["user_id"], "revoked_at": revoked_at}

    def active_membership(self, user_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT m.*, t.name AS team_name
                FROM members m JOIN teams t ON t.team_id = m.team_id
                WHERE m.user_id = ? AND m.left_at IS NULL AND t.active = 1
                """,
                (user_id,),
            ).fetchone()
            return _row_to_dict(row)

    def create_team(
        self,
        *,
        owner_name: str,
        team_name: str,
        owner_user_id: str | None = None,
        github_repo: str | None = None,
        workflow_file: str = "team-worker.yml",
    ) -> dict[str, Any]:
        assert_no_kaggle_secrets(
            {
                "owner_name": owner_name,
                "team_name": team_name,
                "github_repo": github_repo,
                "workflow_file": workflow_file,
            }
        )
        owner = self.ensure_user(owner_name, owner_user_id)
        if self.active_membership(owner["user_id"]):
            raise TeamStateError("user already belongs to an active team")
        team_id = _id("team")
        member_id = _id("mem")
        invite_code = secrets.token_urlsafe(12)
        team_token = secrets.token_urlsafe(32)
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO teams(team_id, name, owner_user_id, invite_code, created_at) VALUES (?, ?, ?, ?, ?)",
                (team_id, team_name.strip() or f"{owner_name}'s team", owner["user_id"], invite_code, _now()),
            )
            conn.execute(
                "INSERT INTO invites(invite_code, team_id, created_at) VALUES (?, ?, ?)",
                (invite_code, team_id, _now()),
            )
            conn.execute(
                """
                INSERT INTO members(member_id, team_id, user_id, role, github_repo, workflow_file, token_hash, joined_at)
                VALUES (?, ?, ?, 'owner', ?, ?, ?, ?)
                """,
                (member_id, team_id, owner["user_id"], github_repo, workflow_file, _hash_token(team_token), _now()),
            )
            conn.execute(
                "UPDATE users SET active_team_id = ? WHERE user_id = ?",
                (team_id, owner["user_id"]),
            )
        return {
            "team_id": team_id,
            "team_name": team_name,
            "owner_user_id": owner["user_id"],
            "member_id": member_id,
            "invite_code": invite_code,
            "team_token": team_token,
        }

    def join_team(
        self,
        *,
        invite_code: str,
        user_name: str,
        user_id: str | None = None,
        github_repo: str | None,
        workflow_file: str = "team-worker.yml",
    ) -> dict[str, Any]:
        assert_no_kaggle_secrets(
            {
                "invite_code": invite_code,
                "user_name": user_name,
                "github_repo": github_repo,
                "workflow_file": workflow_file,
            }
        )
        user = self.ensure_user(user_name, user_id)
        if self.active_membership(user["user_id"]):
            raise TeamStateError("user already belongs to an active team")
        with self.connection() as conn:
            invite = conn.execute(
                """
                SELECT i.*, t.name AS team_name
                FROM invites i JOIN teams t ON t.team_id = i.team_id
                WHERE i.invite_code = ? AND i.active = 1 AND t.active = 1
                """,
                (invite_code,),
            ).fetchone()
            if not invite:
                raise TeamStateError("invite not found or inactive")
            member_id = _id("mem")
            team_token = secrets.token_urlsafe(32)
            conn.execute(
                """
                INSERT INTO members(member_id, team_id, user_id, role, github_repo, workflow_file, token_hash, joined_at)
                VALUES (?, ?, ?, 'member', ?, ?, ?, ?)
                """,
                (member_id, invite["team_id"], user["user_id"], github_repo, workflow_file, _hash_token(team_token), _now()),
            )
            conn.execute(
                "UPDATE users SET active_team_id = ? WHERE user_id = ?",
                (invite["team_id"], user["user_id"]),
            )
        return {
            "team_id": invite["team_id"],
            "team_name": invite["team_name"],
            "user_id": user["user_id"],
            "member_id": member_id,
            "team_token": team_token,
        }

    def leave_team(self, user_id: str) -> dict[str, Any]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE members
                SET left_at = ?, enabled = 0, paused = 1
                WHERE member_id = ?
                """,
                (_now(), membership["member_id"]),
            )
            conn.execute("UPDATE users SET active_team_id = NULL WHERE user_id = ?", (user_id,))
        return {"left": True, "member_id": membership["member_id"]}

    def set_worker_paused(self, user_id: str, paused: bool) -> dict[str, Any]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            conn.execute(
                "UPDATE members SET paused = ? WHERE member_id = ?",
                (1 if paused else 0, membership["member_id"]),
            )
        return {"member_id": membership["member_id"], "paused": paused}

    def connect_worker(
        self,
        user_id: str,
        *,
        github_repo: str,
        github_installation_id: int | None = None,
        kaggle_username: str = "",
        workflow_file: str = "team-worker.yml",
    ) -> dict[str, Any]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        github_repo = (github_repo or "").strip()
        if not github_repo or "/" not in github_repo:
            raise TeamStateError("github_repo must look like owner/repo")
        team_token = secrets.token_urlsafe(32)
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE members
                SET github_repo = ?, github_installation_id = ?, kaggle_username = ?, workflow_file = ?, token_hash = ?, enabled = 1, paused = 0
                WHERE member_id = ?
                """,
                (github_repo, github_installation_id, kaggle_username.strip(), workflow_file, _hash_token(team_token), membership["member_id"]),
            )
        return {
            "member_id": membership["member_id"],
            "github_repo": github_repo,
            "github_installation_id": github_installation_id,
            "kaggle_username": kaggle_username.strip(),
            "workflow_file": workflow_file,
            "team_token": team_token,
        }

    def set_worker_installation(self, user_id: str, github_installation_id: int | None) -> dict[str, Any]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        return self.set_worker_installation_by_member_id(
            member_id=membership["member_id"],
            github_installation_id=github_installation_id,
        )

    def set_worker_installation_by_member_id(self, *, member_id: str, github_installation_id: int | None) -> dict[str, Any]:
        with self.connection() as conn:
            conn.execute(
                "UPDATE members SET github_installation_id = ? WHERE member_id = ?",
                (github_installation_id, member_id),
            )
        return {"member_id": member_id, "github_installation_id": github_installation_id}


    def disconnect_worker(self, user_id: str) -> dict[str, Any]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE members
                SET github_repo = '', github_installation_id = NULL, kaggle_username = '', paused = 1, active_worker_job_id = NULL
                WHERE member_id = ?
                """,
                (membership["member_id"],),
            )
        return {"member_id": membership["member_id"], "disconnected": True}

    def status(self, user_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if not user or not user["active_team_id"]:
                return {"user_id": user_id, "team": None, "members": [], "jobs": []}
            team = conn.execute("SELECT * FROM teams WHERE team_id = ?", (user["active_team_id"],)).fetchone()
            members = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT m.member_id, m.user_id, u.display_name, m.role, m.github_repo, m.github_installation_id, m.kaggle_username,
                           m.workflow_file, m.enabled, m.paused, m.active_worker_job_id, m.left_at
                    FROM members m JOIN users u ON u.user_id = m.user_id
                    WHERE m.team_id = ? AND m.left_at IS NULL
                    ORDER BY m.joined_at
                    """,
                    (user["active_team_id"],),
                )
            ]
            jobs = [
                dict(row)
                for row in conn.execute(
                    "SELECT team_job_id, run_id, status, created_at, completed_at FROM team_jobs WHERE team_id = ? ORDER BY created_at DESC LIMIT 25",
                    (user["active_team_id"],),
                )
            ]
        return {"user_id": user_id, "team": dict(team), "members": members, "jobs": jobs}

    def worker_from_token(self, token: str) -> dict[str, Any]:
        token_hash = _hash_token(token)
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM members WHERE token_hash = ? AND left_at IS NULL AND enabled = 1",
                (token_hash,),
            ).fetchone()
        if not row:
            raise TeamStateError("invalid worker token")
        if row["paused"]:
            raise TeamStateError("worker is paused")
        return dict(row)

    def create_team_job(self, *, user_id: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert_no_kaggle_secrets(payload)
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        files = list(payload.get("files") or [])
        if not files:
            raise TeamStateError("job files are required")
        common_files = [file for file in files if file.get("role") != "ligand_shard"]
        shard_files = [file for file in files if file.get("role") == "ligand_shard"]
        team_job_id = _id("tjob")
        with self.connection() as conn:
            self._release_stale_worker_slots(conn)
            members = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM members
                    WHERE team_id = ? AND left_at IS NULL AND enabled = 1 AND paused = 0
                      AND active_worker_job_id IS NULL AND github_repo IS NOT NULL AND github_repo != ''
                    ORDER BY joined_at
                    """,
                    (membership["team_id"],),
                )
            ]
            if not members:
                raise TeamStateError("no enabled team workers are available")
            assignments = self._assign_team_job_files(
                members=members,
                common_files=common_files,
                shard_files=shard_files,
                requested_assignments=list(payload.get("team_member_assignments") or []),
            )
            if not assignments:
                raise TeamStateError("no team workers were assigned job files")
            conn.execute(
                """
                INSERT INTO team_jobs(team_job_id, team_id, owner_user_id, run_id, status, payload_json, created_at)
                VALUES (?, ?, ?, ?, 'preparing_uploads', ?, ?)
                """,
                (team_job_id, membership["team_id"], user_id, run_id, _json(payload), _now()),
            )
            worker_jobs = []
            for member, assigned in assignments:
                worker_job_id = _id("wjob")
                worker_payload = {
                    "accelerator": str(payload.get("accelerator") or "nvidiaTeslaP100"),
                }
                conn.execute(
                    """
                    INSERT INTO worker_jobs(worker_job_id, team_job_id, team_id, member_id, status, status_payload_json, required_files_json, created_at)
                    VALUES (?, ?, ?, ?, 'needs_upload_tickets', ?, ?, ?)
                    """,
                    (worker_job_id, team_job_id, membership["team_id"], member["member_id"], _json(worker_payload), _json(assigned), _now()),
                )
                conn.execute(
                    "UPDATE members SET active_worker_job_id = ? WHERE member_id = ?",
                    (worker_job_id, member["member_id"]),
                )
                worker_jobs.append({"worker_job_id": worker_job_id, "member_id": member["member_id"], "files": assigned})
        return {"team_job_id": team_job_id, "worker_jobs": worker_jobs}

    def _assign_team_job_files(
        self,
        *,
        members: list[dict[str, Any]],
        common_files: list[dict[str, Any]],
        shard_files: list[dict[str, Any]],
        requested_assignments: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        if not shard_files:
            return [(members[0], common_files)] if common_files else []

        requested_by_member = {
            item.get("member_id"): item.get("shard_count") for item in requested_assignments if item.get("member_id")
        }
        if requested_by_member:
            selected_members = [member for member in members if member["member_id"] in requested_by_member]
            unknown = set(requested_by_member) - {member["member_id"] for member in members}
            if unknown:
                raise TeamStateError("team member assignment includes unavailable workers")
        else:
            selected_members = members
        if not selected_members:
            raise TeamStateError("team member assignment excludes all available workers")

        explicit_counts: dict[str, int] = {}
        auto_members: list[dict[str, Any]] = []
        for member in selected_members:
            raw_count = requested_by_member.get(member["member_id"])
            if raw_count is None:
                auto_members.append(member)
                continue
            count = int(raw_count)
            if count < 0:
                raise TeamStateError("team member shard counts must be zero or greater")
            if count == 0:
                continue
            explicit_counts[member["member_id"]] = count

        assigned_shards: dict[str, list[dict[str, Any]]] = {member["member_id"]: [] for member in selected_members}
        shard_index = 0
        for member in selected_members:
            for _ in range(explicit_counts.get(member["member_id"], 0)):
                if shard_index >= len(shard_files):
                    break
                assigned_shards[member["member_id"]].append(shard_files[shard_index])
                shard_index += 1

        remaining = shard_files[shard_index:]
        auto_targets = auto_members or [member for member in selected_members if member["member_id"] in explicit_counts]
        if remaining and not auto_targets:
            raise TeamStateError("team member shard counts do not cover all shards")
        for index, shard in enumerate(remaining):
            member = auto_targets[index % len(auto_targets)]
            assigned_shards[member["member_id"]].append(shard)

        assignments: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for member in selected_members:
            member_shards = assigned_shards.get(member["member_id"], [])
            if not member_shards:
                continue
            assignments.append((member, [*common_files, *member_shards]))
        return assignments

    def list_worker_jobs(self, token: str) -> list[dict[str, Any]]:
        worker = self.worker_from_token(token)
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM worker_jobs
                WHERE member_id = ? AND status IN ('needs_upload_tickets', 'uploads_ready', 'queued_run', 'running', 'collecting')
                ORDER BY created_at
                """,
                (worker["member_id"],),
            ).fetchall()
        jobs = []
        for row in rows:
            item = dict(row)
            item["required_files"] = json.loads(item.pop("required_files_json"))
            item["payload"] = json.loads(item.pop("status_payload_json") or '{}')
            jobs.append(item)
        return jobs

    def record_upload_tickets(self, *, token: str, worker_job_id: str, tickets: list[dict[str, Any]]) -> dict[str, Any]:
        worker = self.worker_from_token(token)
        assert_no_kaggle_secrets(tickets)
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM worker_jobs WHERE worker_job_id = ?", (worker_job_id,)).fetchone()
            if not job or job["member_id"] != worker["member_id"]:
                raise TeamStateError("worker job does not belong to this worker")
            for ticket in tickets:
                conn.execute(
                    """
                    INSERT INTO upload_tickets(ticket_id, team_job_id, worker_job_id, team_id, member_id, filename,
                                               size_bytes, upload_url, blob_token, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticket.get("ticket_id") or _id("upl"),
                        job["team_job_id"],
                        worker_job_id,
                        job["team_id"],
                        worker["member_id"],
                        ticket["filename"],
                        int(ticket.get("size_bytes") or ticket.get("size") or 0),
                        ticket["upload_url"],
                        ticket["blob_token"],
                        ticket.get("expires_at"),
                        _now(),
                    ),
                )
            conn.execute("UPDATE worker_jobs SET status = 'uploads_ready' WHERE worker_job_id = ?", (worker_job_id,))
            self._refresh_team_job_status(conn, job["team_job_id"])
        return {"ok": True, "tickets": len(tickets)}

    def list_upload_tickets(self, *, user_id: str, team_job_id: str) -> list[dict[str, Any]]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM team_jobs WHERE team_job_id = ?", (team_job_id,)).fetchone()
            if not job or job["team_id"] != membership["team_id"]:
                raise TeamStateError("team job does not belong to this team")
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM upload_tickets WHERE team_job_id = ? ORDER BY created_at",
                    (team_job_id,),
                )
            ]

    def list_worker_upload_tickets(self, *, token: str, worker_job_id: str) -> list[dict[str, Any]]:
        worker = self.worker_from_token(token)
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM worker_jobs WHERE worker_job_id = ?", (worker_job_id,)).fetchone()
            if not job or job["member_id"] != worker["member_id"]:
                raise TeamStateError("worker job does not belong to this worker")
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM upload_tickets WHERE worker_job_id = ? ORDER BY created_at",
                    (worker_job_id,),
                )
            ]
    def acknowledge_upload(self, *, user_id: str, ticket_id: str) -> dict[str, Any]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            ticket = conn.execute("SELECT * FROM upload_tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if not ticket or ticket["team_id"] != membership["team_id"]:
                raise TeamStateError("upload ticket does not belong to this team")
            conn.execute(
                "UPDATE upload_tickets SET uploaded_at = COALESCE(uploaded_at, ?) WHERE ticket_id = ?",
                (_now(), ticket_id),
            )
            job = conn.execute("SELECT * FROM worker_jobs WHERE worker_job_id = ?", (ticket["worker_job_id"],)).fetchone()
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN uploaded_at IS NOT NULL THEN 1 ELSE 0 END) AS uploaded_count
                FROM upload_tickets
                WHERE worker_job_id = ?
                """,
                (ticket["worker_job_id"],),
            ).fetchone()
            if (
                job
                and job["status"] == "uploads_ready"
                and int(counts["total_count"] or 0) > 0
                and int(counts["uploaded_count"] or 0) == int(counts["total_count"] or 0)
            ):
                conn.execute(
                    "UPDATE worker_jobs SET status = 'queued_run' WHERE worker_job_id = ?",
                    (ticket["worker_job_id"],),
                )
                self._refresh_team_job_status(conn, ticket["team_job_id"])
                return {"ok": True}
        return {"ok": True}

    def mark_worker_job(self, *, token: str, worker_job_id: str, status: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        worker = self.worker_from_token(token)
        assert_no_kaggle_secrets(payload or {})
        completed = status in {"complete", "failed", "timeout"}
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM worker_jobs WHERE worker_job_id = ?", (worker_job_id,)).fetchone()
            if not job or job["member_id"] != worker["member_id"]:
                raise TeamStateError("worker job does not belong to this worker")
            completed_at = _now() if completed else None
            conn.execute(
                "UPDATE worker_jobs SET status = ?, status_payload_json = ?, completed_at = COALESCE(?, completed_at) WHERE worker_job_id = ?",
                (status, _json(payload or {}), completed_at, worker_job_id),
            )
            if completed:
                conn.execute(
                    "UPDATE members SET active_worker_job_id = NULL WHERE member_id = ?",
                    (worker["member_id"],),
                )
            conn.execute(
                """
                INSERT INTO callbacks(callback_id, team_job_id, worker_job_id, member_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (_id("cb"), job["team_job_id"], worker_job_id, worker["member_id"], status, _json(payload or {}), _now()),
            )
            self._refresh_team_job_status(conn, job["team_job_id"])
        return {"ok": True}

    def save_result_zip(self, *, token: str, worker_job_id: str, data: bytes, storage_root: Path) -> dict[str, Any]:
        worker = self.worker_from_token(token)
        storage_root.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM worker_jobs WHERE worker_job_id = ?", (worker_job_id,)).fetchone()
            if not job or job["member_id"] != worker["member_id"]:
                raise TeamStateError("worker job does not belong to this worker")
            path = storage_root / job["team_job_id"] / f"{worker_job_id}.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            complete_at = _now()
            conn.execute(
                "UPDATE worker_jobs SET result_zip_path = ?, status = 'complete', status_payload_json = ?, completed_at = ? WHERE worker_job_id = ?",
                (str(path), _json({"result_zip_path": str(path)}), complete_at, worker_job_id),
            )
            conn.execute(
                "UPDATE members SET active_worker_job_id = NULL WHERE member_id = ?",
                (worker["member_id"],),
            )
            self._refresh_team_job_status(conn, job["team_job_id"])
        return {"ok": True, "path": str(path)}

    def completed_result_zips(self, *, user_id: str, team_job_id: str) -> list[Path]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT result_zip_path FROM worker_jobs
                WHERE team_job_id = ? AND team_id = ? AND status = 'complete' AND result_zip_path IS NOT NULL
                """,
                (team_job_id, membership["team_id"]),
            ).fetchall()
        return [Path(row["result_zip_path"]) for row in rows]

    def list_team_job_workers(self, *, user_id: str, team_job_id: str) -> list[dict[str, Any]]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM team_jobs WHERE team_job_id = ?", (team_job_id,)).fetchone()
            if not job or job["team_id"] != membership["team_id"]:
                raise TeamStateError("team job does not belong to this team")
            rows = conn.execute(
                """
                SELECT worker_job_id, member_id, status, result_zip_path, created_at, completed_at, status_payload_json
                FROM worker_jobs
                WHERE team_job_id = ?
                ORDER BY created_at, worker_job_id
                """,
                (team_job_id,),
            ).fetchall()
        workers = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("status_payload_json") or "{}")
            workers.append(item)
        return workers

    def team_job_dispatch_targets(self, *, user_id: str, team_job_id: str) -> list[dict[str, Any]]:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            job = conn.execute("SELECT * FROM team_jobs WHERE team_job_id = ?", (team_job_id,)).fetchone()
            if not job or job["team_id"] != membership["team_id"]:
                raise TeamStateError("team job does not belong to this team")
            rows = conn.execute(
                """
                SELECT w.worker_job_id, w.member_id, w.status, m.github_repo, m.github_installation_id, m.workflow_file
                FROM worker_jobs w
                JOIN members m ON m.member_id = w.member_id
                WHERE w.team_job_id = ?
                ORDER BY w.created_at, w.worker_job_id
                """,
                (team_job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_result_zip(self, *, user_id: str, team_job_id: str, worker_job_id: str) -> bytes:
        membership = self.active_membership(user_id)
        if not membership:
            raise TeamStateError("user is not in an active team")
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT result_zip_path FROM worker_jobs
                WHERE worker_job_id = ? AND team_job_id = ? AND team_id = ? AND status = 'complete'
                """,
                (worker_job_id, team_job_id, membership["team_id"]),
            ).fetchone()
        if not row or not row["result_zip_path"]:
            raise TeamStateError("completed result zip is not available for this worker job")
        path = Path(row["result_zip_path"])
        if not path.exists():
            raise TeamStateError("result zip file is missing on the team server")
        return path.read_bytes()


def extract_result_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract one worker result ZIP while blocking path traversal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (output_dir / info.filename).resolve()
            if root != target and root not in target.parents:
                raise TeamStateError(f"unsafe zip member path: {info.filename}")
        zf.extractall(output_dir)
    return output_dir



