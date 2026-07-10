from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

LOCAL_STATE_PATH = Path.home() / ".ezscreen" / "team" / "client.json"


def default_server_url() -> str:
    env_url = os.environ.get("EZSCREEN_TEAM_SERVER_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    try:
        from ezscreen import config

        configured = str(config.load().get("team", {}).get("server_url", "")).strip()
        if configured:
            return configured.rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8787"


class TeamClientError(RuntimeError):
    pass


def load_local_state() -> dict[str, Any]:
    if not LOCAL_STATE_PATH.exists():
        return {}
    return json.loads(LOCAL_STATE_PATH.read_text(encoding="utf-8"))


def save_local_state(state: dict[str, Any]) -> None:
    LOCAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def clear_local_session(*, preserve_keys: set[str] | None = None) -> dict[str, Any]:
    preserve_keys = preserve_keys or {"server_url", "user_name"}
    state = load_local_state()
    kept = {key: value for key, value in state.items() if key in preserve_keys}
    save_local_state(kept)
    return kept


class TeamClient:
    def __init__(self, server_url: str | None = None, token: str | None = None) -> None:
        self.server_url = (server_url or default_server_url()).rstrip("/")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        token: str | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        auth_token = token if token is not None else self.token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        req = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TeamClientError(f"HTTP {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else {}

    def create_team(
        self,
        *,
        owner_name: str,
        team_name: str,
        github_repo: str = "",
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {"owner_name": owner_name, "team_name": team_name, "github_repo": github_repo}
        if owner_user_id:
            payload["owner_user_id"] = owner_user_id
        return self.request(
            "POST",
            "/v1/teams",
            payload,
        )

    def join_team(
        self,
        *,
        invite_code: str,
        user_name: str,
        github_repo: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {"invite_code": invite_code, "user_name": user_name, "github_repo": github_repo}
        if user_id:
            payload["user_id"] = user_id
        return self.request(
            "POST",
            "/v1/teams/join",
            payload,
        )

    def status(self, user_id: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"user_id": user_id})
        return self.request("GET", f"/v1/team/status?{query}")

    def start_github_device_login(self) -> dict[str, Any]:
        return self.request("POST", "/v1/auth/github/device/start", {})

    def poll_github_device_login(self, device_code: str) -> dict[str, Any]:
        return self.request("POST", "/v1/auth/github/device/poll", {"device_code": device_code})

    def session(self) -> dict[str, Any]:
        return self.request("GET", "/v1/auth/session")

    def logout(self) -> dict[str, Any]:
        return self.request("POST", "/v1/auth/logout", {})

    def connect_worker(self, *, github_repo: str, workflow_file: str = "team-worker.yml") -> dict[str, Any]:
        return self.request(
            "POST",
            "/v1/team/worker/connect",
            {"github_repo": github_repo, "workflow_file": workflow_file},
        )

    def disconnect_worker(self, *, user_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if user_id:
            payload["user_id"] = user_id
        return self.request("POST", "/v1/team/worker/disconnect", payload)


    def upload_result_zip(self, *, worker_job_id: str, data: bytes) -> dict[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/zip"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            f"{self.server_url}/v1/worker/result/{worker_job_id}",
            data=data,
            method="PUT",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TeamClientError(f"HTTP {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else {}
