from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request
from urllib.parse import parse_qs, urlparse

from ezscreen.team.github import GitHubError, dispatch_worker_workflow_via_app, get_repo_installation
from ezscreen.team.state import TeamStateError, TeamStore

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def refresh_status_installations(
    *,
    status: dict[str, Any],
    app_config: dict[str, str] | None,
    resolve_installation_id,
    persist_installation_id,
) -> dict[str, Any]:
    if not app_config or not status.get("team"):
        return status
    for member in status.get("members", []):
        repo = str(member.get("github_repo") or "").strip()
        if not repo or member.get("github_installation_id"):
            continue
        try:
            installation_id = resolve_installation_id(repo)
        except Exception:
            continue
        if not installation_id:
            continue
        persist_installation_id(member_id=str(member["member_id"]), github_installation_id=int(installation_id))
        member["github_installation_id"] = int(installation_id)
    return status


class TeamHandler(BaseHTTPRequestHandler):
    store: TeamStore
    storage_root: Path

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, data: bytes, *, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _token(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        return ""

    def _session(self) -> dict[str, Any] | None:
        token = self._token()
        if not token:
            return None
        try:
            return self.store.session_from_token(token)
        except TeamStateError:
            return None

    def _user_id_from_request(
        self,
        payload: dict[str, Any] | None = None,
        query: dict[str, list[str]] | None = None,
    ) -> str:
        payload = payload or {}
        query = query or {}
        user_id = payload.get("user_id") or (query.get("user_id") or [""])[0]
        if user_id:
            return str(user_id)
        session = self._session()
        if not session:
            raise TeamStateError("user_id or signed-in session is required")
        return str(session["user_id"])

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, TeamStateError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _github_client_id(self) -> str:
        client_id = os.environ.get("EZSCREEN_GITHUB_CLIENT_ID", "").strip()
        if not client_id:
            raise TeamStateError("EZSCREEN_GITHUB_CLIENT_ID is not configured on the team server")
        return client_id

    def _public_server_url(self) -> str:
        public_url = os.environ.get("EZSCREEN_TEAM_PUBLIC_URL", "").strip()
        if public_url:
            return public_url.rstrip("/")
        return f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"

    def _github_app_config(self) -> dict[str, str] | None:
        app_id = os.environ.get("EZSCREEN_GITHUB_APP_ID", "").strip()
        private_key = os.environ.get("EZSCREEN_GITHUB_APP_PRIVATE_KEY", "")
        private_key_path = os.environ.get("EZSCREEN_GITHUB_APP_PRIVATE_KEY_PATH", "").strip()
        install_url = os.environ.get("EZSCREEN_GITHUB_APP_INSTALL_URL", "").strip()
        if private_key_path and not private_key:
            private_key = Path(private_key_path).read_text(encoding="utf-8")
        private_key = private_key.replace("\\n", "\n").strip()
        if not app_id or not private_key:
            return None
        return {"app_id": app_id, "private_key": private_key, "install_url": install_url}

    def _repo_installation_id(self, repo: str) -> int | None:
        config = self._github_app_config()
        if not config:
            return None
        installation = get_repo_installation(
            repo=repo,
            app_id=config["app_id"],
            private_key_pem=config["private_key"],
        )
        if not installation or not installation.get("id"):
            return None
        return int(installation["id"])

    def _dispatch_team_job(self, *, user_id: str, team_job_id: str, mode: str) -> dict[str, Any]:
        if mode not in {"prepare_upload", "run_kaggle", "collect"}:
            raise TeamStateError("unsupported dispatch mode")
        targets = self.store.team_job_dispatch_targets(user_id=user_id, team_job_id=team_job_id)
        config = self._github_app_config()
        if not config:
            return {
                "dispatched": [],
                "skipped": [
                    {"worker_job_id": item["worker_job_id"], "repo": item.get("github_repo") or "", "reason": "app_not_configured"}
                    for item in targets
                ],
                "failed": [],
                "mode": mode,
                "dispatch_backend": "github_app",
            }
        dispatched: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []
        for item in targets:
            repo = str(item.get("github_repo") or "").strip()
            if not repo:
                skipped.append({"worker_job_id": item["worker_job_id"], "reason": "repo_not_connected"})
                continue
            installation_id = item.get("github_installation_id")
            if not installation_id:
                try:
                    installation_id = self._repo_installation_id(repo)
                except GitHubError as exc:
                    failed.append({"worker_job_id": item["worker_job_id"], "repo": repo, "reason": str(exc)})
                    continue
                if installation_id:
                    self.store.set_worker_installation_by_member_id(
                        member_id=str(item["member_id"]),
                        github_installation_id=int(installation_id),
                    )
            if not installation_id:
                skipped.append({"worker_job_id": item["worker_job_id"], "repo": repo, "reason": "app_installation_missing"})
                continue
            try:
                dispatch_worker_workflow_via_app(
                    repo=repo,
                    app_id=config["app_id"],
                    private_key_pem=config["private_key"],
                    installation_id=int(installation_id),
                    mode=mode,
                    worker_job_id=str(item["worker_job_id"]),
                    workflow_file=str(item.get("workflow_file") or "team-worker.yml"),
                )
                dispatched.append(
                    {
                        "worker_job_id": str(item["worker_job_id"]),
                        "repo": repo,
                        "mode": mode,
                        "installation_id": str(installation_id),
                    }
                )
            except GitHubError as exc:
                failed.append({"worker_job_id": item["worker_job_id"], "repo": repo, "reason": str(exc)})
        return {
            "dispatched": dispatched,
            "skipped": skipped,
            "failed": failed,
            "mode": mode,
            "dispatch_backend": "github_app",
        }

    def _github_form_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _github_user(self, access_token: str) -> dict[str, Any]:
        req = urllib.request.Request(
            GITHUB_USER_URL,
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "ezscreen-team-server",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _start_github_device_login(self) -> dict[str, Any]:
        return self._github_form_post(
            GITHUB_DEVICE_CODE_URL,
            {"client_id": self._github_client_id(), "scope": "read:user"},
        )

    def _poll_github_device_login(self, device_code: str) -> dict[str, Any]:
        if not device_code:
            raise TeamStateError("device_code is required")
        response = self._github_form_post(
            GITHUB_ACCESS_TOKEN_URL,
            {
                "client_id": self._github_client_id(),
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        if "error" in response:
            if response["error"] in {"authorization_pending", "slow_down"}:
                return {"status": "pending", "error": response["error"], "interval": response.get("interval")}
            raise TeamStateError(str(response.get("error_description") or response["error"]))
        access_token = response.get("access_token")
        if not access_token:
            raise TeamStateError("GitHub did not return an access token")
        github_user = self._github_user(access_token)
        user = self.store.ensure_github_user(
            github_id=github_user["id"],
            login=github_user["login"],
            name=github_user.get("name") or github_user["login"],
        )
        session = self.store.create_session(
            user_id=user["user_id"],
            provider="github",
            provider_login=github_user["login"],
        )
        return {
            "status": "complete",
            "session_token": session["session_token"],
            "user_id": user["user_id"],
            "user_name": user["display_name"],
            "github_login": github_user["login"],
            "expires_at": session["expires_at"],
        }

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            query = parse_qs(parsed.query)
            if parsed.path == "/health":
                return self._send_json(HTTPStatus.OK, {"ok": True})
            if parsed.path == "/v1/auth/session":
                session = self.store.session_from_token(self._token())
                return self._send_json(
                    HTTPStatus.OK,
                    {
                        "user_id": session["user_id"],
                        "user_name": session["display_name"],
                        "provider": session["provider"],
                        "provider_login": session["provider_login"],
                        "expires_at": session["expires_at"],
                    },
                )
            if parsed.path == "/v1/team/status":
                status = self.store.status(self._user_id_from_request(query=query))
                status = refresh_status_installations(
                    status=status,
                    app_config=self._github_app_config(),
                    resolve_installation_id=self._repo_installation_id,
                    persist_installation_id=self.store.set_worker_installation_by_member_id,
                )
                return self._send_json(HTTPStatus.OK, status)
            if parsed.path == "/v1/github/app":
                config = self._github_app_config()
                return self._send_json(
                    HTTPStatus.OK,
                    {
                        "configured": bool(config),
                        "install_url": (config or {}).get("install_url", ""),
                    },
                )
            if parsed.path == "/v1/worker/jobs":
                return self._send_json(HTTPStatus.OK, {"jobs": self.store.list_worker_jobs(self._token())})
            if parsed.path == "/v1/worker/upload-tickets":
                worker_job_id = query.get("worker_job_id", [""])[0]
                return self._send_json(
                    HTTPStatus.OK,
                    {"tickets": self.store.list_worker_upload_tickets(token=self._token(), worker_job_id=worker_job_id)},
                )
            if len(parts) == 4 and parts[0] == "v1" and parts[1] == "jobs" and parts[3] == "tickets":
                return self._send_json(
                    HTTPStatus.OK,
                    {"tickets": self.store.list_upload_tickets(user_id=self._user_id_from_request(query=query), team_job_id=parts[2])},
                )
            if len(parts) == 4 and parts[0] == "v1" and parts[1] == "jobs" and parts[3] == "workers":
                return self._send_json(
                    HTTPStatus.OK,
                    {"workers": self.store.list_team_job_workers(user_id=self._user_id_from_request(query=query), team_job_id=parts[2])},
                )
            if len(parts) == 5 and parts[0] == "v1" and parts[1] == "jobs" and parts[3] == "result":
                return self._send_bytes(
                    HTTPStatus.OK,
                    self.store.read_result_zip(
                        user_id=self._user_id_from_request(query=query),
                        team_job_id=parts[2],
                        worker_job_id=parts[4],
                    ),
                    content_type="application/zip",
                )
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            return self._handle_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            payload = self._read_json()
            if parsed.path == "/v1/auth/github/device/start":
                return self._send_json(HTTPStatus.OK, self._start_github_device_login())
            if parsed.path == "/v1/auth/github/device/poll":
                return self._send_json(HTTPStatus.OK, self._poll_github_device_login(str(payload.get("device_code", ""))))
            if parsed.path == "/v1/auth/logout":
                return self._send_json(HTTPStatus.OK, self.store.revoke_session(self._token()))
            if parsed.path == "/v1/teams":
                session = self._session()
                if session:
                    payload["owner_user_id"] = session["user_id"]
                    payload["owner_name"] = session["display_name"]
                return self._send_json(HTTPStatus.OK, self.store.create_team(**payload))
            if parsed.path == "/v1/teams/join":
                session = self._session()
                if session:
                    payload["user_id"] = session["user_id"]
                    payload["user_name"] = session["display_name"]
                return self._send_json(HTTPStatus.OK, self.store.join_team(**payload))
            if parsed.path == "/v1/team/worker/pause":
                return self._send_json(
                    HTTPStatus.OK,
                    self.store.set_worker_paused(self._user_id_from_request(payload), bool(payload.get("paused", True))),
                )
            if parsed.path == "/v1/team/worker/connect":
                user_id = self._user_id_from_request(payload)
                github_repo = str(payload.get("github_repo", ""))
                github_installation_id = None
                install_error = ""
                if github_repo.strip():
                    try:
                        github_installation_id = self._repo_installation_id(github_repo)
                    except Exception as exc:
                        install_error = str(exc)
                return self._send_json(
                    HTTPStatus.OK,
                    {
                        **self.store.connect_worker(
                        user_id,
                        github_repo=github_repo,
                        github_installation_id=github_installation_id,
                        kaggle_username=str(payload.get("kaggle_username", "")),
                        workflow_file=str(payload.get("workflow_file", "team-worker.yml")),
                        ),
                        "app_install_url": ((self._github_app_config() or {}).get("install_url", "")),
                        "app_installation_error": install_error,
                    },
                )
            if parsed.path == "/v1/team/worker/disconnect":
                return self._send_json(
                    HTTPStatus.OK,
                    self.store.disconnect_worker(self._user_id_from_request(payload)),
                )
            if parsed.path == "/v1/team/leave":
                return self._send_json(HTTPStatus.OK, self.store.leave_team(self._user_id_from_request(payload)))
            if parsed.path == "/v1/jobs":
                payload.setdefault("user_id", self._user_id_from_request(payload))
                result = self.store.create_team_job(**payload)
                return self._send_json(HTTPStatus.OK, result)
            if parsed.path == "/v1/worker/upload-tickets":
                return self._send_json(
                    HTTPStatus.OK,
                    self.store.record_upload_tickets(
                        token=self._token(),
                        worker_job_id=payload["worker_job_id"],
                        tickets=list(payload.get("tickets") or []),
                    ),
                )
            if parsed.path == "/v1/worker/status":
                return self._send_json(
                    HTTPStatus.OK,
                    self.store.mark_worker_job(
                        token=self._token(),
                        worker_job_id=payload["worker_job_id"],
                        status=payload["status"],
                        payload=payload.get("payload"),
                    ),
                )
            if parsed.path == "/v1/uploads/ack":
                result = self.store.acknowledge_upload(
                    user_id=self._user_id_from_request(payload),
                    ticket_id=payload["ticket_id"],
                )
                return self._send_json(HTTPStatus.OK, result)
            if len(parts) == 4 and parts[0] == "v1" and parts[1] == "jobs" and parts[3] == "dispatch":
                return self._send_json(
                    HTTPStatus.OK,
                    self._dispatch_team_job(
                        user_id=self._user_id_from_request(payload),
                        team_job_id=parts[2],
                        mode=str(payload.get("mode", "")).strip(),
                    ),
                )
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            return self._handle_error(exc)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) == 4 and parts[:3] == ["v1", "worker", "result"]:
                length = int(self.headers.get("Content-Length", "0"))
                data = self.rfile.read(length)
                return self._send_json(
                    HTTPStatus.OK,
                    self.store.save_result_zip(
                        token=self._token(),
                        worker_job_id=parts[3],
                        data=data,
                        storage_root=self.storage_root,
                    ),
                )
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            return self._handle_error(exc)


def build_server(
    *,
    host: str,
    port: int,
    db_path: Path | None = None,
    storage_root: Path | None = None,
) -> ThreadingHTTPServer:
    class Handler(TeamHandler):
        pass

    Handler.store = TeamStore(db_path)
    Handler.storage_root = storage_root or (Path.home() / ".ezscreen" / "team" / "results")
    Handler.storage_root.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer((host, port), Handler)


def serve(*, host: str = "127.0.0.1", port: int = 8787, db_path: Path | None = None, storage_root: Path | None = None) -> None:
    server = build_server(host=host, port=port, db_path=db_path, storage_root=storage_root)
    print(f"ezscreen team server listening on http://{host}:{port}")
    print("Kaggle keys are never accepted by this server; keep them in worker repo secrets.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping team server.")


