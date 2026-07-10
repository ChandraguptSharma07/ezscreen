from __future__ import annotations

import os
import webbrowser

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from ezscreen.team.client import (
    TeamClient,
    clear_local_session,
    default_server_url,
    load_local_state,
    save_local_state,
)
from ezscreen.tui.widgets.breadcrumb import Breadcrumb


def format_sign_in_error(exc: Exception) -> str:
    message = str(exc)
    if "HTTP 404" in message and "/v1/auth/github/device/start" in message or "not_found" in message:
        return "This team server is running older code. Restart or update the team server, then sign in again."
    if "EZSCREEN_GITHUB_CLIENT_ID" in message:
        return "GitHub sign-in is not configured on this team server. Set EZSCREEN_GITHUB_CLIENT_ID and restart it."
    return message


class TeamRunsScreen(Screen):
    """Create, join, and monitor one-team-only team runners."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, initial_action: str = "status") -> None:
        super().__init__()
        self.initial_action = initial_action
        self._pending_device_code = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Team"])
        with Horizontal(id="team-main"):
            with Vertical(id="team-control"):
                yield Label("Team", classes="section-title")
                yield Static("", id="team-service-text")
                yield Static("", id="team-auth-text")
                yield Static("", id="team-account-note", classes="form-status")
                yield Label("GitHub setup token", classes="form-label")
                yield Input(id="team-github-token", password=True, placeholder="stored only on this device for worker dispatch")
                with Horizontal(classes="form-row", id="team-signin-row"):
                    yield Button("Sign in with GitHub", id="btn-team-signin", variant="primary")
                    yield Button("I signed in", id="btn-team-auth-poll")
                with Vertical(id="team-worker-actions"):
                    with Horizontal(classes="form-row"):
                        yield Button("Connect Worker", id="btn-team-connect", variant="primary")
                        yield Button("Pause Worker", id="btn-team-pause")
                        yield Button("Resume Worker", id="btn-team-resume")
                    with Horizontal(classes="form-row"):
                        yield Button("Disconnect Worker", id="btn-team-disconnect")
                        yield Button("Leave Team", id="btn-team-leave", variant="error")
                with Vertical(id="team-setup-section"):
                    yield Label("Display name", classes="form-label", id="lbl-team-user")
                    yield Input(id="team-user", placeholder="Your name")
                    yield Label("Team name", classes="form-label", id="lbl-team-name")
                    yield Input(id="team-name", placeholder="My lab")
                    yield Button("Create Team", id="btn-team-create", variant="primary")
                    yield Label("Invite code", classes="form-label", id="lbl-team-invite")
                    yield Input(id="team-invite", placeholder="invite code")
                    yield Button("Join Team", id="btn-team-join")
                yield Static("", id="team-status-text")
            with Vertical(id="team-dashboard"):
                yield Label("Members", classes="section-title")
                yield DataTable(id="team-members", cursor_type="row")
                yield Label("Team Runs", classes="section-title")
                yield DataTable(id="team-jobs", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        state = load_local_state()
        self.query_one("#team-service-text", Static).update(f"[#6e7681]Team service: {default_server_url()}[/#6e7681]")
        self.query_one("#team-user", Input).value = state.get("user_name", "")
        self._refresh_auth_text(state)
        self.query_one("#team-worker-actions", Vertical).display = False
        self.refresh_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "btn-team-create": self.create_team,
            "btn-team-join": self.join_team,
            "btn-team-signin": self.start_sign_in,
            "btn-team-auth-poll": self.finish_sign_in,
            "btn-team-connect": self.connect_worker,
            "btn-team-pause": lambda: self.set_pause(True),
            "btn-team-resume": lambda: self.set_pause(False),
            "btn-team-disconnect": self.disconnect_worker,
            "btn-team-leave": self.leave_team,
        }
        fn = actions.get(event.button.id or "")
        if fn:
            fn()

    def _client(self) -> TeamClient:
        state = load_local_state()
        return TeamClient(default_server_url(), token=state.get("session_token"))

    def _set_status(self, message: str) -> None:
        self.query_one("#team-status-text", Static).update(message)

    def _refresh_auth_text(self, state: dict | None = None) -> None:
        state = state or load_local_state()
        note = self.query_one("#team-account-note", Static)
        note.update(
            "[#6e7681]Sign out only removes this device session. Use Connect Worker, Pause Worker, Disconnect Worker, or Leave Team to manage runner availability. ezscreen provisions a dedicated private worker repo and will not reuse an arbitrary research repo.[/#6e7681]"
        )
        if state.get("session_token") and state.get("github_login"):
            self.query_one("#team-auth-text", Static).update(
                f"[#3fb950]Signed in as {state['github_login']}[/#3fb950]"
            )
            return
        self.query_one("#team-auth-text", Static).update(
            "[#6e7681]Sign in once so team membership follows your GitHub account.[/#6e7681]"
        )

    def _sync_team_mode(self, in_team: bool) -> None:
        self.query_one("#team-setup-section", Vertical).display = not in_team
        worker_actions = self.query_one("#team-worker-actions", Vertical)
        worker_actions.display = in_team
        github_row = self.query_one("#team-signin-row", Horizontal)
        github_row.display = not bool(load_local_state().get("session_token")) or not in_team

    def _signed_in_identity(self) -> tuple[str | None, str]:
        state = load_local_state()
        user_id = state.get("user_id") if state.get("session_token") else None
        user_name = state.get("user_name") or self.query_one("#team-user", Input).value.strip()
        return user_id, user_name

    def start_sign_in(self) -> None:
        try:
            result = TeamClient(default_server_url()).start_github_device_login()
            self._pending_device_code = result["device_code"]
            uri = result.get("verification_uri_complete") or result.get("verification_uri")
            code = result.get("user_code", "")
            if uri:
                webbrowser.open(uri)
            self._set_status(f"[#79c0ff]GitHub sign-in code: {code}. Authorize it, then choose I signed in.[/#79c0ff]")
        except Exception as exc:
            self._set_status(f"[#f85149]{format_sign_in_error(exc)}[/#f85149]")

    def finish_sign_in(self) -> None:
        if not self._pending_device_code:
            self._set_status("[#e3b341]Start GitHub sign-in first.[/#e3b341]")
            return
        try:
            result = TeamClient(default_server_url()).poll_github_device_login(self._pending_device_code)
            if result.get("status") != "complete":
                self._set_status("[#e3b341]GitHub authorization is still pending.[/#e3b341]")
                return
            state = load_local_state()
            state.update(
                {
                    "server_url": default_server_url(),
                    "session_token": result["session_token"],
                    "user_id": result["user_id"],
                    "user_name": result.get("user_name") or result.get("github_login", ""),
                    "github_login": result.get("github_login", ""),
                }
            )
            save_local_state(state)
            self.query_one("#team-user", Input).value = state["user_name"]
            self._pending_device_code = ""
            self._refresh_auth_text(state)
            self._set_status("[#3fb950]Signed in with GitHub.[/#3fb950]")
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def create_team(self) -> None:
        try:
            server_url = default_server_url()
            owner_user_id, owner = self._signed_in_identity()
            team_name = self.query_one("#team-name", Input).value.strip()
            result = self._client().create_team(
                owner_name=owner, team_name=team_name, github_repo="", owner_user_id=owner_user_id
            )
            state = load_local_state()
            state.update(
                {
                    "server_url": server_url,
                    "user_id": result["owner_user_id"],
                    "user_name": owner,
                    "github_repo": "",
                    "team_id": result["team_id"],
                    "member_id": result["member_id"],
                }
            )
            save_local_state(
                state
            )
            self._set_status(f"[#3fb950]Invite code: {result['invite_code']}[/#3fb950]")
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def join_team(self) -> None:
        try:
            server_url = default_server_url()
            user_id, user = self._signed_in_identity()
            invite = self.query_one("#team-invite", Input).value.strip()
            result = self._client().join_team(
                invite_code=invite, user_name=user, github_repo="", user_id=user_id
            )
            state = load_local_state()
            state.update(
                {
                    "server_url": server_url,
                    "user_id": result["user_id"],
                    "user_name": user,
                    "github_repo": "",
                    "team_id": result["team_id"],
                    "member_id": result["member_id"],
                }
            )
            save_local_state(
                state
            )
            self._set_status("[#3fb950]Joined. Use Connect Worker to provision a dedicated private GitHub worker repo.[/#3fb950]")
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def connect_worker(self) -> None:
        state = load_local_state()
        if not state.get("session_token"):
            self._set_status("[#f85149]Sign in first so ezscreen can bind this worker to your account.[/#f85149]")
            return
        github_token = self.query_one("#team-github-token", Input).value.strip() or os.environ.get("GITHUB_TOKEN", "").strip()
        if not github_token:
            self._set_status("[#f85149]Enter a GitHub setup token or set GITHUB_TOKEN for this session.[/#f85149]")
            return
        try:
            from ezscreen.team.github import configure_worker_repo, provision_managed_worker_repo
            from ezscreen.team.local_credentials import (
                resolve_worker_kaggle_credentials,
                save_github_dispatch_token,
            )

            kaggle_creds = resolve_worker_kaggle_credentials(use_local=True)
            repo_info = provision_managed_worker_repo(github_token=github_token)
            worker_repo = repo_info["repo"]
            result = self._client().connect_worker(github_repo=worker_repo)
            configure_worker_repo(
                repo=worker_repo,
                github_token=github_token,
                kaggle_username=kaggle_creds.username,
                kaggle_key=kaggle_creds.key,
                team_token=result["team_token"],
                server_url=default_server_url(),
            )
            save_github_dispatch_token(worker_repo, github_token)
            state["github_repo"] = worker_repo
            save_local_state(state)
            self.query_one("#team-github-token", Input).value = ""
            action = "created" if repo_info.get("created") else "synced"
            self._set_status(
                f"[#3fb950]Worker connected via dedicated private repo {worker_repo} ({action}). "
                "Its dispatch token is stored only on this device.[/#3fb950]"
            )
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def set_pause(self, paused: bool) -> None:
        state = load_local_state()
        try:
            self._client().request(
                "POST",
                "/v1/team/worker/pause",
                {"user_id": state["user_id"], "paused": paused},
            )
            self._set_status("[#e3b341]Worker paused.[/#e3b341]" if paused else "[#3fb950]Worker enabled.[/#3fb950]")
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def disconnect_worker(self) -> None:
        state = load_local_state()
        try:
            from ezscreen.team.local_credentials import remove_github_dispatch_token

            repo = str(state.get("github_repo") or "")
            self._client().disconnect_worker(user_id=state.get("user_id"))
            if repo:
                remove_github_dispatch_token(repo)
            state["github_repo"] = ""
            save_local_state(state)
            self._set_status("[#e3b341]Worker disconnected. Team membership is unchanged; reconnect later to receive jobs again.[/#e3b341]")
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def leave_team(self) -> None:
        state = load_local_state()
        try:
            self._client().request("POST", "/v1/team/leave", {"user_id": state["user_id"]})
            clear_local_session(preserve_keys={"server_url", "user_name", "session_token", "github_login"})
            self._set_status("[#e3b341]Left team; future worker jobs disabled.[/#e3b341]")
            self.refresh_status()
        except Exception as exc:
            self._set_status(f"[#f85149]{exc}[/#f85149]")

    def refresh_status(self) -> None:
        members = self.query_one("#team-members", DataTable)
        jobs = self.query_one("#team-jobs", DataTable)
        members.clear(columns=True)
        jobs.clear(columns=True)
        members.add_columns("Name", "Repo", "State")
        jobs.add_columns("Run", "Status", "Created")
        state = load_local_state()
        if not state.get("server_url") or not state.get("user_id"):
            self._sync_team_mode(False)
            members.add_row("-", "-", "not joined")
            return
        try:
            status = TeamClient(state["server_url"]).status(state["user_id"])
        except Exception as exc:
            self._sync_team_mode(bool(state.get("team_id")))
            members.add_row("-", "-", "offline")
            self._set_status(f"[#f85149]{exc}[/#f85149]")
            return
        if not status.get("team"):
            self._sync_team_mode(False)
            members.add_row("-", "-", "not joined")
            return
        self._sync_team_mode(True)
        for member in status.get("members", []):
            if not (member.get("github_repo") or "").strip():
                state_label = "disconnected"
            else:
                state_label = "paused" if member.get("paused") else "enabled"
            members.add_row(
                member.get("display_name", ""),
                member.get("github_repo") or "-",
                state_label,
            )
        for job in status.get("jobs", []):
            jobs.add_row(job.get("run_id", ""), job.get("status", ""), job.get("created_at", ""))
