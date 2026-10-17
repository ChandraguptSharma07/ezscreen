from __future__ import annotations

import webbrowser
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from ezscreen.team.client import (
    TeamClient,
    clear_local_session,
    default_server_url,
    load_local_state,
    save_local_state,
)
from ezscreen.tui.screens.team_runs import format_sign_in_error
from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class AuthScreen(Screen):
    """Required account setup: GitHub sign-in plus Kaggle credentials."""

    BINDINGS = [Binding("escape", "dismiss_or_warn", "Back")]

    def __init__(self, required: bool = False) -> None:
        super().__init__()
        self.required = required
        self._pending_device_code = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Account"] if self.required else ["Home", "Account"])
        with Vertical(id="auth-form"):
            yield Label("ezscreen Account", classes="form-section")
            yield Static("", id="github-status", classes="form-status")
            yield Static(f"Team service: {default_server_url()}", id="team-service-text", classes="form-status")
            with Horizontal(classes="form-row"):
                yield Button("Sign in with GitHub", id="btn-team-signin", variant="primary")
                yield Button("I signed in", id="btn-team-auth-poll")
                yield Button("Log out", id="btn-team-signout")

            yield Label("Kaggle Credentials", classes="form-section")
            yield Label("Path to kaggle.json", classes="form-label")
            with Horizontal(classes="form-row"):
                yield Input(placeholder="~/.kaggle/kaggle.json", id="kaggle-path")
                yield Button("Validate", id="btn-kaggle-validate")
            yield Static("", id="kaggle-status", classes="form-status")

            yield Label("NVIDIA NIM API Key  (optional)", classes="form-section")
            yield Label("NIM API Key", classes="form-label")
            with Horizontal(classes="form-row"):
                yield Input(placeholder="nvapi-...", password=True, id="nim-key")
                yield Button("Validate", id="btn-nim-validate")
            yield Static("", id="nim-status", classes="form-status")

            if self.required:
                yield Static(
                    "[#e3b341]GitHub sign-in and Kaggle credentials are required before using ezscreen.[/#e3b341]",
                    id="account-required-text",
                    classes="form-status",
                )
            yield Static(
                "[#6e7681]Sign out removes only this device session. It does not leave your team or pause your worker. See PRIVACY.md and TERMS.md for data-use terms.[/#6e7681]",
                id="account-session-note",
                classes="form-status",
            )
            with Horizontal(classes="form-row form-actions"):
                yield Button("Save", id="btn-save", variant="primary")
                if not self.required:
                    yield Button("Cancel", id="btn-cancel")
        yield Footer()

    def on_mount(self) -> None:
        from ezscreen import auth

        try:
            creds = auth.load_credentials()
            if path := creds.get("kaggle_json_path"):
                self.query_one("#kaggle-path", Input).value = path
            if key := auth.get_nim_key(creds):
                self.query_one("#nim-key", Input).value = key
        except Exception:
            pass
        self._refresh_github_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        dispatch = {
            "btn-team-signin": self._start_sign_in,
            "btn-team-auth-poll": self._finish_sign_in,
            "btn-team-signout": self._sign_out,
            "btn-kaggle-validate": self._start_kaggle_validate,
            "btn-nim-validate": self._start_nim_validate,
            "btn-save": self._save,
            "btn-cancel": self.app.pop_screen,
        }
        if fn := dispatch.get(event.button.id):
            fn()

    def action_dismiss_or_warn(self) -> None:
        if self.required:
            self.app.notify("Complete GitHub sign-in and Kaggle setup before using ezscreen.", severity="warning", timeout=4)
            return
        self.app.pop_screen()

    def _refresh_github_status(self) -> None:
        state = load_local_state()
        status = self.query_one("#github-status", Static)
        if state.get("session_token") and state.get("github_login"):
            account_name = state.get('user_name') or state['github_login']
            status.update(
                f"[#3fb950]ezscreen account: {account_name} (@{state['github_login']})[/#3fb950]"
            )
            return
        status.update("[#6e7681]Sign in with GitHub to create or access your ezscreen account.[/#6e7681]")

    def _start_sign_in(self) -> None:
        try:
            result = TeamClient(default_server_url()).start_github_device_login()
            self._pending_device_code = result["device_code"]
            uri = result.get("verification_uri_complete") or result.get("verification_uri")
            code = result.get("user_code", "")
            if uri:
                webbrowser.open(uri)
            self.query_one("#github-status", Static).update(
                f"[#79c0ff]GitHub sign-in code: {code}. Authorize it, then choose I signed in to finish your ezscreen account login.[/#79c0ff]"
            )
        except Exception as exc:
            self.query_one("#github-status", Static).update(f"[#f85149]{format_sign_in_error(exc)}[/#f85149]")

    def _finish_sign_in(self) -> None:
        if not self._pending_device_code:
            self.query_one("#github-status", Static).update("[#e3b341]Start GitHub sign-in first.[/#e3b341]")
            return
        try:
            result = TeamClient(default_server_url()).poll_github_device_login(self._pending_device_code)
            if result.get("status") != "complete":
                self.query_one("#github-status", Static).update(
                    "[#e3b341]GitHub authorization is still pending.[/#e3b341]"
                )
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
            self._pending_device_code = ""
            self.query_one("#github-status", Static).update(
                f"[#3fb950]Logged in to ezscreen as {state.get('user_name') or state.get('github_login')}.[/#3fb950]"
            )
        except Exception as exc:
            self.query_one("#github-status", Static).update(f"[#f85149]{exc}[/#f85149]")

    def _sign_out(self) -> None:
        state = load_local_state()
        session_token = state.get("session_token")
        if session_token:
            try:
                TeamClient(default_server_url(), token=session_token).logout()
            except Exception as exc:
                self.query_one("#github-status", Static).update(
                    f"[#f85149]Could not revoke ezscreen session on server: {exc}[/#f85149]"
                )
                return
        clear_local_session()
        self._pending_device_code = ""
        self._refresh_github_status()
        self.query_one("#github-status", Static).update(
            "[#e3b341]Signed out of the ezscreen account session.[/#e3b341]"
        )
        if self.required:
            self.app.notify("Sign in again to continue using ezscreen.", severity="warning", timeout=4)
            return
        self.app.notify("Signed out.", timeout=3)

    def _start_kaggle_validate(self) -> None:
        path_str = self.query_one("#kaggle-path", Input).value.strip()
        if not path_str:
            self.query_one("#kaggle-status", Static).update(
                "[#f85149]Enter a path first.[/#f85149]"
            )
            return
        self.query_one("#kaggle-status", Static).update(
            "[#e3b341]Validating...[/#e3b341]"
        )
        self.run_worker(lambda: self._do_kaggle_validate(path_str), thread=True)

    def _do_kaggle_validate(self, path_str: str) -> None:
        from ezscreen import auth
        try:
            path = Path(path_str).expanduser()
            data = auth.validate_kaggle_json(path)
            auth._live_kaggle_check(data)
            msg = f"[#3fb950]Valid - logged in as {data['username']}[/#3fb950]"
        except Exception as exc:
            msg = f"[#f85149]{exc}[/#f85149]"
        self.app.call_from_thread(self.query_one("#kaggle-status", Static).update, msg)

    def _start_nim_validate(self) -> None:
        key = self.query_one("#nim-key", Input).value.strip()
        if not key:
            self.query_one("#nim-status", Static).update(
                "[#f85149]Enter a key first.[/#f85149]"
            )
            return
        self.query_one("#nim-status", Static).update(
            "[#e3b341]Validating...[/#e3b341]"
        )
        self.run_worker(lambda: self._do_nim_validate(key), thread=True)

    def _do_nim_validate(self, key: str) -> None:
        from ezscreen import auth
        try:
            auth.validate_nim_key(key)
            msg = "[#3fb950]Valid.[/#3fb950]"
        except Exception as exc:
            msg = f"[#f85149]{exc}[/#f85149]"
        self.app.call_from_thread(self.query_one("#nim-status", Static).update, msg)

    def _save(self) -> None:
        from ezscreen import auth

        kaggle_path = self.query_one("#kaggle-path", Input).value.strip()
        nim_key = self.query_one("#nim-key", Input).value.strip()
        state = load_local_state()
        if not state.get("session_token") or not state.get("github_login"):
            self.app.notify("Sign in with GitHub first.", severity="error", timeout=4)
            return
        if not kaggle_path:
            self.app.notify("Provide kaggle.json before continuing.", severity="error", timeout=4)
            return
        try:
            auth.validate_kaggle_json(Path(kaggle_path).expanduser())
        except Exception as exc:
            self.app.notify(str(exc), severity="error", timeout=5)
            return
        creds = auth.load_credentials()
        creds["kaggle_json_path"] = kaggle_path
        if nim_key:
            creds["nim_api_key"] = nim_key
        auth.save_credentials(creds)
        self.app.notify("Account settings saved.", timeout=3)
        if self.required:
            from ezscreen.tui.screens.home import HomeScreen

            self.app.switch_screen(HomeScreen())
            return
        self.app.pop_screen()
