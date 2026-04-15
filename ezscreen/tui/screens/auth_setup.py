from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class AuthScreen(Screen):
    """Configure Kaggle credentials and optional NVIDIA NIM API key."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Auth Setup"])
        with Vertical(id="auth-form"):
            # ── Kaggle ──────────────────────────────────────────────────
            yield Label("Kaggle Credentials", classes="form-section")
            yield Label("Path to kaggle.json", classes="form-label")
            with Horizontal(classes="form-row"):
                yield Input(placeholder="~/.kaggle/kaggle.json", id="kaggle-path")
                yield Button("Validate", id="btn-kaggle-validate")
            yield Static("", id="kaggle-status", classes="form-status")

            # ── NIM ─────────────────────────────────────────────────────
            yield Label("NVIDIA NIM API Key  (optional)", classes="form-section")
            yield Label("NIM API Key", classes="form-label")
            with Horizontal(classes="form-row"):
                yield Input(placeholder="nvapi-...", password=True, id="nim-key")
                yield Button("Validate", id="btn-nim-validate")
            yield Static("", id="nim-status", classes="form-status")

            # ── Actions ─────────────────────────────────────────────────
            with Horizontal(classes="form-row form-actions"):
                yield Button("Save", id="btn-save", variant="primary")
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        dispatch = {
            "btn-kaggle-validate": self._start_kaggle_validate,
            "btn-nim-validate":    self._start_nim_validate,
            "btn-save":            self._save,
            "btn-cancel":          self.app.pop_screen,
        }
        if fn := dispatch.get(event.button.id):
            fn()

    # ------------------------------------------------------------------
    # Kaggle validation (thread worker)
    # ------------------------------------------------------------------

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
            msg = f"[#3fb950]Valid — logged in as {data['username']}[/#3fb950]"
        except Exception as exc:
            msg = f"[#f85149]{exc}[/#f85149]"
        self.app.call_from_thread(self.query_one("#kaggle-status", Static).update, msg)

    # ------------------------------------------------------------------
    # NIM validation (thread worker)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self) -> None:
        from ezscreen import auth
        kaggle_path = self.query_one("#kaggle-path", Input).value.strip()
        nim_key     = self.query_one("#nim-key",     Input).value.strip()
        creds = auth.load_credentials()
        if kaggle_path:
            creds["kaggle_json_path"] = kaggle_path
        if nim_key:
            creds["nim_api_key"] = nim_key
        auth.save_credentials(creds)
        self.app.notify("Credentials saved.", timeout=3)
        self.app.pop_screen()
