from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Switch

from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_DEPTH_OPTIONS: list[tuple[str, str]] = [
    ("Fast — triage only",           "Fast"),
    ("Balanced — standard VS ★",     "Balanced"),
    ("Thorough — flexible ligands",  "Thorough"),
]


class SettingsScreen(Screen):
    """Edit ~/.ezscreen/config.toml preferences."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Settings"])
        with Vertical(id="settings-form"):
            yield Label("Run Defaults", classes="form-section")
            yield Label("Default pH", classes="form-label")
            yield Input(id="cfg-ph", placeholder="7.4")
            yield Label("ADMET pre-filter", classes="form-label")
            yield Switch(id="cfg-admet")
            yield Label("Default search depth", classes="form-label")
            yield Select(_DEPTH_OPTIONS, id="cfg-depth")
            yield Label("Shard retry limit", classes="form-label")
            yield Input(id="cfg-retries", placeholder="3")
            yield Label("Auto-resume threshold (%)", classes="form-label")
            yield Input(id="cfg-resume", placeholder="10")

            yield Label("Docking Defaults", classes="form-section")
            yield Label("Box padding (Å)", classes="form-label")
            yield Input(id="cfg-padding", placeholder="5.0")
            yield Label("Enumerate tautomers", classes="form-label")
            yield Switch(id="cfg-tautomers")

            yield Label("Notifications", classes="form-section")
            yield Label("Desktop notifications", classes="form-label")
            yield Switch(id="cfg-desktop-notify")
            yield Label("SMTP host", classes="form-label")
            yield Input(id="cfg-smtp-host", placeholder="smtp.gmail.com")
            yield Label("SMTP port", classes="form-label")
            yield Input(id="cfg-smtp-port", placeholder="587")
            yield Label("From address", classes="form-label")
            yield Input(id="cfg-smtp-from", placeholder="me@example.com")
            yield Label("To address", classes="form-label")
            yield Input(id="cfg-smtp-to", placeholder="me@example.com")

            with Horizontal(classes="form-row form-actions"):
                yield Button("Save",              id="btn-save",  variant="primary")
                yield Button("Reset to defaults", id="btn-reset")
                yield Button("Cancel",            id="btn-cancel")
        yield Footer()

    def on_mount(self) -> None:
        self._populate()

    def _populate(self) -> None:
        from ezscreen import config
        try:
            cfg = config.load()
        except Exception:
            cfg = config.DEFAULTS

        r = cfg.get("run", {})
        d = cfg.get("defaults", {})

        self.query_one("#cfg-ph",       Input).value  = str(r.get("default_ph",          7.4))
        self.query_one("#cfg-admet",    Switch).value = bool(r.get("admet_pre_filter",    True))
        self.query_one("#cfg-retries",  Input).value  = str(r.get("shard_retry_limit",    3))
        self.query_one("#cfg-resume",   Input).value  = str(r.get("auto_resume_threshold", 10))
        self.query_one("#cfg-padding",  Input).value  = str(d.get("box_padding",          5.0))
        self.query_one("#cfg-tautomers", Switch).value = bool(d.get("enumerate_tautomers", False))

        n = cfg.get("notify", {})
        self.query_one("#cfg-desktop-notify", Switch).value = bool(n.get("desktop_enabled", False))
        self.query_one("#cfg-smtp-host", Input).value = str(n.get("smtp_host", ""))
        self.query_one("#cfg-smtp-port", Input).value = str(n.get("smtp_port", "587"))
        self.query_one("#cfg-smtp-from", Input).value = str(n.get("from_address", ""))
        self.query_one("#cfg-smtp-to",   Input).value = str(n.get("to_address", ""))

        depth = r.get("default_search_depth", "Balanced")
        sel = self.query_one("#cfg-depth", Select)
        if depth in {v for _, v in _DEPTH_OPTIONS}:
            sel.value = depth

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self._save()
        elif event.button.id == "btn-reset":
            self._reset()
        elif event.button.id == "btn-cancel":
            self.app.pop_screen()

    def _save(self) -> None:
        from ezscreen import config

        def _f(wid: str, default: float) -> float:
            try:
                return float(self.query_one(wid, Input).value)
            except Exception:
                return default

        def _i(wid: str, default: int) -> int:
            try:
                return int(self.query_one(wid, Input).value)
            except Exception:
                return default

        depth_val = self.query_one("#cfg-depth", Select).value
        depth = str(depth_val) if depth_val != Select.BLANK else "Balanced"

        cfg = {
            "run": {
                "default_ph":            _f("#cfg-ph",      7.4),
                "admet_pre_filter":      self.query_one("#cfg-admet",    Switch).value,
                "default_search_depth":  depth,
                "shard_retry_limit":     _i("#cfg-retries", 3),
                "auto_resume_threshold": _i("#cfg-resume",  10),
            },
            "defaults": {
                "box_padding":          _f("#cfg-padding",   5.0),
                "enumerate_tautomers":  self.query_one("#cfg-tautomers", Switch).value,
            },
            "notify": {
                "desktop_enabled": self.query_one("#cfg-desktop-notify", Switch).value,
                "smtp_host":       self.query_one("#cfg-smtp-host", Input).value.strip(),
                "smtp_port":       _i("#cfg-smtp-port", 587),
                "from_address":    self.query_one("#cfg-smtp-from", Input).value.strip(),
                "to_address":      self.query_one("#cfg-smtp-to",   Input).value.strip(),
            },
        }
        config.save(cfg)
        self.app.notify("Settings saved.", timeout=3)
        self.app.pop_screen()

    def _reset(self) -> None:
        from ezscreen import config
        config.save(config.DEFAULTS)
        self.app.notify("Reset to defaults.", timeout=3)
        self._populate()
