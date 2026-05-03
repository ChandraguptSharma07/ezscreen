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

            yield Label("Ligand Pre-filter", classes="form-section")
            yield Label("Enable GPU size filter", classes="form-label")
            yield Switch(id="cfg-gpu-filter-enable")
            yield Label("Max heavy atoms", classes="form-label")
            yield Input(id="cfg-max-ha", placeholder="70")
            yield Label("Max mol. weight (Da)", classes="form-label")
            yield Input(id="cfg-max-mw", placeholder="700.0")
            yield Label("Max rotatable bonds", classes="form-label")
            yield Input(id="cfg-max-rb", placeholder="20")
            yield Label("Run 3D prep on Kaggle GPU (disable to prep locally before upload)", classes="form-label")
            yield Switch(id="cfg-prep-on-kaggle")
            yield Label("MMFF minimisation", classes="form-label")
            yield Switch(id="cfg-mmff-converge", name="Run to convergence (maxIters=0, faster)")
            yield Label("Fixed MMFF iterations (ignored when convergence mode on)", classes="form-label")
            yield Input(id="cfg-mmff-iters", placeholder="200")

            yield Label("Local Docking", classes="form-section")
            yield Label("Enable score filter", classes="form-label")
            yield Switch(id="cfg-score-floor-enable")
            yield Label("Score floor (kcal/mol)", classes="form-label")
            yield Input(id="cfg-score-floor", placeholder="-15.0")
            yield Label("Score ceiling (kcal/mol)", classes="form-label")
            yield Input(id="cfg-score-ceiling", placeholder="0.0")
            yield Label("Exhaustiveness (local Vina)", classes="form-label")
            yield Input(id="cfg-exhaustiveness", placeholder="4")
            yield Label("CPU cores (0 = auto)", classes="form-label")
            yield Input(id="cfg-cpu-cores", placeholder="0")

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

        pc = cfg.get("prep", {})
        self.query_one("#cfg-gpu-filter-enable", Switch).value = bool(pc.get("enable_gpu_size_filter", True))
        self.query_one("#cfg-max-ha", Input).value = str(pc.get("max_heavy_atoms",     70))
        self.query_one("#cfg-max-mw", Input).value = str(pc.get("max_mw",           700.0))
        self.query_one("#cfg-max-rb", Input).value = str(pc.get("max_rotatable_bonds", 20))
        self.query_one("#cfg-prep-on-kaggle", Switch).value = bool(pc.get("prep_on_kaggle", True))
        mmff = int(pc.get("mmff_max_iters", 0))
        self.query_one("#cfg-mmff-converge", Switch).value = (mmff == 0)
        self.query_one("#cfg-mmff-iters", Input).value = str(mmff if mmff > 0 else 200)

        lc = cfg.get("local", {})
        self.query_one("#cfg-score-floor-enable", Switch).value = bool(lc.get("enable_score_floor", True))
        self.query_one("#cfg-score-floor",   Input).value = str(lc.get("score_floor",   -15.0))
        self.query_one("#cfg-score-ceiling", Input).value = str(lc.get("score_ceiling",   0.0))
        self.query_one("#cfg-exhaustiveness", Input).value = str(lc.get("exhaustiveness",  4))
        self.query_one("#cfg-cpu-cores",      Input).value = str(lc.get("cpu_cores",        0))

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
            "prep": {
                "enable_gpu_size_filter": self.query_one("#cfg-gpu-filter-enable", Switch).value,
                "max_heavy_atoms":        _i("#cfg-max-ha", 70),
                "max_mw":                 _f("#cfg-max-mw", 700.0),
                "max_rotatable_bonds":    _i("#cfg-max-rb", 20),
                "prep_on_kaggle":         self.query_one("#cfg-prep-on-kaggle", Switch).value,
                "mmff_max_iters":         0 if self.query_one("#cfg-mmff-converge", Switch).value else _i("#cfg-mmff-iters", 200),
            },
            "local": {
                "enable_score_floor": self.query_one("#cfg-score-floor-enable", Switch).value,
                "score_floor":        _f("#cfg-score-floor",   -15.0),
                "score_ceiling":      _f("#cfg-score-ceiling",   0.0),
                "exhaustiveness":     _i("#cfg-exhaustiveness",  4),
                "cpu_cores":          _i("#cfg-cpu-cores",        0),
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
