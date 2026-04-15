from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class ValidateScreen(Screen):
    """Run DiffDock-L pose re-scoring on top hits from a finished run."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Validate"])
        with Vertical(id="validate-form"):
            yield Label("Receptor file  (.pdb)", classes="form-label")
            yield Input(placeholder="/path/to/receptor.pdb", id="val-receptor")

            yield Label("Hits file  (.sdf or .smi)", classes="form-label")
            yield Input(placeholder="/path/to/hits.sdf", id="val-hits")

            yield Label("Output directory", classes="form-label")
            yield Input(placeholder="/path/to/output/", id="val-output")

            yield Label("NIM API Key  (leave blank to use saved key)", classes="form-label")
            yield Input(placeholder="nvapi-...", password=True, id="val-nim-key")

            with Horizontal(classes="form-row form-actions"):
                yield Button("Run Validation", id="btn-validate", variant="primary")
                yield Static("", id="val-status", classes="form-status")

            yield Label("Progress", classes="form-section")
            yield RichLog(id="val-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._prefill_nim_key()

    def _prefill_nim_key(self) -> None:
        try:
            from ezscreen import auth
            creds = auth.load_credentials()
            key = auth.get_nim_key(creds)
            if key:
                self.query_one("#val-nim-key", Input).value = key
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-validate":
            self._start()

    def _start(self) -> None:
        receptor = self.query_one("#val-receptor", Input).value.strip()
        hits     = self.query_one("#val-hits",     Input).value.strip()
        out_dir  = self.query_one("#val-output",   Input).value.strip()

        if not receptor:
            self._set_status("[#f85149]Enter a receptor file.[/#f85149]")
            return
        if not Path(receptor).expanduser().exists():
            self._set_status(f"[#f85149]Receptor not found: {receptor}[/#f85149]")
            return
        if not hits:
            self._set_status("[#f85149]Enter a hits file.[/#f85149]")
            return
        if not Path(hits).expanduser().exists():
            self._set_status(f"[#f85149]Hits file not found: {hits}[/#f85149]")
            return
        if not out_dir:
            self._set_status("[#f85149]Enter an output directory.[/#f85149]")
            return

        self._set_status("[#e3b341]Submitting...[/#e3b341]")
        self.query_one("#btn-validate").disabled = True
        self.query_one("#val-log", RichLog).clear()

        nim_key = self.query_one("#val-nim-key", Input).value.strip() or None

        self.run_worker(
            lambda: self._do_validate(
                Path(receptor).expanduser(),
                Path(hits).expanduser(),
                Path(out_dir).expanduser(),
                nim_key,
            ),
            thread=True,
        )

    def _do_validate(
        self,
        receptor: Path,
        hits: Path,
        out_dir: Path,
        nim_key: str | None,
    ) -> None:
        try:
            from ezscreen.nim.diffdock import run_diffdock_l

            out_dir.mkdir(parents=True, exist_ok=True)

            self.app.call_from_thread(self._log, "[#79c0ff]Connecting to NIM DiffDock-L...[/#79c0ff]")
            result = run_diffdock_l(
                receptor_path=str(receptor),
                ligand_path=str(hits),
                output_dir=str(out_dir),
                nim_key=nim_key,
                progress_cb=lambda msg: self.app.call_from_thread(self._log, msg),
            )
            self.app.call_from_thread(self._finish, result, out_dir)
        except Exception as exc:
            self.app.call_from_thread(self._error, str(exc))

    def _log(self, msg: str) -> None:
        self.query_one("#val-log", RichLog).write(msg)

    def _finish(self, result: dict, out_dir: Path) -> None:
        n = result.get("poses_written", "?")
        self._set_status(f"[#3fb950]Done — {n} poses written[/#3fb950]")
        self._log(f"[#3fb950]Output → {out_dir}[/#3fb950]")
        self.query_one("#btn-validate").disabled = False

    def _error(self, msg: str) -> None:
        self._set_status(f"[#f85149]Error: {msg}[/#f85149]")
        self._log(f"[#f85149]{msg}[/#f85149]")
        self.query_one("#btn-validate").disabled = False

    def _set_status(self, msg: str) -> None:
        self.query_one("#val-status", Static).update(msg)
