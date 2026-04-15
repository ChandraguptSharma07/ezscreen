from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Static

from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class AdmetScreen(Screen):
    """Run ADMET filtering on an SDF or SMILES file."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "ADMET Filter"])
        with Vertical(id="admet-form"):
            yield Label("Input file  (.sdf or .smi)", classes="form-label")
            yield Input(placeholder="/path/to/library.sdf", id="admet-input")

            yield Label("Filters", classes="form-section")
            yield Checkbox("Lipinski Rule of Five",       id="f-lipinski",     value=True)
            yield Checkbox("PAINS alerts",                id="f-pains",        value=True)
            yield Checkbox("Brenk toxicophores",          id="f-toxicophores", value=True)
            yield Checkbox("Veber oral bioavailability",  id="f-veber",        value=True)
            yield Checkbox("Egan BBB permeability",       id="f-egan",         value=False)

            yield Button("Run Filter", id="btn-run", variant="primary")
            yield Static("", id="admet-results", classes="results-panel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self._run()

    def _run(self) -> None:
        input_path = self.query_one("#admet-input", Input).value.strip()
        if not input_path:
            self.query_one("#admet-results", Static).update(
                "[#f85149]Enter an input file path.[/#f85149]"
            )
            return

        p = Path(input_path).expanduser()
        if not p.exists():
            self.query_one("#admet-results", Static).update(
                f"[#f85149]File not found: {p}[/#f85149]"
            )
            return

        self.query_one("#admet-results", Static).update(
            "[#e3b341]Filtering...[/#e3b341]"
        )
        self.query_one("#btn-run").disabled = True
        self.run_worker(lambda: self._do_filter(p), thread=True)

    def _do_filter(self, input_path: Path) -> None:
        from ezscreen.admet.filter import FilterConfig, filter_library

        output_path = input_path.with_stem(input_path.stem + "_admet")
        cfg = FilterConfig(
            lipinski=     self.query_one("#f-lipinski",     Checkbox).value,
            pains=        self.query_one("#f-pains",        Checkbox).value,
            toxicophores= self.query_one("#f-toxicophores", Checkbox).value,
            veber=        self.query_one("#f-veber",        Checkbox).value,
            egan_bbb=     self.query_one("#f-egan",         Checkbox).value,
        )
        try:
            result = filter_library(str(input_path), str(output_path), cfg)
            msg = self._format_result(result, output_path)
        except Exception as exc:
            msg = f"[#f85149]Error: {exc}[/#f85149]"

        self.app.call_from_thread(self._show_result, msg)

    def _format_result(self, result: dict, output_path: Path) -> str:
        total    = result["total_input"]
        removed  = result["admet_removed"]
        passed   = total - removed
        lines = [
            f"[#3fb950]Done.[/#3fb950]  {passed:,} passed  /  {removed:,} removed  /  {total:,} total",
        ]
        breakdown = result.get("admet_breakdown", {})
        for rule, count in breakdown.items():
            if count:
                lines.append(f"  [#6e7681]{rule.replace('_', ' ')}: {count:,}[/#6e7681]")
        lines.append(f"\n[#6e7681]Output → {output_path}[/#6e7681]")
        return "\n".join(lines)

    def _show_result(self, msg: str) -> None:
        self.query_one("#admet-results", Static).update(msg)
        self.query_one("#btn-run").disabled = False
