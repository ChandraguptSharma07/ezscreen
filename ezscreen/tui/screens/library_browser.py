from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
)

from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_SOURCE_ZINC   = "zinc"
_SOURCE_CHEMBL = "chembl"

_ZINC_PRESETS: list[tuple[str, str]] = [
    ("Drug-like  (MW 250–500, logP -1–5)", "drug-like"),
    ("Lead-like  (MW 150–350, logP -3–3)", "lead-like"),
    ("Fragment-like  (MW 100–250, logP -3–3)", "fragment-like"),
]

_ZINC_SIZES: list[tuple[str, str]] = [
    ("1 000 compounds",  "1k"),
    ("10 000 compounds", "10k"),
    ("100 000 compounds", "100k"),
    ("Custom count",     "custom"),
]


class LibraryBrowserScreen(Screen[str | None]):
    """Download a compound library from ZINC or ChEMBL.

    When pushed with a result callback, dismisses with the output path on
    successful download so the caller can populate a file-path input.
    """

    BINDINGS = [Binding("escape", "action_dismiss_none", "Back")]

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Library Browser"])
        with Vertical(id="lib-form"):
            yield Label("Source", classes="form-section")
            with RadioSet(id="lib-source"):
                yield RadioButton("ZINC15 (drug-like / lead-like / fragment-like)", value=True, id="src-zinc")
                yield RadioButton("ChEMBL (actives by UniProt ID)", id="src-chembl")

            # --- ZINC options ---
            with Vertical(id="zinc-opts"):
                yield Label("Property preset", classes="form-label")
                yield Select(_ZINC_PRESETS, id="zinc-preset")
                yield Label("Library size", classes="form-label")
                with RadioSet(id="zinc-size"):
                    yield RadioButton("1 000 compounds",   id="sz-1k")
                    yield RadioButton("10 000 compounds",  value=True, id="sz-10k")
                    yield RadioButton("100 000 compounds", id="sz-100k")
                    yield RadioButton("Custom count",      id="sz-custom")
                yield Label("Custom count", classes="form-label", id="custom-count-label")
                yield Input(placeholder="e.g. 25000", id="zinc-custom-count")
                yield Label("Purchasable compounds only", classes="form-label")
                yield Switch(id="zinc-purchasable", value=True)

            # --- ChEMBL options ---
            with Vertical(id="chembl-opts"):
                yield Label("UniProt accession", classes="form-label")
                yield Input(placeholder="e.g. P00533  (EGFR)", id="chembl-uniprot")
                yield Label("IC50 threshold (µM)", classes="form-label")
                yield Input(placeholder="1.0", id="chembl-ic50")
                yield Label("Max compounds (leave blank for all)", classes="form-label")
                yield Input(placeholder="e.g. 500", id="chembl-max")

            yield Label("Output path", classes="form-label")
            yield Input(placeholder="~/Downloads/library.smi", id="lib-output")

            yield Button("Download", id="btn-download", variant="primary")
            yield Static("", id="lib-status", classes="results-panel")
            yield Button("Use this file", id="btn-use", variant="success", display=False)
        yield Footer()

    def on_mount(self) -> None:
        self._sync_source_visibility()
        self._sync_custom_count_visibility()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "lib-source":
            self._sync_source_visibility()
        elif event.radio_set.id == "zinc-size":
            self._sync_custom_count_visibility()

    def _sync_source_visibility(self) -> None:
        src = self._selected_source()
        self.query_one("#zinc-opts").display   = (src == _SOURCE_ZINC)
        self.query_one("#chembl-opts").display = (src == _SOURCE_CHEMBL)

    def _sync_custom_count_visibility(self) -> None:
        is_custom = self.query_one("#sz-custom", RadioButton).value
        self.query_one("#custom-count-label").display  = is_custom
        self.query_one("#zinc-custom-count").display   = is_custom

    def _selected_source(self) -> str:
        return _SOURCE_CHEMBL if self.query_one("#src-chembl", RadioButton).value else _SOURCE_ZINC

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-download":
            self._start_download()
        elif event.button.id == "btn-use":
            path = getattr(event.button, "_success_path", None)
            self.dismiss(str(path) if path else None)

    def _start_download(self) -> None:
        out_raw = self.query_one("#lib-output", Input).value.strip()
        if not out_raw:
            self._status("[#f85149]Enter an output path.[/#f85149]")
            return

        output_path = Path(out_raw).expanduser()
        self._status("[#e3b341]Downloading...[/#e3b341]")
        self.query_one("#btn-download").disabled = True

        src = self._selected_source()
        if src == _SOURCE_ZINC:
            self.run_worker(lambda: self._download_zinc(output_path), thread=True)
        else:
            self.run_worker(lambda: self._download_chembl(output_path), thread=True)

    def _download_zinc(self, output_path: Path) -> None:
        from ezscreen.libraries.zinc import download_zinc_library

        preset_val = self.query_one("#zinc-preset", Select).value
        preset = str(preset_val) if preset_val != Select.BLANK else "drug-like"
        purchasable = self.query_one("#zinc-purchasable").value  # type: ignore[attr-defined]

        # resolve size / custom count
        if self.query_one("#sz-custom", RadioButton).value:
            raw = self.query_one("#zinc-custom-count", Input).value.strip()
            try:
                custom_count = int(raw)
            except ValueError:
                self.app.call_from_thread(
                    self._finish, "[#f85149]Custom count must be a whole number.[/#f85149]"
                )
                return
            size, count = "custom", custom_count
        elif self.query_one("#sz-1k", RadioButton).value:
            size, count = "1k", None
        elif self.query_one("#sz-100k", RadioButton).value:
            size, count = "100k", None
        else:
            size, count = "10k", None

        try:
            n = download_zinc_library(
                output_path,
                size=size,
                count=count,
                preset=preset,
                purchasable=purchasable,
            )
            msg = f"[#3fb950]Done.[/#3fb950]  {n:,} compounds saved to {output_path}"
            self.app.call_from_thread(self._finish, msg, output_path)
            return
        except Exception as exc:
            msg = f"[#f85149]Error: {exc}[/#f85149]"

        self.app.call_from_thread(self._finish, msg)

    def _download_chembl(self, output_path: Path) -> None:
        from ezscreen.libraries.chembl import fetch_chembl_actives

        uniprot = self.query_one("#chembl-uniprot", Input).value.strip().upper()
        if not uniprot:
            self.app.call_from_thread(
                self._finish, "[#f85149]Enter a UniProt accession.[/#f85149]"
            )
            return

        try:
            ic50 = float(self.query_one("#chembl-ic50", Input).value.strip() or "1.0")
        except ValueError:
            ic50 = 1.0

        max_raw = self.query_one("#chembl-max", Input).value.strip()
        max_compounds = int(max_raw) if max_raw.isdigit() else None

        try:
            n = fetch_chembl_actives(
                uniprot,
                output_path,
                ic50_um=ic50,
                max_compounds=max_compounds,
            )
            if n:
                msg = f"[#3fb950]Done.[/#3fb950]  {n:,} actives saved to {output_path}"
                self.app.call_from_thread(self._finish, msg, output_path)
                return
            else:
                msg = f"[#e3b341]No actives found for {uniprot} at IC50 <= {ic50} µM.[/#e3b341]"
        except Exception as exc:
            msg = f"[#f85149]Error: {exc}[/#f85149]"

        self.app.call_from_thread(self._finish, msg)

    def _status(self, msg: str) -> None:
        self.query_one("#lib-status", Static).update(msg)

    def _finish(self, msg: str, success_path: Path | None = None) -> None:
        self._status(msg)
        self.query_one("#btn-download").disabled = False
        use_btn = self.query_one("#btn-use", Button)
        if success_path is not None:
            use_btn.display = True
            use_btn._success_path = success_path  # type: ignore[attr-defined]
        else:
            use_btn.display = False
