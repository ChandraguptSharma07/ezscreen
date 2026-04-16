from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label

from ezscreen.tui.widgets.breadcrumb import Breadcrumb
from ezscreen.tui.widgets.run_card import _STATUS_STYLE, RunCard


class StatusScreen(Screen):
    """All runs with 30-second auto-refresh and per-run detail panel."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._runs: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Status Monitor"])
        with Horizontal(id="status-main"):
            with Vertical(id="runs-list"):
                yield Label("All Runs", classes="section-title")
                yield DataTable(id="status-table", cursor_type="row")
            with Vertical(id="run-detail"):
                yield Label("Run Detail", classes="section-title")
                yield RunCard(id="run-card")
                with Vertical(id="detail-actions"):
                    yield Button("View Results", id="btn-view",     variant="default")
                    yield Button("Download",     id="btn-download", variant="default")
                    yield Button("Clean",        id="btn-clean",    variant="default")
                    yield Button("Resume Failed Shards", id="btn-resume", variant="warning")
        yield Footer()

    def on_mount(self) -> None:
        self._populate_table()
        self._reset_detail()
        self.set_interval(30, self._populate_table)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        from ezscreen import checkpoint
        try:
            checkpoint.init_db()
            runs = checkpoint.list_runs()
            self._runs = {r["run_id"]: r for r in runs}
        except Exception:
            self._runs = {}

        table = self.query_one("#status-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Run ID", "Status", "Done", "Compounds", "Elapsed", "Created")

        if not self._runs:
            table.add_row("—", "—", "—", "—", "—", "no runs yet")
            return

        from ezscreen.tui.widgets.run_card import _elapsed
        for run in self._runs.values():
            style  = _STATUS_STYLE.get(run["status"], "white")
            total  = run["total_compounds"]
            done   = run["completed_compounds"]
            pct    = f" ({100 * done // total}%)" if total else ""
            table.add_row(
                run["run_id"],
                Text(run["status"], style=style),
                f"{done:,}{pct}",
                f"{total:,}",
                _elapsed(run["created_at"]),
                run["created_at"][:10],
                key=run["run_id"],
            )

    def _reset_detail(self) -> None:
        self.query_one("#run-card", RunCard).update(
            "[#6e7681]Select a run to see details.[/#6e7681]"
        )
        self.query_one("#detail-actions").display = False

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        run_id = str(event.row_key.value) if event.row_key else None
        if not run_id or run_id == "—":
            return
        run = self._runs.get(run_id)
        if not run:
            return
        self.query_one("#run-card", RunCard).show(run)
        self.query_one("#detail-actions").display = True
        self.app.nav.selected_run_id = run_id

        from ezscreen import checkpoint
        try:
            has_failed = bool(checkpoint.get_failed_shards(run_id))
        except Exception:
            has_failed = False
        self.query_one("#btn-resume").display = has_failed

    def on_button_pressed(self, event: Button.Pressed) -> None:
        run_id = self.app.nav.selected_run_id
        if not run_id:
            return
        if event.button.id == "btn-view":
            from ezscreen.tui.screens.results_viewer import ResultsScreen
            self.app.push_screen(ResultsScreen(run_id))
        elif event.button.id == "btn-download":
            self.app.notify(f"Run:  ezscreen download {run_id}", timeout=6)
        elif event.button.id == "btn-clean":
            self.app.notify(f"Run:  ezscreen clean {run_id}", timeout=6)
        elif event.button.id == "btn-resume":
            self._run_resume(run_id)

    def _run_resume(self, run_id: str) -> None:
        work_dir = Path.home() / ".ezscreen" / "runs" / run_id
        self.query_one("#btn-resume").disabled = True
        self.app.notify(f"Resuming failed shards for {run_id}...", timeout=4)

        def _worker() -> None:
            from ezscreen.backends.kaggle.runner import resume_failed_shards
            try:
                result = resume_failed_shards(run_id, work_dir)
                self.app.call_from_thread(self._on_resume_done, result)
            except Exception as exc:
                self.app.call_from_thread(
                    self.app.notify,
                    f"Resume failed: {exc}",
                    severity="error",
                    timeout=8,
                )
                self.app.call_from_thread(
                    setattr, self.query_one("#btn-resume"), "disabled", False
                )

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _on_resume_done(self, result: dict) -> None:
        self.query_one("#btn-resume").disabled = False
        status = result.get("status", "unknown")
        n      = result.get("n_shards", 0)
        ok     = result.get("n_succeeded", 0)
        if status == "nothing_to_resume":
            self.app.notify("No failed shards to resume.", timeout=4)
        elif status == "failed" and "error" in result:
            self.app.notify(result["error"], severity="error", timeout=8)
        else:
            msg = f"Resume done — {ok}/{n} shard(s) succeeded."
            self.app.notify(msg, severity="information" if ok == n else "warning", timeout=6)
        self._populate_table()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_refresh(self) -> None:
        self._populate_table()
        self.app.notify("Refreshed.", timeout=2)
