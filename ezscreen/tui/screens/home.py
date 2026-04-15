from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static

from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_STATUS_STYLE: dict[str, str] = {
    "complete": "bold #3fb950",
    "running":  "bold #79c0ff",
    "failed":   "bold #f85149",
    "pending":  "#e3b341",
}


class HomeScreen(Screen):
    """Dashboard — recent runs table, quick actions, auth status."""

    BINDINGS = [
        Binding("n", "new_run", "New Run"),
        Binding("s", "open_status", "Status"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home"])
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Quick Actions", classes="section-title")
                yield ListView(
                    ListItem(Label("  New Run"),         id="action-run"),
                    ListItem(Label("  Status Monitor"),  id="action-status"),
                    ListItem(Label("  ADMET Filter"),    id="action-admet"),
                    ListItem(Label("  Validate"),        id="action-validate"),
                    ListItem(Label("  Auth Setup"),      id="action-auth"),
                    ListItem(Label("  Settings"),        id="action-settings"),
                    id="quick-actions",
                )
            with Vertical(id="runs-panel"):
                yield Label("Recent Runs", classes="section-title")
                yield DataTable(id="runs-table", cursor_type="row")
            with Vertical(id="status-panel"):
                yield Label("System Status", classes="section-title")
                yield Static("", id="system-status")
        yield Footer()

    def on_mount(self) -> None:
        self._populate_runs()
        self._populate_status()

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _populate_runs(self) -> None:
        from ezscreen import checkpoint
        try:
            checkpoint.init_db()
            runs = checkpoint.list_runs()
        except Exception:
            runs = []

        table = self.query_one("#runs-table", DataTable)
        table.add_columns("Run ID", "Status", "Compounds", "Created")

        if not runs:
            table.add_row("—", "—", "—", "no runs yet")
            return

        for run in runs[:30]:
            style = _STATUS_STYLE.get(run["status"], "white")
            table.add_row(
                run["run_id"],
                Text(run["status"], style=style),
                f"{run['total_compounds']:,}",
                run["created_at"][:10],
                key=run["run_id"],
            )

    def _populate_status(self) -> None:
        from ezscreen import auth as _auth
        try:
            creds      = _auth.load_credentials()
            kaggle_ok  = (p := _auth.get_kaggle_json_path(creds)) is not None and p.exists()
            nim_ok     = bool(_auth.get_nim_key(creds))
        except Exception:
            kaggle_ok = nim_ok = False

        def _fmt(ok: bool, optional: bool = False) -> str:
            if ok:
                return "[bold #3fb950]configured[/bold #3fb950]"
            return "[#6e7681]not set (optional)[/#6e7681]" if optional else "[bold #f85149]not set[/bold #f85149]"

        hint = "\n\n[#6e7681]Run  ezscreen auth  to configure.[/#6e7681]" if not kaggle_ok else ""

        self.query_one("#system-status", Static).update(
            f"Kaggle:  {_fmt(kaggle_ok)}\n"
            f"NIM key: {_fmt(nim_ok, optional=True)}"
            f"{hint}"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        actions = {
            "action-run":      self.action_new_run,
            "action-status":   self.action_open_status,
            "action-admet":    self.action_open_admet,
            "action-validate": self.action_open_validate,
            "action-auth":     self.action_open_auth,
            "action-settings": self.action_open_settings,
        }
        fn = actions.get(event.item.id)
        if fn:
            fn()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        run_id = str(event.row_key.value) if event.row_key else None
        if run_id and run_id != "—":
            self.app.nav.selected_run_id = run_id
            from ezscreen.tui.screens.results_viewer import ResultsScreen
            self.app.push_screen(ResultsScreen(run_id))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_new_run(self) -> None:
        self._placeholder("New Run")  # Phase 10.4

    def action_open_status(self) -> None:
        from ezscreen.tui.screens.status_monitor import StatusScreen
        self.app.push_screen(StatusScreen())

    def action_open_admet(self) -> None:
        from ezscreen.tui.screens.admet_filter import AdmetScreen
        self.app.push_screen(AdmetScreen())

    def action_open_validate(self) -> None:
        from ezscreen.tui.screens.validate_screen import ValidateScreen
        self.app.push_screen(ValidateScreen())

    def action_open_auth(self) -> None:
        from ezscreen.tui.screens.auth_setup import AuthScreen
        self.app.push_screen(AuthScreen())

    def action_open_settings(self) -> None:
        from ezscreen.tui.screens.settings import SettingsScreen
        self.app.push_screen(SettingsScreen())

    def _placeholder(self, title: str) -> None:
        from ezscreen.tui.screens._placeholder import PlaceholderScreen
        self.app.push_screen(PlaceholderScreen(title))
