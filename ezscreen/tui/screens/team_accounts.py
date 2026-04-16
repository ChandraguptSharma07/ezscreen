from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)

from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class TeamAccountsScreen(Screen):
    """Manage collaborator Kaggle accounts for round-robin shard distribution."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "Team Accounts"])
        with Horizontal(id="team-main"):
            with Vertical(id="accounts-panel"):
                yield Label("Collaborator Accounts", classes="section-title")
                yield DataTable(id="accounts-table", cursor_type="row")
                yield Button("Remove Selected", id="btn-remove", variant="error")
            with Vertical(id="add-panel"):
                yield Label("Add Account", classes="section-title")
                yield Label("Display name", classes="form-label")
                yield Input(id="inp-name", placeholder="alice")
                yield Label("Email", classes="form-label")
                yield Input(id="inp-email", placeholder="alice@example.com")
                yield Label("kaggle.json path", classes="form-label")
                yield Input(id="inp-path", placeholder="~/.kaggle/alice_kaggle.json")
                yield Checkbox(
                    "I confirm this collaborator has agreed to share Kaggle compute quota",
                    id="chk-consent",
                )
                yield Button("Add Account", id="btn-add", variant="primary")
                yield Static("", id="add-status")
        yield Footer()

    def on_mount(self) -> None:
        self._load_accounts()
        self.query_one("#btn-remove").display = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_accounts(self) -> None:
        from ezscreen import auth
        table = self.query_one("#accounts-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Name", "Email", "Username", "Path")
        accounts = auth.list_team_accounts()
        if not accounts:
            table.add_row("—", "—", "—", "no team accounts yet")
            return
        for acct in accounts:
            table.add_row(
                acct.get("name", ""),
                acct.get("email", ""),
                acct.get("username", "—"),
                acct.get("kaggle_json_path", ""),
                key=acct["name"],
            )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        selected = str(event.row_key.value) if event.row_key else None
        self.query_one("#btn-remove").display = bool(selected and selected != "—")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-add":
            self._add_account()
        elif event.button.id == "btn-remove":
            self._remove_selected()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _add_account(self) -> None:
        from ezscreen import auth

        name  = self.query_one("#inp-name",  Input).value.strip()
        email = self.query_one("#inp-email", Input).value.strip()
        path  = self.query_one("#inp-path",  Input).value.strip()
        consent = self.query_one("#chk-consent", Checkbox).value

        status = self.query_one("#add-status", Static)

        if not name:
            status.update("[#f85149]Display name is required.[/#f85149]")
            return
        if not path:
            status.update("[#f85149]kaggle.json path is required.[/#f85149]")
            return
        if not consent:
            status.update("[#f85149]Consent checkbox must be checked.[/#f85149]")
            return

        kj_path = Path(path).expanduser()
        if not kj_path.exists():
            status.update(f"[#f85149]File not found: {kj_path}[/#f85149]")
            return

        try:
            auth.add_team_account(name, email, kj_path)
            self._load_accounts()
            self.query_one("#inp-name",  Input).value = ""
            self.query_one("#inp-email", Input).value = ""
            self.query_one("#inp-path",  Input).value = ""
            self.query_one("#chk-consent", Checkbox).value = False
            status.update(f"[#3fb950]Account '{name}' added.[/#3fb950]")
        except Exception as exc:
            status.update(f"[#f85149]{exc}[/#f85149]")

    def _remove_selected(self) -> None:
        from ezscreen import auth

        table = self.query_one("#accounts-table", DataTable)
        row   = table.cursor_row
        try:
            key = str(table.get_row_at(row)[0])
            if key == "—":
                return
            auth.remove_team_account(key)
            self._load_accounts()
            self.query_one("#btn-remove").display = False
            self.app.notify(f"Account '{key}' removed.", timeout=3)
        except Exception as exc:
            self.app.notify(str(exc), severity="error", timeout=5)
