from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label

from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class PlaceholderScreen(Screen):
    """Stub screen shown for features not yet implemented in the TUI."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", self._title])
        yield Label(f"{self._title} — coming soon", classes="placeholder-msg")
        yield Footer()
