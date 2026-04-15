from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding

from ezscreen.tui.nav import NavState


class EzscreenApp(App):
    """Full-screen TUI for ezscreen."""

    CSS_PATH = [Path(__file__).parent / "theme.tcss"]
    TITLE = "ezscreen"
    SUB_TITLE = "GPU-accelerated virtual screening"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.nav = NavState()

    def on_mount(self) -> None:
        from ezscreen.tui.screens.home import HomeScreen
        self.push_screen(HomeScreen())

    def action_help(self) -> None:
        self.notify("Press q to quit  |  Escape to go back")
