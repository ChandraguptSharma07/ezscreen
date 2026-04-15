from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.worker import WorkerState

from ezscreen.tui.nav import NavState


class EzscreenApp(App):
    """Full-screen TUI for ezscreen."""

    CSS_PATH  = [Path(__file__).parent / "theme.tcss"]
    TITLE     = "ezscreen"
    SUB_TITLE = "GPU-accelerated virtual screening"

    BINDINGS = [
        Binding("q",             "quit",        "Quit",    priority=True),
        Binding("question_mark", "show_help",   "Help"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.nav = NavState()

    def on_mount(self) -> None:
        from ezscreen.tui.screens.home import HomeScreen
        self.push_screen(HomeScreen())

    # ------------------------------------------------------------------
    # Help overlay
    # ------------------------------------------------------------------

    def action_show_help(self) -> None:
        from ezscreen.tui.screens.help_overlay import HelpScreen
        self.push_screen(HelpScreen())

    # ------------------------------------------------------------------
    # Graceful quit — warn if a submission worker is active
    # ------------------------------------------------------------------

    def action_quit(self) -> None:
        if self._has_active_submission():
            self.notify(
                "A Kaggle job is running in the background and will continue "
                "after the TUI closes.  Use  ezscreen resume <run_id>  to reconnect.",
                title="Job in progress",
                severity="warning",
                timeout=5,
            )
            self.call_later(self._do_quit)
        else:
            self._do_quit()

    def _do_quit(self) -> None:
        self.exit()

    def _has_active_submission(self) -> bool:
        """Return True if any RunWizardScreen is currently submitting."""
        try:
            from ezscreen.tui.screens.run_wizard import RunWizardScreen
            for screen in self.screen_stack:
                if isinstance(screen, RunWizardScreen) and screen._submitted:
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Error boundary — catch unhandled worker failures
    # ------------------------------------------------------------------

    def on_worker_state_changed(self, event) -> None:
        if event.state is WorkerState.ERROR:
            exc = event.worker.error
            msg = str(exc) if exc else "Unknown error"
            # Truncate very long messages
            if len(msg) > 120:
                msg = msg[:117] + "..."
            self.notify(
                msg,
                title="Unexpected error",
                severity="error",
                timeout=8,
            )
