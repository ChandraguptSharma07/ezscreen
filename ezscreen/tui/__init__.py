from __future__ import annotations


def launch_tui() -> None:
    """Launch the full-screen TUI application."""
    from ezscreen.tui.app import EzscreenApp
    EzscreenApp().run()
