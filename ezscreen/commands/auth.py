from __future__ import annotations

from ezscreen import auth as _auth


def invoke(update: str | None = None) -> None:
    """Wire up the `ezscreen auth` subcommand."""
    _auth.run_wizard(update=update)
