from __future__ import annotations

from textual.widgets import Static


class Breadcrumb(Static):
    """Renders a navigation breadcrumb trail at the top of each screen.

    Usage::
        yield Breadcrumb(["Home", "Status Monitor"])
    """

    def __init__(self, trail: list[str]) -> None:
        super().__init__("  >  ".join(trail), classes="breadcrumb")
