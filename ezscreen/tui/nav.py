from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NavState:
    """Shared navigation state passed between TUI screens via app.nav."""

    breadcrumb: list[str] = field(default_factory=lambda: ["Home"])
    selected_run_id: str | None = None

    def push(self, label: str) -> None:
        self.breadcrumb.append(label)

    def pop(self) -> None:
        if len(self.breadcrumb) > 1:
            self.breadcrumb.pop()

    def text(self) -> str:
        return "  >  ".join(self.breadcrumb)
