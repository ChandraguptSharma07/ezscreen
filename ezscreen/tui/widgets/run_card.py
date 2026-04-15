from __future__ import annotations

from datetime import datetime, timezone

from rich.text import Text
from textual.widgets import Static

_STATUS_STYLE: dict[str, str] = {
    "complete": "bold #3fb950",
    "running":  "bold #79c0ff",
    "failed":   "bold #f85149",
    "partial":  "#e3b341",
    "pending":  "#e3b341",
}


def _elapsed(created_at: str) -> str:
    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        d  = datetime.now(timezone.utc) - ts
        h, rem = divmod(int(d.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "—"


class RunCard(Static):
    """Compact run summary for the Status Monitor detail panel."""

    def show(self, run: dict) -> None:
        style  = _STATUS_STYLE.get(run["status"], "white")
        total  = run["total_compounds"]
        done   = run["completed_compounds"]
        pct    = f"  ({100 * done // total}%)" if total else ""

        t = Text()
        t.append(f"{run['run_id']}\n\n", style="bold #79c0ff")

        def row(label: str, value: str, val_style: str = "#f0f6fc") -> None:
            t.append(f"{label:<12}", style="#6e7681")
            t.append(f"{value}\n", style=val_style)

        row("Status",     run["status"],          style)
        row("Created",    run["created_at"][:10])
        row("Elapsed",    _elapsed(run["created_at"]), "#6e7681")
        row("Compounds",  f"{total:,}")
        row("Done",       f"{done:,}{pct}")

        self.update(t)
