from __future__ import annotations

import time
from datetime import datetime, timezone

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from ezscreen import checkpoint

console = Console()

_STATUS_STYLE: dict[str, str] = {
    "running":  "yellow",
    "complete": "green",
    "failed":   "red",
    "partial":  "yellow",
}

_REFRESH_SECS = 30


def _elapsed(created_at: str) -> str:
    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        d  = datetime.now(timezone.utc) - ts
        h, rem = divmod(int(d.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "-"


def _make_table(runs: list[dict]) -> Table:
    t = Table(box=box.ROUNDED, show_lines=False, expand=False, title="Recent runs")
    t.add_column("Run ID",    style="bold cyan",  no_wrap=True)
    t.add_column("Status",    no_wrap=True)
    t.add_column("Created",   style="dim",        no_wrap=True)
    t.add_column("Compounds", justify="right")
    t.add_column("Done",      justify="right")
    t.add_column("Elapsed",   justify="right",    style="dim")

    for r in runs:
        s      = r["status"]
        colour = _STATUS_STYLE.get(s, "")
        total  = r["total_compounds"]
        done   = r["completed_compounds"]
        pct    = f" ({100*done//total}%)" if total else ""
        t.add_row(
            r["run_id"],
            f"[{colour}]{s}[/{colour}]" if colour else s,
            r["created_at"][:19].replace("T", " "),
            str(total),
            f"{done}{pct}",
            _elapsed(r["created_at"]),
        )
    return t


def invoke(live: bool = False) -> None:
    """
    Show all recent runs.
    Pass live=True (via --live flag) for 30-second auto-refresh.
    """
    checkpoint.init_db()
    runs = checkpoint.list_runs()

    if not runs:
        console.print("[dim]No runs yet — start one with [bold]ezscreen run[/bold][/dim]")
        return

    if not live:
        console.print(_make_table(runs))
        console.print(
            f"[dim]  {len(runs)} run(s)  ·  "
            "resume: [bold]ezscreen resume <id>[/bold]  ·  "
            "clean:  [bold]ezscreen clean <id>[/bold]  ·  "
            "live:   [bold]ezscreen status --live[/bold][/dim]"
        )
        return

    # Live auto-refresh every 30 s — Ctrl-C to exit
    console.print("[dim]Live mode — refreshes every 30 s  ·  Ctrl-C to exit[/dim]")
    last_refresh = 0.0
    with Live(console=console, refresh_per_second=1, screen=False) as live_obj:
        try:
            while True:
                now = time.monotonic()
                if now - last_refresh >= _REFRESH_SECS:
                    runs         = checkpoint.list_runs()
                    last_refresh = now
                live_obj.update(_make_table(runs))
                time.sleep(1)
        except KeyboardInterrupt:
            pass
