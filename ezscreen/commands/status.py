from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ezscreen import checkpoint

console = Console()

_STATUS_STYLE: dict[str, str] = {
    "running":  "yellow",
    "complete": "green",
    "failed":   "red",
    "partial":  "yellow",
}


def invoke() -> None:
    """Display all recent runs with live status pulled from the local DB."""
    checkpoint.init_db()
    runs = checkpoint.list_runs()

    if not runs:
        console.print("[dim]No runs yet. Start one with [bold]ezscreen run[/bold][/dim]")
        return

    t = Table(title="Recent runs", show_lines=False, expand=False)
    t.add_column("Run ID",    style="bold cyan",  no_wrap=True)
    t.add_column("Status",    no_wrap=True)
    t.add_column("Created",   style="dim",        no_wrap=True)
    t.add_column("Compounds", justify="right")
    t.add_column("Done",      justify="right")

    for r in runs:
        s = r["status"]
        colour = _STATUS_STYLE.get(s, "")
        t.add_row(
            r["run_id"],
            f"[{colour}]{s}[/{colour}]" if colour else s,
            r["created_at"][:19].replace("T", " "),
            str(r["total_compounds"]),
            str(r["completed_compounds"]),
        )

    console.print(t)
    console.print(
        f"[dim]  {len(runs)} run(s) total  ·  "
        "resume with [bold]ezscreen resume <run-id>[/bold]  ·  "
        "clean with [bold]ezscreen clean <run-id>[/bold][/dim]"
    )
