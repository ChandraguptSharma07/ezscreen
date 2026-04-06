from __future__ import annotations

from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel

from ezscreen.admet.filter import FilterConfig, V1_DISCLAIMER, filter_library

console = Console()


def invoke(input_path: Path, output_path: Path | None = None) -> None:
    """Standalone ADMET filter — works on any SDF file."""
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        return

    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_admet")

    # Show disclaimer
    console.print(Panel(f"[dim]{V1_DISCLAIMER}[/dim]", title="[bold]ADMET v1[/bold]"))

    # Let user toggle filters
    active = questionary.checkbox(
        "Which filters to apply?",
        choices=[
            questionary.Choice("Lipinski Rule of Five",   value="lipinski",     checked=True),
            questionary.Choice("PAINS alerts",             value="pains",        checked=True),
            questionary.Choice("Brenk toxicophores",       value="toxicophores", checked=True),
            questionary.Choice("Veber oral bioavailability", value="veber",      checked=True),
            questionary.Choice("Egan BBB permeability",    value="egan_bbb",     checked=False),
        ],
    ).ask()

    if active is None:
        return

    cfg = FilterConfig(
        lipinski=    "lipinski"     in active,
        pains=       "pains"        in active,
        toxicophores="toxicophores" in active,
        veber=       "veber"        in active,
        egan_bbb=    "egan_bbb"     in active,
    )

    console.print(f"\n  Filtering [bold]{input_path.name}[/bold]...")
    result = filter_library(str(input_path), str(output_path), cfg)

    passed  = result["total_input"] - result["admet_removed"]
    removed = result["admet_removed"]

    console.print(f"  [green]Passed :[/green]  {passed:,}")
    console.print(f"  [red]Removed:[/red] {removed:,}")

    if removed:
        for rule, count in result["admet_breakdown"].items():
            if count:
                console.print(f"    [dim]{rule.replace('_', ' ')}: {count:,}[/dim]")

    console.print(f"\n  Output → [bold]{output_path}[/bold]")
