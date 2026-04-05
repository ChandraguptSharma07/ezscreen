from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="ezscreen",
    help="GPU-accelerated virtual screening — powered by Kaggle T4 GPUs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def run() -> None:
    """Run an interactive virtual screening job."""
    raise NotImplementedError


@app.command()
def auth() -> None:
    """Set up or update Kaggle and NIM credentials."""
    raise NotImplementedError


@app.command()
def validate() -> None:
    """Run Stage 2 hit validation with DiffDock-L via NVIDIA NIM."""
    raise NotImplementedError


@app.command()
def admet() -> None:
    """Run standalone ADMET filtering on a CSV or SDF file."""
    raise NotImplementedError


@app.command()
def view() -> None:
    """Open the results viewer for a completed run."""
    raise NotImplementedError


@app.command()
def status() -> None:
    """Show all recent Kaggle jobs with live status."""
    raise NotImplementedError


@app.command()
def resume(run_id: str = typer.Argument(..., help="Run ID to resume (e.g. ezs-4f2a8c).")) -> None:
    """Resume an interrupted screening run."""
    raise NotImplementedError


@app.command()
def clean(run_id: str = typer.Argument(..., help="Run ID to clean up (e.g. ezs-4f2a8c).")) -> None:
    """Delete Kaggle dataset and kernel artifacts for a run."""
    raise NotImplementedError
