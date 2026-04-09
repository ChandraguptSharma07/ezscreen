from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ezscreen import version_check

app = typer.Typer(
    name="ezscreen",
    help="GPU-accelerated virtual screening — powered by Kaggle T4 GPUs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Run before every command to start the async version check."""
    version_check.start()
    # The banner will be printed by the atexit handler or when commands finish,
    # but we can also use rich console directly. An atexit handler or just 
    # relying on the end of the script works well. Let's register an atexit.
    import atexit
    
    @atexit.register
    def _print_version_banner() -> None:
        banner = version_check.banner()
        if banner:
            console.print(f"\n{banner}")


@app.command()
def run() -> None:
    """Run an interactive virtual screening job."""
    from ezscreen.commands import run as _run
    _run.invoke()


@app.command()
def auth(
    update: str = typer.Option(None, "--update", "-u", help="Update a specific key: 'Kaggle credentials' | 'NIM API key' | 'Both'"),
) -> None:
    """Set up or update Kaggle and NIM credentials."""
    from ezscreen.commands import auth as _auth
    _auth.invoke(update=update)


@app.command()
def validate(
    receptor: Path = typer.Argument(..., help="Path to receptor PDB file"),
    hits:     Path = typer.Argument(..., help="Path to hits SDF or CSV file"),
    output:   Path = typer.Option(Path("validation_out"), "--output", "-o", help="Output directory"),
) -> None:
    """Run Stage 2 hit validation with DiffDock-L via NVIDIA NIM."""
    from ezscreen.commands import validate as _validate
    _validate.invoke(receptor_path=receptor, hits_path=hits, output_dir=output)


@app.command()
def admet(
    input_file:  Path = typer.Argument(..., help="Input SDF file"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output SDF (default: <input>_admet.sdf)"),
) -> None:
    """Run standalone ADMET filtering on an SDF file."""
    from ezscreen.commands import admet as _admet
    _admet.invoke(input_path=input_file, output_path=output_file)


@app.command()
def view(
    results_dir: str = typer.Argument(..., help="Run ID (e.g. ezs-4f2a8c) or results directory path"),
    top:         int  = typer.Option(25, "--top", "-n", help="Number of top hits to show"),
) -> None:
    """Open the results viewer for a completed run."""
    import os
    from ezscreen.commands import view as _view
    p = Path(results_dir)
    # If it looks like a run ID and doesn't exist as a path, resolve via ~/.ezscreen/runs
    if not p.exists() and results_dir.startswith("ezs-"):
        p = Path.home() / ".ezscreen" / "runs" / results_dir / "output"
    _view.invoke(results_dir=p, top_n=top)


@app.command()
def status(
    live: bool = typer.Option(False, "--live", "-l", help="Auto-refresh every 30 s."),
) -> None:
    """Show all recent runs with live status."""
    from ezscreen.commands import status as _status
    _status.invoke(live=live)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume (e.g. ezs-4f2a8c)"),
) -> None:
    """Resume an interrupted screening run."""
    from ezscreen import checkpoint
    from rich.panel import Panel

    checkpoint.init_db()
    run = checkpoint.get_run(run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found in local database.[/red]")
        raise typer.Exit(1)

    if run["status"] == "failed":
        console.print(f"[red]Run {run_id} failed — start a fresh run with: ezscreen run[/red]")
        raise typer.Exit(1)

    incomplete = checkpoint.get_incomplete_shards(run_id)
    if not incomplete:
        console.print(f"[dim]Run {run_id} has no incomplete shards — already done.[/dim]")
        return

    console.print(Panel(
        f"[bold]{run_id}[/bold]  status: {run['status']}\n"
        f"{len(incomplete)} shard(s) still incomplete",
        title="[bold]Resume[/bold]",
    ))
    console.print("[dim]Re-submit functionality uses the same run.py flow with resume context.[/dim]")
    console.print("[yellow]Full resume is planned for v1.1 — run ezscreen run to start fresh.[/yellow]")


@app.command()
def download(
    run_id: str = typer.Argument(..., help="Run ID to download results for (e.g. ezs-4f2a8c)"),
) -> None:
    """Download results for a completed Kaggle run (use if download failed after a run)."""
    from ezscreen import auth as _auth
    from ezscreen.backends.kaggle import runner as kaggle_runner
    from ezscreen.commands import view as _view

    creds    = _auth.load_credentials()
    kaggle_path = _auth.get_kaggle_json_path(creds)
    import json
    username = json.loads(kaggle_path.read_text())["username"] if kaggle_path and kaggle_path.exists() else None
    if not username:
        console.print("[red]Kaggle credentials not found — run: ezscreen auth[/red]")
        raise typer.Exit(1)
    kernel_ref = f"{username}/{run_id}"
    work_dir   = Path.home() / ".ezscreen" / "runs" / run_id

    console.print(f"  [dim]Downloading results for {kernel_ref}...[/dim]")
    output_dir = kaggle_runner._download_output(kernel_ref, work_dir)
    console.print(f"  [green]✓ Results → {output_dir}[/green]")
    _view.invoke(results_dir=output_dir)


@app.command()
def clean(
    run_id: str = typer.Argument(..., help="Run ID to clean (e.g. ezs-4f2a8c)"),
) -> None:
    """Delete Kaggle dataset and kernel artifacts for a run."""
    from ezscreen import auth as _auth
    from ezscreen.backends.kaggle import runner as kaggle_runner
    import questionary

    creds    = _auth.load_credentials()
    username = Path(creds.get("kaggle_json_path", "")).stem or "user"

    confirmed = questionary.confirm(
        f"Delete all Kaggle artifacts for {run_id}?", default=False
    ).ask()
    if not confirmed:
        return

    kaggle_runner.clean_run(run_id, username)
    console.print(f"  [green]✓ Cleaned {run_id}[/green]")

