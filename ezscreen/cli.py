from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from ezscreen import version_check

app = typer.Typer(
    name="ezscreen",
    help="GPU-accelerated virtual screening — powered by Kaggle GPUs.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

console = Console()
team_app = typer.Typer(help="Manage team runners and worker control-plane state.")
benchmark_app = typer.Typer(help="Bootstrap and manage benchmark workspaces.")
app.add_typer(team_app, name="team")
app.add_typer(benchmark_app, name="benchmark")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Start version check, then launch the TUI if no subcommand was given."""
    version_check.start()
    import atexit

    @atexit.register
    def _print_version_banner() -> None:
        banner = version_check.banner()
        if banner:
            console.print(f"\n{banner}")

    if ctx.invoked_subcommand is None:
        from ezscreen.tui import launch_tui

        launch_tui()


@app.command()
def auth(
    update: str = typer.Option(
        None,
        "--update",
        "-u",
        help="Update a specific key: 'Kaggle credentials' | 'NIM API key' | 'Both'",
    ),
) -> None:
    """Set up or update Kaggle and NIM credentials."""
    from ezscreen.commands import auth as _auth

    _auth.invoke(update=update)


@app.command()
def validate(
    receptor: Path = typer.Argument(..., help="Path to receptor PDB file"),
    hits: Path = typer.Argument(..., help="Path to hits SDF or CSV file"),
    output: Path = typer.Option(
        Path("validation_out"), "--output", "-o", help="Output directory"
    ),
) -> None:
    """Run Stage 2 hit validation with DiffDock-L via NVIDIA NIM."""
    from ezscreen.commands import validate as _validate

    _validate.invoke(receptor_path=receptor, hits_path=hits, output_dir=output)


@app.command()
def admet(
    input_file: Path = typer.Argument(..., help="Input SDF file"),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output SDF (default: <input>_admet.sdf)"
    ),
) -> None:
    """Run standalone ADMET filtering on an SDF or SMILES file."""
    from ezscreen.commands import admet as _admet

    _admet.invoke(input_path=input_file, output_path=output_file)


@app.command()
def view(
    results_dir: str = typer.Argument(
        ..., help="Run ID (e.g. ezs-4f2a8c) or results directory path"
    ),
    top: int = typer.Option(25, "--top", "-n", help="Number of top hits to show"),
) -> None:
    """Open the results viewer for a completed run."""
    from ezscreen.commands import view as _view

    p = Path(results_dir)
    # Resolve run ID shorthand to full path under ~/.ezscreen/runs
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
    from rich.panel import Panel

    from ezscreen import checkpoint

    checkpoint.init_db()
    run = checkpoint.get_run(run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found in local database.[/red]")
        raise typer.Exit(1)

    if run["status"] == "failed":
        console.print(
            f"[red]Run {run_id} failed — start a fresh run with: ezscreen[/red]"
        )
        raise typer.Exit(1)

    incomplete = checkpoint.get_incomplete_shards(run_id)
    if not incomplete:
        console.print(
            f"[dim]Run {run_id} has no incomplete shards — already done.[/dim]"
        )
        return

    console.print(
        Panel(
            f"[bold]{run_id}[/bold]  status: {run['status']}\n"
            f"{len(incomplete)} shard(s) still incomplete",
            title="[bold]Resume[/bold]",
        )
    )
    console.print("[dim]Re-submit uses the same run.py flow with resume context.[/dim]")
    console.print(
        "[yellow]Full resume is planned for v1.1 — launch ezscreen to start fresh.[/yellow]"  # noqa: E501
    )


@app.command()
def download(
    run_id: str = typer.Argument(
        ..., help="Run ID to download results for (e.g. ezs-4f2a8c)"
    ),  # noqa: E501
) -> None:
    """Download results for a completed Kaggle run (if download failed after a run)."""
    from ezscreen import auth as _auth
    from ezscreen.backends.kaggle import runner as kaggle_runner
    from ezscreen.commands import view as _view

    creds = _auth.load_credentials()
    kaggle_path = _auth.get_kaggle_json_path(creds)
    import json

    username = (
        json.loads(kaggle_path.read_text())["username"]
        if kaggle_path and kaggle_path.exists()
        else None
    )
    if not username:
        console.print("[red]Kaggle credentials not found — run: ezscreen auth[/red]")
        raise typer.Exit(1)
    kernel_ref = f"{username}/{run_id}"
    work_dir = Path.home() / ".ezscreen" / "runs" / run_id

    console.print(f"  [dim]Downloading results for {kernel_ref}...[/dim]")
    output_dir = kaggle_runner._download_output(kernel_ref, work_dir)
    console.print(f"  [green]✓ Results → {output_dir}[/green]")
    _view.invoke(results_dir=output_dir)


@app.command()
def clean(
    run_id: str = typer.Argument(..., help="Run ID to clean (e.g. ezs-4f2a8c)"),
) -> None:
    """Delete Kaggle dataset and kernel artifacts for a run."""
    import questionary

    from ezscreen import auth as _auth
    from ezscreen.backends.kaggle import runner as kaggle_runner

    creds = _auth.load_credentials()
    username = Path(creds.get("kaggle_json_path", "")).stem or "user"

    confirmed = questionary.confirm(
        f"Delete all Kaggle artifacts for {run_id}?", default=False
    ).ask()
    if not confirmed:
        return

    kaggle_runner.clean_run(run_id, username)
    console.print(f"  [green]✓ Cleaned {run_id}[/green]")


@benchmark_app.command("init")
def benchmark_init(
    benchmark_id: str = typer.Argument(
        ..., help="Benchmark campaign id, for example bench-20260714-core-panel"
    ),
    target: str = typer.Option(
        None, "--target", help="Target id for a benchmark cell scaffold"
    ),
    regime: str = typer.Option(
        None, "--regime", help="Condition or regime id for a benchmark cell scaffold"
    ),
    run_label: str = typer.Option(
        None, "--run-label", help="Run label for a benchmark cell scaffold"
    ),
    root: Path = typer.Option(
        None, "--root", help="Optional benchmark root directory override"
    ),
) -> None:
    """Create a benchmark campaign or benchmark cell scaffold from repo templates."""
    from ezscreen.commands import benchmark as _benchmark

    try:
        path = _benchmark.init_scaffold(
            benchmark_id,
            target=target,
            regime=regime,
            run_label=run_label,
            root=root,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if target and regime and run_label:
        console.print(f"[green]Benchmark cell scaffold created[/green] {path}")
    else:
        console.print(f"[green]Benchmark campaign scaffold created[/green] {path}")


@benchmark_app.command("run-cell")
def benchmark_run_cell(
    cell_dir: Path = typer.Argument(..., help="Path to a benchmark cell directory"),
    actives: Path = typer.Option(
        None, "--actives", help="Optional override for the known-actives .smi file"
    ),
    scores: Path = typer.Option(
        None, "--scores", help="Optional override for the archived scores.csv file"
    ),
    score_col: str = typer.Option(
        None, "--score-col", help="Score column to rank by; defaults to first score/affinity column"
    ),
) -> None:
    """Run benchmark metrics for one archived cell and write report + summary files."""
    from ezscreen.commands import benchmark as _benchmark

    try:
        result = _benchmark.run_cell_benchmark(
            cell_dir,
            actives_path=actives,
            scores_csv=scores,
            score_col=score_col,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Benchmark cell failed[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Benchmark cell completed[/green] {result['cell_dir']}")
    console.print(f"  metrics -> {result['metrics_path']}")
    console.print(f"  report  -> {result['report_path']}")



@benchmark_app.command("correct-scores")
def benchmark_correct_scores(
    scores_csv: Path = typer.Argument(..., help="Path to scores.csv to annotate"),
    output: Path = typer.Option(None, "--output", help="Output CSV; defaults to in-place"),
    score_col: str = typer.Option(None, "--score-col", help="Raw score column; defaults to first score/affinity column"),
    output_col: str = typer.Option("bias_corrected_score", "--output-col", help="Corrected score column name"),
) -> None:
    """Append ligand-bias-corrected score column for benchmark reranking."""
    from ezscreen.commands import benchmark as _benchmark

    try:
        result = _benchmark.correct_scores_file(
            scores_csv,
            output_csv=output,
            score_col=score_col,
            output_col=output_col,
        )
    except Exception as exc:
        console.print(f"[red]Score correction failed[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Corrected scores written[/green] {result['output_csv']}")
    console.print(f"  score_col -> {result['score_col']}")
    console.print(f"  output_col -> {result['output_col']}")
    console.print(f"  formula -> {result['formula']}")



@benchmark_app.command("compare-rescores")
def benchmark_compare_rescores(
    scores_csv: Path = typer.Argument(..., help="Path to scores.csv with raw and rescoring columns"),
    actives: Path = typer.Option(..., "--actives", help="Known-actives .smi/.ism file with identifiers"),
    output_dir: Path = typer.Option(None, "--output-dir", help="Output directory; defaults beside scores.csv"),
    score_cols: str = typer.Option(None, "--score-cols", help="Comma-separated score columns; defaults to numeric score/affinity columns"),
) -> None:
    """Compare EF/AUC and diagnostics for multiple score columns in one scores.csv."""
    from ezscreen.commands import benchmark as _benchmark

    cols = [part.strip() for part in score_cols.split(",") if part.strip()] if score_cols else None
    try:
        result = _benchmark.compare_rescore_file(
            actives_path=actives,
            scores_csv=scores_csv,
            output_dir=output_dir,
            score_cols=cols,
        )
    except Exception as exc:
        console.print(f"[red]Rescore comparison failed[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Rescore comparison written[/green] {result['output_dir']}")
    console.print(f"  metrics -> {result['metrics_table']}")
    console.print(f"  summary -> {result['summary_path']}")

@benchmark_app.command("submit-kaggle-cell")
def benchmark_submit_kaggle_cell(
    cell_dir: Path = typer.Argument(..., help="Path to a benchmark cell directory"),
    receptor_pdbqt: Path = typer.Option(..., "--receptor-pdbqt", help="Prepared receptor PDBQT uploaded to Kaggle"),
    ligands: Path = typer.Option(..., "--ligands", help="Benchmark ligand library (.smi/.ism/.sdf) to shard for Kaggle prep"),
    box_center: str = typer.Option(..., "--box-center", help="Docking box center as x,y,z"),
    box_size: str = typer.Option(..., "--box-size", help="Docking box size as x,y,z"),
    actives: Path = typer.Option(None, "--actives", help="Known-actives .smi file for metrics; defaults to cell inputs/actives.smi"),
    run_id: str = typer.Option(None, "--run-id", help="Optional ezscreen run id"),
    shards: int = typer.Option(None, "--shards", help="Optional number of raw ligand shards"),
    accelerator: str = typer.Option("nvidiaTeslaT4", "--accelerator", help="Kaggle accelerator metadata value"),
    engine: str = typer.Option("unidock", "--engine", help="Kaggle docking engine"),
    search_mode: str = typer.Option("balance", "--search-mode", help="Kaggle notebook search mode"),
    enumerate_enabled: bool = typer.Option(
        False,
        "--enumerate/--no-enumerate",
        help="Enumerate variants; disabled for the paper baseline",
    ),
    intended_compounds: Path = typer.Option(
        None,
        "--intended-compounds",
        help="Source-compound denominator manifest; required for non-SMILES inputs",
    ),
    campaign_dir: Path = typer.Option(
        None,
        "--campaign-dir",
        help="Campaign supervised by campaign_state.json",
    ),
    campaign_account: str = typer.Option(
        None,
        "--campaign-account",
        help="Non-secret Kaggle account label for quota accounting",
    ),
    campaign_stage: str = typer.Option(
        "baseline",
        "--campaign-stage",
        help="Quota allocation stage",
    ),
    planned_gpu_hours: float = typer.Option(
        None,
        "--planned-gpu-hours",
        help="Required positive quota reservation for supervised submissions",
    ),
) -> None:
    """Submit one benchmark cell to Kaggle, then archive outputs and compute metrics."""
    from ezscreen.commands import benchmark as _benchmark

    try:
        result = _benchmark.submit_kaggle_cell(
            cell_dir,
            receptor_pdbqt=receptor_pdbqt,
            ligands=ligands,
            box_center=box_center,
            box_size=box_size,
            actives_path=actives,
            run_id=run_id,
            n_shards=shards,
            accelerator=accelerator,
            engine=engine,
            search_mode=search_mode,
            enumerate_enabled=enumerate_enabled,
            intended_compounds_path=intended_compounds,
            campaign_dir=campaign_dir,
            campaign_account=campaign_account,
            campaign_stage=campaign_stage,
            planned_gpu_hours=planned_gpu_hours,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Kaggle benchmark submission failed[/red] {exc}")
        raise typer.Exit(1) from exc

    if result.get("status") == "complete":
        console.print(f"[green]Kaggle benchmark cell completed[/green] {result['cell_dir']}")
        console.print(f"  run_id -> {result['run_id']}")
        console.print(f"  scores -> {result['scores_csv']}")
        console.print(f"  report -> {result['report_path']}")
    else:
        console.print(f"[red]Kaggle benchmark cell {result.get('status', 'failed')}[/red]")
        console.print(f"  run_id -> {result.get('run_id', '')}")
        raise typer.Exit(1)

@benchmark_app.command("validate-campaign")
def benchmark_validate_campaign(
    campaign_dir: Path = typer.Argument(
        ..., help="Path to a benchmark campaign directory"
    ),
) -> None:
    """Validate one benchmark campaign manifest, target registry, and referenced input files."""
    from ezscreen.benchmark.summary import validate_campaign

    try:
        result = validate_campaign(campaign_dir)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]Benchmark campaign validated[/green] {result['campaign_dir']}"
    )
    console.print(f"  targets -> {result['target_count']}")
    console.print(f"  checked files -> {result['checked_file_count']}")
    if result.get("warnings"):
        console.print("  [yellow]warnings:[/yellow]")
        for item in result["warnings"]:
            console.print(f"    - {item}")


@benchmark_app.command("audit-campaign")
def benchmark_audit_campaign(
    campaign_dir: Path = typer.Argument(
        ..., help="Path to a benchmark campaign directory"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit nonzero when publishable-pipeline gates fail"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable audit JSON"
    ),
) -> None:
    """Audit whether a campaign is shaped like publishable pipeline evidence."""
    import json as json_lib

    from ezscreen.benchmark.summary import audit_campaign_quality

    try:
        result = audit_campaign_quality(campaign_dir)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if json_output:
        console.print(json_lib.dumps(result, indent=2))
    else:
        console.print(
            f"[green]Benchmark campaign audit[/green] {result['campaign_dir']}"
        )
        console.print(f"  readiness -> {result['readiness']}")
        console.print(
            f"  targets   -> {result['target_count']} across {result['target_class_count']} class(es)"
        )
        console.print(
            f"  cells     -> {result['complete_cell_count']}/{result['cell_count']} complete"
        )
        console.print(f"  repeats   -> max {result['max_repeat_count']}")
        console.print(f"  comparators -> max {result['max_comparator_count']}")
        if result["strengths"]:
            console.print("  strengths:")
            for item in result["strengths"]:
                console.print(f"    - {item}")
        if result["issues"]:
            console.print("  [red]publishability gaps:[/red]")
            for item in result["issues"]:
                console.print(f"    - {item}")
        if result["warnings"]:
            console.print("  [yellow]warnings:[/yellow]")
            for item in result["warnings"]:
                console.print(f"    - {item}")

    if strict and result["issues"]:
        raise typer.Exit(1)


@benchmark_app.command("summarize")
def benchmark_summarize(
    campaign_dir: Path = typer.Argument(
        ..., help="Path to a benchmark campaign directory"
    ),
) -> None:
    """Summarize benchmark cells into campaign-level CSV tables."""
    from ezscreen.benchmark.summary import summarize_campaign

    try:
        result = summarize_campaign(campaign_dir)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(
        f"[green]Benchmark campaign summarized[/green] {result['campaign_dir']}"
    )
    console.print(f"  cells      -> {result['cell_count']}")
    console.print(f"  metrics    -> {result['metrics_path']}")
    console.print(f"  failures   -> {result['failure_path']}")
    console.print(f"  throughput -> {result['throughput_path']}")
    console.print(f"  variance   -> {result['variance_path']}")


@benchmark_app.command("supervise")
def benchmark_supervise(
    campaign_dir: Path = typer.Argument(
        ..., help="Path to a benchmark campaign directory"
    ),
    once: bool = typer.Option(
        False, "--once", help="Poll once, persist state, then exit"
    ),
    poll_seconds: int = typer.Option(
        30, "--poll-seconds", min=1, help="Seconds between polling cycles"
    ),
) -> None:
    """Supervise Kaggle benchmark jobs with restart-safe quota and recovery state."""
    from ezscreen.benchmark.supervisor import supervise_campaign

    try:
        result = supervise_campaign(
            campaign_dir,
            once=once,
            poll_seconds=poll_seconds,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    state = result["state"]
    quota = result["quota"]
    console.print(
        f"[green]Benchmark supervisor[/green] status={state.get('status')}"
    )
    console.print(
        f"  planned={quota.get('planned_gpu_hours', 0)} GPU-h "
        f"observed={quota.get('observed_gpu_hours', 0)} GPU-h"
    )


@benchmark_app.command("submit-notebook")
def benchmark_submit_notebook(
    campaign_dir: Path = typer.Argument(..., help="Benchmark campaign directory"),
    notebook: Path = typer.Argument(..., help="Rendered notebook to submit"),
    inputs: list[Path] = typer.Option(
        ..., "--input", help="Immutable notebook input; repeat for multiple files"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Unique Kaggle kernel id"),
    job_id: str = typer.Option(None, "--job-id", help="Campaign job id"),
    stage: str = typer.Option("smoke", "--stage", help="Campaign quota stage"),
    backend: str = typer.Option(
        "cpu", "--backend", help="Compute backend: cpu, p100, or t4x2"
    ),
    account: str = typer.Option(
        None, "--account", help="Saved Kaggle account label; auto-selects if omitted"
    ),
    planned_gpu_hours: float = typer.Option(
        0.0, "--planned-gpu-hours", help="GPU reservation; must be zero for CPU"
    ),
    resume_blocked: bool = typer.Option(
        False,
        "--resume-blocked",
        help="Resume an explicitly authorized blocked campaign repair",
    ),
    dataset_ref: str = typer.Option(
        None,
        "--dataset-ref",
        help="Reuse an already-mounted immutable Kaggle owner/slug dataset",
    ),
    supersedes: list[str] = typer.Option(
        None,
        "--supersedes",
        help="Prior campaign job replaced by this retry; repeat as needed",
    ),
    finalizer: str = typer.Option(
        None,
        "--finalizer",
        help="Post-recovery finalizer: bayesbind-vina or bayesbind-gnina",
    ),
    intended_csv: Path = typer.Option(
        None,
        "--intended-csv",
        help="Frozen intended compounds CSV required by BayesBind finalizers",
    ),
) -> None:
    """Submit an auditable notebook using the same backend path as users."""
    from ezscreen.benchmark.kaggle_notebook import submit_notebook

    try:
        result = submit_notebook(
            campaign_dir,
            notebook,
            inputs,
            run_id=run_id,
            job_id=job_id,
            stage=stage,
            backend=backend,
            account_name=account,
            planned_gpu_hours=planned_gpu_hours,
            resume_blocked=resume_blocked,
            dataset_ref=dataset_ref,
            supersedes_job_ids=supersedes,
            finalizer=finalizer,
            intended_csv=intended_csv,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Benchmark notebook submitted[/green] "
        f"{result['kernel_ref']} backend={result['backend']}"
    )


@benchmark_app.command("recover-smina")
def benchmark_recover_smina(
    output_dir: Path = typer.Argument(
        ..., help="Directory containing smina logs/ and poses/"
    ),
    expected_attempts: int = typer.Option(
        None, "--expected-attempts", min=1, help="Strict expected docking count"
    ),
) -> None:
    """Recover auditable smina scores from normal log and pose outputs."""
    from ezscreen.benchmark.smina_output import recover_smina_directory

    try:
        result = recover_smina_directory(
            output_dir, expected_attempts=expected_attempts
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Recovered smina outputs[/green] "
        f"scored={result['scored']}/{result['expected_attempts']} "
        f"gate_pass={result['gate_pass']}"
    )


@benchmark_app.command("freeze-bayesbind")
def benchmark_freeze_bayesbind(
    campaign_dir: Path = typer.Argument(..., help="Benchmark campaign directory"),
    extracted_dir: Path = typer.Argument(
        ..., help="Extracted official BayesBindV1.5 directory or its parent"
    ),
    archive: Path = typer.Option(
        None, "--archive", help="Official BayesBindV1.5.tar.gz for hash verification"
    ),
    random_limit: int = typer.Option(
        1000, "--random-limit", min=1, help="Paper-used random rows per target"
    ),
    active_limit: int = typer.Option(
        1000, "--active-limit", min=1, help="Paper-used measured rows per target"
    ),
    active_cutoff: float = typer.Option(
        5.0, "--active-cutoff", help="Paper pChEMBL active cutoff"
    ),
) -> None:
    """Freeze paper rows and validity-corrected BayesBind source compounds."""
    from ezscreen.benchmark.bayesbind_paper import freeze_bayesbind

    try:
        result = freeze_bayesbind(
            campaign_dir,
            extracted_dir,
            archive_path=archive,
            random_limit=random_limit,
            active_limit=active_limit,
            active_cutoff=active_cutoff,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind protocol frozen[/green] "
        f"targets={result['target_count']} "
        f"paper_rows={result['paper_intended_rows']} "
        f"source_compounds={result['source_compounds']} "
        f"label_conflicts={result['cross_label_conflicts']}"
    )

@benchmark_app.command("extract-bayesbind-ligands")
def benchmark_extract_bayesbind_ligands(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    archive: Path = typer.Option(
        None, "--archive", help="Official BigBindV1.5.tar.gz path; defaults to campaign inputs"
    ),
    inventory: Path = typer.Option(
        None, "--inventory", help="BigBind member inventory JSONL; defaults to campaign archive_manifests"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Extraction/provenance output directory"
    ),
) -> None:
    """Resolve and selectively extract exact BayesBind ligand SDF bytes."""
    from ezscreen.benchmark.bayesbind_paper import extract_exact_bayesbind_ligands

    try:
        result = extract_exact_bayesbind_ligands(
            campaign_dir,
            archive_path=archive,
            inventory_path=inventory,
            output_dir=output_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind exact ligands extracted[/green] "
        f"paper_rows={result['paper_row_count']} "
        f"unique_ligands={result['unique_lig_file_count']}"
    )
    console.print(f"  manifest -> {result['manifest_path']}")

@benchmark_app.command("render-bayesbind-vina")
def benchmark_render_bayesbind_vina(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    target_id: str = typer.Argument(..., help="Exact target id from targets.json"),
    output_notebook: Path = typer.Argument(..., help="Rendered Kaggle notebook path"),
    run_id: str = typer.Option(..., "--run-id", help="Unique auditable run id"),
    sample_size: int = typer.Option(
        None, "--sample-size", min=2, help="Deterministic smoke/calibration subset; omit for full target"
    ),
    sample_seed: int = typer.Option(42, "--sample-seed", help="Deterministic sample and docking seed"),
    exact_sdf: bool = typer.Option(
        False,
        "--exact-sdf/--reconstructed-smiles",
        help="Use selectively extracted exact BigBind SDF bytes instead of SMILES reconstruction",
    ),
    ligand_workers: int = typer.Option(
        1,
        "--ligand-workers",
        min=1,
        help="Independent Vina ligand subprocesses; each keeps --cpu 1",
    ),
) -> None:
    """Stage a paper-protocol CPU Vina notebook through the public workflow."""
    from ezscreen.benchmark.bayesbind_paper import stage_vina_notebook

    try:
        result = stage_vina_notebook(
            campaign_dir,
            target_id,
            output_notebook,
            run_id=run_id,
            sample_size=sample_size,
            sample_seed=sample_seed,
            use_exact_sdf=exact_sdf,
            ligand_workers=ligand_workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind Vina notebook staged[/green] "
        f"target={result['target_id']} compounds={result['sample_size']} "
        f"actives={result['active_count']} random={result['random_count']}"
    )
    console.print(f"  notebook -> {result['notebook']}")
    console.print(f"  manifest -> {result['manifest']}")


@benchmark_app.command("freeze-bayesbind-full-run-plan")
def benchmark_freeze_bayesbind_full_run_plan(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    calibration_output_dir: Path = typer.Argument(..., help="Finalized calibration output directory"),
    calibration_stage_manifest: Path = typer.Argument(..., help="Calibration stage_manifest.json"),
    method: str = typer.Option("vina", "--method", help="Method to plan: vina or gnina"),
    target_id: list[str] = typer.Option(
        None,
        "--target-id",
        help="Target id to include; repeat as needed, omit for all frozen targets",
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help="Plan JSON output path; defaults under campaign full_run_plans/",
    ),
    account_count: int = typer.Option(
        None,
        "--account-count",
        min=1,
        help="Override quota account count",
    ),
    max_notebooks_per_account: int = typer.Option(
        None,
        "--max-notebooks-per-account",
        min=1,
        help="Override max concurrent notebooks per account",
    ),
    safety_factor: float = typer.Option(
        1.25,
        "--safety-factor",
        min=1.0,
        help="Runtime multiplier applied to measured calibration seconds/compound",
    ),
    max_shard_wall_hours: float = typer.Option(
        11.0,
        "--max-shard-wall-hours",
        min=0.1,
        help="Maximum estimated walltime for each planned notebook shard",
    ),
) -> None:
    """Freeze calibration-derived full-target shard and quota plan."""
    from ezscreen.benchmark.bayesbind_paper import freeze_bayesbind_full_run_plan

    try:
        result = freeze_bayesbind_full_run_plan(
            campaign_dir,
            calibration_output_dir,
            calibration_stage_manifest,
            method=method,
            target_ids=target_id,
            output_path=output_path,
            account_count=account_count,
            max_notebooks_per_account=max_notebooks_per_account,
            safety_factor=safety_factor,
            max_shard_wall_hours=max_shard_wall_hours,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    quota = result["quota_gate"]
    planning = result["planning"]
    console.print(
        f"[green]BayesBind full-run plan frozen[/green] "
        f"method={result['method']} status={result['status']} "
        f"targets={planning['target_count']} notebooks={planning['total_notebooks']} "
        f"planned_gpu={quota['planned_gpu_hours']:.3f}h"
    )
    console.print(f"  plan -> {result['plan_path']}")


@benchmark_app.command("render-bayesbind-plan-shard")
def benchmark_render_bayesbind_plan_shard(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    plan_path: Path = typer.Argument(..., help="Ready full-run plan JSON"),
    target_id: str = typer.Argument(..., help="Target id present in the plan"),
    shard_index: int = typer.Argument(..., min=1, help="1-based shard index from the plan"),
    output_notebook: Path = typer.Option(
        None,
        "--output-notebook",
        help="Rendered notebook path; defaults under campaign notebooks/",
    ),
    ligand_workers: int = typer.Option(
        None,
        "--ligand-workers",
        min=1,
        help="Override Vina ligand workers for this shard",
    ),
    gnina_workers: int = typer.Option(
        None,
        "--gnina-workers",
        min=1,
        help="Override GNINA workers for this shard",
    ),
) -> None:
    """Render one exact-SDF shard from a frozen BayesBind full-run plan."""
    from ezscreen.benchmark.bayesbind_paper import stage_bayesbind_plan_shard

    try:
        result = stage_bayesbind_plan_shard(
            campaign_dir,
            plan_path,
            target_id,
            shard_index,
            output_notebook=output_notebook,
            ligand_workers=ligand_workers,
            gnina_workers=gnina_workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind planned shard staged[/green] "
        f"method={result['method']} target={result['target_id']} "
        f"shard={result['shard_index']} compounds={result['sample_size']}"
    )
    console.print(f"  notebook -> {result['notebook']}")
    console.print(f"  manifest -> {result['manifest']}")


@benchmark_app.command("submit-bayesbind-plan-shards")
def benchmark_submit_bayesbind_plan_shards(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    plan_path: Path = typer.Argument(..., help="Ready full-run plan JSON"),
    target_id: list[str] = typer.Option(
        None,
        "--target-id",
        help="Target id to submit; repeat as needed, omit for all plan targets",
    ),
    shard_index: list[int] = typer.Option(
        None,
        "--shard-index",
        min=1,
        help="Shard index to submit; repeat as needed, omit for all selected target shards",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        help="Override backend; defaults cpu for Vina, t4x2 for GNINA",
    ),
    account_name: str = typer.Option(
        None,
        "--account",
        help="Kaggle account label; omit for first healthy account",
    ),
    stage: str = typer.Option("baseline", "--stage", help="Supervisor quota stage label"),
    max_submissions: int = typer.Option(
        None,
        "--max-submissions",
        min=1,
        help="Submit only the first N selected shards",
    ),
    resume_blocked: bool = typer.Option(
        False,
        "--resume-blocked",
        help="Resume a blocked campaign before first submission",
    ),
) -> None:
    """Stage and submit exact-SDF BayesBind shards from a frozen plan."""
    from ezscreen.benchmark.bayesbind_paper import submit_bayesbind_plan_shards

    try:
        result = submit_bayesbind_plan_shards(
            campaign_dir,
            plan_path,
            target_ids=target_id,
            shard_indices=shard_index,
            backend=backend,
            account_name=account_name,
            stage=stage,
            max_submissions=max_submissions,
            resume_blocked=resume_blocked,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind planned shards submitted[/green] "
        f"method={result['method']} backend={result['backend']} "
        f"count={result['submitted_count']} planned_gpu={result['planned_gpu_hours']:.3f}h"
    )


@benchmark_app.command("finalize-bayesbind-plan-target")
def benchmark_finalize_bayesbind_plan_target(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    plan_path: Path = typer.Argument(..., help="Ready full-run plan JSON"),
    target_id: str = typer.Argument(..., help="Target id to finalize from completed plan shards"),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        help="Target-level output directory; defaults under campaign final_results/",
    ),
) -> None:
    """Merge completed BayesBind plan shards into target-level metrics."""
    from ezscreen.benchmark.bayesbind_paper import finalize_bayesbind_plan_target

    try:
        result = finalize_bayesbind_plan_target(
            campaign_dir,
            plan_path,
            target_id,
            output_dir=output_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    counts = result["counts"]
    console.print(
        f"[green]BayesBind plan target finalized[/green] "
        f"method={result['method']} target={result['target_id']} "
        f"coverage={result['coverage']:.3f} scored={counts['scored']} failed={counts['failed']}"
    )
    console.print(f"  metrics -> {result['metrics_path']}")


@benchmark_app.command("preflight-bayesbind-execution")
def benchmark_preflight_bayesbind_execution(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    method: str = typer.Option("vina", "--method", help="Method to preflight: vina or gnina"),
    expected_target_count: int = typer.Option(
        25,
        "--expected-target-count",
        min=1,
        help="Expected target count for exact BayesBind protocol",
    ),
    plan_path: Path = typer.Option(
        None,
        "--plan-path",
        help="Full-run plan JSON to preflight; defaults to newest ready method plan",
    ),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help="Preflight JSON output path; defaults under campaign preflight/",
    ),
    minimum_free_gb_warning: float = typer.Option(
        50.0,
        "--minimum-free-gb-warning",
        min=0.0,
        help="Warn when campaign disk free space is below this GiB",
    ),
) -> None:
    """Report exact BayesBind execution readiness without using credentials."""
    from ezscreen.benchmark.bayesbind_paper import preflight_bayesbind_execution

    try:
        result = preflight_bayesbind_execution(
            campaign_dir,
            method=method,
            expected_target_count=expected_target_count,
            output_path=output_path,
            minimum_free_gb_warning=minimum_free_gb_warning,
            plan_path=plan_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind execution preflight written[/green] "
        f"method={result['method']} status={result['status']} "
        f"issues={len(result['issues'])} warnings={len(result['warnings'])}"
    )
    console.print(f"  next -> {result['next_action']}")
    console.print(f"  preflight -> {result['preflight_path']}")


@benchmark_app.command("audit-bayesbind-campaign")
def benchmark_audit_bayesbind_campaign(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    plan_path: Path = typer.Argument(..., help="Ready full-run plan JSON"),
    method: str = typer.Option(None, "--method", help="Expected method; defaults from plan"),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help="Audit JSON output path; defaults under final_results/",
    ),
    require_official_comparison: bool = typer.Option(
        False,
        "--require-official-comparison",
        help="Require per-target official comparison JSON files to be ready",
    ),
    expected_target_count: int = typer.Option(
        25,
        "--expected-target-count",
        min=1,
        help="Expected number of plan targets for exact BayesBind protocol",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero when audit is not ready",
    ),
) -> None:
    """Audit finalized BayesBind target outputs against a frozen full-run plan."""
    from ezscreen.benchmark.bayesbind_paper import audit_bayesbind_campaign_results

    try:
        result = audit_bayesbind_campaign_results(
            campaign_dir,
            plan_path,
            method=method,
            output_path=output_path,
            require_official_comparison=require_official_comparison,
            expected_target_count=expected_target_count,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind campaign audit written[/green] "
        f"method={result['method']} status={result['status']} "
        f"targets={result['ready_target_count']}/{result['target_count']} "
        f"issues={len(result['issues'])}"
    )
    console.print(f"  audit -> {result['audit_path']}")
    if strict and result["status"] != "ready":
        raise typer.Exit(1)


@benchmark_app.command("compare-bayesbind-official")
def benchmark_compare_bayesbind_official(
    output_dir: Path = typer.Argument(..., help="Finalized rerun output directory"),
    intended_csv: Path = typer.Argument(..., help="Frozen intended source-compound CSV"),
    official_dir: Path = typer.Argument(..., help="Recovered official baseline reference directory"),
    method: str = typer.Option("vina", "--method", help="Official method: vina or gnina"),
    output_path: Path = typer.Option(
        None,
        "--output-path",
        help="Comparison JSON output path; defaults into output_dir",
    ),
) -> None:
    """Compare finalized BayesBind rerun outputs with official reference evidence."""
    from ezscreen.benchmark.bayesbind_paper import compare_bayesbind_rerun_to_official

    try:
        result = compare_bayesbind_rerun_to_official(
            output_dir,
            intended_csv,
            official_dir,
            method=method,
            output_path=output_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    counts = result["counts"]
    console.print(
        f"[green]BayesBind official comparison written[/green] "
        f"target={result['target_id']} status={result['status']} "
        f"paired={counts['paired_scored']} missing_rerun={counts['rerun_missing_intended']} "
        f"missing_official={counts['official_missing_intended']}"
    )
    console.print(f"  comparison -> {result['comparison_path']}")


@benchmark_app.command("recover-bayesbind-official-baseline")
def benchmark_recover_bayesbind_official_baseline(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    method: str = typer.Option("vina", "--method", help="Official baseline method: vina or gnina"),
    target_id: list[str] = typer.Option(
        None,
        "--target-id",
        help="Target id to recover; repeat as needed, omit for all frozen targets",
    ),
    archive: Path = typer.Option(
        None,
        "--archive",
        help="Official BayesBindBaselinesV1.5 archive path",
    ),
    inventory: Path = typer.Option(
        None,
        "--inventory",
        help="Official baseline member inventory JSONL",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        help="Reference output directory",
    ),
) -> None:
    """Recover and compare official BayesBind baseline score evidence."""
    from ezscreen.benchmark.bayesbind_paper import recover_official_bayesbind_baseline

    try:
        result = recover_official_bayesbind_baseline(
            campaign_dir,
            method=method,
            archive_path=archive,
            inventory_path=inventory,
            output_dir=output_dir,
            target_ids=target_id,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    totals = result["totals"]
    console.print(
        f"[green]BayesBind official baseline recovered[/green] "
        f"method={result['method']} status={result['status']} "
        f"targets={totals['target_count']} rows={totals['official_result_rows']} "
        f"unmatched={totals['source_unmatched']} missing_poses={totals['pose_members_missing']}"
    )
    console.print(f"  manifest -> {result['manifest_path']}")


@benchmark_app.command("finalize-bayesbind-vina")
def benchmark_finalize_bayesbind_vina(
    output_dir: Path = typer.Argument(..., help="Recovered Kaggle output directory"),
    intended_csv: Path = typer.Argument(..., help="Exact intended source-compound CSV"),
) -> None:
    """Calculate validity-corrected target metrics from recovered Vina outputs."""
    from ezscreen.benchmark.bayesbind_paper import finalize_vina_output

    try:
        result = finalize_vina_output(output_dir, intended_csv)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind Vina output finalized[/green] "
        f"target={result['target_id']} coverage={result['coverage']:.3f} "
        f"EF1={result['ef1']:.3f} AUC={result['roc_auc']:.4f}"
    )



@benchmark_app.command("render-bayesbind-gnina")
def benchmark_render_bayesbind_gnina(
    campaign_dir: Path = typer.Argument(..., help="Frozen BayesBind campaign directory"),
    target_id: str = typer.Argument(..., help="Exact target id from targets.json"),
    output_notebook: Path = typer.Argument(..., help="Rendered Kaggle notebook path"),
    run_id: str = typer.Option(..., "--run-id", help="Unique auditable run id"),
    sample_size: int = typer.Option(
        None, "--sample-size", min=2, help="Deterministic smoke/calibration subset; omit for full target"
    ),
    sample_seed: int = typer.Option(42, "--sample-seed", help="Deterministic sample seed"),
    gnina_workers: int = typer.Option(
        1,
        "--gnina-workers",
        min=1,
        help="Independent GNINA ligand subprocesses; default 1 for GPU stability",
    ),
) -> None:
    """Stage a paper-comparator GNINA full-docking notebook."""
    from ezscreen.benchmark.bayesbind_paper import stage_gnina_notebook

    try:
        result = stage_gnina_notebook(
            campaign_dir,
            target_id,
            output_notebook,
            run_id=run_id,
            sample_size=sample_size,
            sample_seed=sample_seed,
            gnina_workers=gnina_workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind GNINA notebook staged[/green] "
        f"target={result['target_id']} compounds={result['sample_size']} "
        f"actives={result['active_count']} random={result['random_count']}"
    )
    console.print(f"  notebook -> {result['notebook']}")
    console.print(f"  manifest -> {result['manifest']}")


@benchmark_app.command("finalize-bayesbind-gnina")
def benchmark_finalize_bayesbind_gnina(
    output_dir: Path = typer.Argument(..., help="Recovered Kaggle output directory"),
    intended_csv: Path = typer.Argument(..., help="Exact intended source-compound CSV"),
) -> None:
    """Calculate validity-corrected target metrics from recovered GNINA outputs."""
    from ezscreen.benchmark.bayesbind_paper import finalize_gnina_output

    try:
        result = finalize_gnina_output(output_dir, intended_csv)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]BayesBind GNINA output finalized[/green] "
        f"target={result['target_id']} coverage={result['coverage']:.3f} "
        f"EF1={result['ef1']:.3f} AUC={result['roc_auc']:.4f}"
    )


@team_app.command("serve")
def team_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8787, "--port", help="Bind port"),
    db: Path = typer.Option(None, "--db", help="SQLite state path"),
    storage: Path = typer.Option(None, "--storage", help="Result ZIP storage path"),
) -> None:
    """Run the stdlib HTTP control-plane server."""
    from ezscreen.team.server import serve

    serve(host=host, port=port, db_path=db, storage_root=storage)


@team_app.command("login")
def team_login(
    server_url: str = typer.Option(
        None,
        "--server-url",
        help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url.",
    ),
    poll_seconds: int = typer.Option(
        180, "--poll-seconds", help="How long to wait for GitHub authorization."
    ),
) -> None:
    """Sign in to the team server with GitHub device login."""
    import time
    import webbrowser

    from ezscreen.team.client import (
        TeamClient,
        default_server_url,
        load_local_state,
        save_local_state,
    )

    server_url = server_url or default_server_url()
    client = TeamClient(server_url)
    start = client.start_github_device_login()
    verify_url = start.get("verification_uri_complete") or start.get("verification_uri")
    if verify_url:
        webbrowser.open(verify_url)
    console.print(
        f"Open GitHub and enter code: [bold]{start.get('user_code', '')}[/bold]"
    )
    if verify_url:
        console.print(verify_url)
    deadline = time.time() + poll_seconds
    interval = int(start.get("interval") or 5)
    while time.time() < deadline:
        time.sleep(interval)
        result = client.poll_github_device_login(start["device_code"])
        if result.get("status") == "pending":
            if result.get("error") == "slow_down":
                interval += 5
            continue
        state = load_local_state()
        state.update(
            {
                "server_url": server_url,
                "session_token": result["session_token"],
                "user_id": result["user_id"],
                "user_name": result.get("user_name") or result.get("github_login", ""),
                "github_login": result.get("github_login", ""),
            }
        )
        save_local_state(state)
        console.print(
            f"[green]Signed in[/green] {state.get('github_login') or state['user_name']}"
        )
        return
    console.print("[yellow]GitHub authorization timed out.[/yellow]")
    raise typer.Exit(1)


@team_app.command("logout")
def team_logout() -> None:
    """Sign out of the local ezscreen account session."""
    from ezscreen.team.client import (
        TeamClient,
        clear_local_session,
        default_server_url,
        load_local_state,
    )

    state = load_local_state()
    session_token = state.get("session_token")
    if not session_token:
        console.print(
            "[yellow]No ezscreen account session is currently signed in.[/yellow]"
        )
        return
    user_name = state.get("user_name") or state.get("github_login") or "account"
    TeamClient(
        state.get("server_url") or default_server_url(), token=session_token
    ).logout()
    kept = clear_local_session()
    console.print(f"[green]Signed out[/green] {user_name}")
    if kept.get("server_url"):
        console.print(f"Team server kept at {kept['server_url']}")


@team_app.command("create")
def team_create(
    server_url: str = typer.Option(
        None,
        "--server-url",
        help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url.",
    ),
    owner: str = typer.Option(..., "--owner", help="Owner display name"),
    team_name: str = typer.Option(..., "--name", help="Team name"),
    github_repo: str = typer.Option(
        "",
        "--repo",
        help="Optional dedicated worker repo override. Defaults to an ezscreen-managed private repo.",
    ),
    team_account: str = typer.Option(
        None,
        "--team-account",
        help="Saved ezscreen team account name to use for this worker's Kaggle key.",
    ),
    kaggle_username: str = typer.Option(
        None,
        "--kaggle-username",
        help="Worker Kaggle username; defaults to local kaggle.json username",
    ),
    kaggle_key: str = typer.Option(
        None,
        "--kaggle-key",
        help="Worker Kaggle API key; defaults to KAGGLE_KEY or local kaggle.json",
    ),
    use_local_kaggle: bool = typer.Option(
        False,
        "--use-local-kaggle/--no-use-local-kaggle",
        help="Read existing ezscreen Kaggle credentials if no key is provided",
    ),
    github_token: str = typer.Option(
        None,
        "--github-token",
        help="GitHub token with repo and Actions secrets write access. Defaults to GITHUB_TOKEN.",
    ),
) -> None:
    """Create a team and optionally provision a dedicated private worker repo."""
    from ezscreen.team.client import (
        TeamClient,
        default_server_url,
        load_local_state,
        save_local_state,
    )

    server_url = server_url or default_server_url()
    state = load_local_state()
    session_token = state.get("session_token")
    owner_user_id = state.get("user_id") if session_token else None

    worker_repo = ""
    github_token = github_token or os.environ.get("GITHUB_TOKEN")
    if github_token:
        from ezscreen.team.github import provision_managed_worker_repo

        repo_info = provision_managed_worker_repo(
            github_token=github_token, repo_name=github_repo or None
        )
        worker_repo = repo_info["repo"]
    elif github_repo:
        worker_repo = github_repo

    result = TeamClient(server_url, token=session_token).create_team(
        owner_name=owner,
        team_name=team_name,
        github_repo=worker_repo,
        owner_user_id=owner_user_id,
    )
    state.update(
        {
            "server_url": server_url,
            "user_id": result["owner_user_id"],
            "team_id": result["team_id"],
            "member_id": result["member_id"],
            "user_name": owner,
            "github_repo": worker_repo,
        }
    )
    save_local_state(state)
    console.print(f"[green]Team created[/green] {result['team_id']}")
    console.print(f"Invite code: [bold]{result['invite_code']}[/bold]")

    if github_token and worker_repo:
        from ezscreen.team.github import configure_worker_repo
        from ezscreen.team.local_credentials import resolve_worker_kaggle_credentials

        kaggle_creds = resolve_worker_kaggle_credentials(
            team_account_name=team_account,
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            use_local=use_local_kaggle,
        )
        configure_worker_repo(
            repo=worker_repo,
            github_token=github_token,
            kaggle_username=kaggle_creds.username,
            kaggle_key=kaggle_creds.key,
            team_token=result["team_token"],
            server_url=server_url,
        )
        source = f" ({kaggle_creds.source})"
        console.print(f"[green]Dedicated worker ready[/green] {worker_repo}{source}")
    else:
        console.print(
            "Set GITHUB_TOKEN, then run `ezscreen team connect-worker` to provision a dedicated private worker repo."
        )


@team_app.command("join")
def team_join(
    server_url: str = typer.Option(
        None,
        "--server-url",
        help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url.",
    ),
    invite_code: str = typer.Option(..., "--invite", help="Team invite code"),
    user: str = typer.Option(..., "--user", help="Member display name"),
    repo: str = typer.Option(
        "",
        "--repo",
        help="Optional dedicated worker repo override. Defaults to an ezscreen-managed private repo.",
    ),
    team_account: str = typer.Option(
        None,
        "--team-account",
        help="Saved ezscreen team account name to use for this worker's Kaggle key.",
    ),
    kaggle_username: str = typer.Option(
        None,
        "--kaggle-username",
        help="Worker Kaggle username; defaults to local kaggle.json username",
    ),
    kaggle_key: str = typer.Option(
        None,
        "--kaggle-key",
        help="Worker Kaggle API key. Defaults to KAGGLE_KEY or local ezscreen kaggle.json.",
    ),
    use_local_kaggle: bool = typer.Option(
        False,
        "--use-local-kaggle/--no-use-local-kaggle",
        help="Read existing ezscreen Kaggle credentials if no key is provided.",
    ),
    github_token: str = typer.Option(
        None,
        "--github-token",
        help="GitHub token with repo and Actions secrets write access. Defaults to GITHUB_TOKEN.",
    ),
) -> None:
    """Join a team and provision a dedicated private worker repo."""
    from ezscreen.team.client import (
        TeamClient,
        default_server_url,
        load_local_state,
        save_local_state,
    )
    from ezscreen.team.github import (
        configure_worker_repo,
        provision_managed_worker_repo,
    )
    from ezscreen.team.local_credentials import resolve_worker_kaggle_credentials

    server_url = server_url or default_server_url()
    github_token = github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        console.print(
            "[red]GitHub token is required via --github-token or GITHUB_TOKEN.[/red]"
        )
        raise typer.Exit(1)

    kaggle_creds = resolve_worker_kaggle_credentials(
        team_account_name=team_account,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        use_local=use_local_kaggle,
    )
    state = load_local_state()
    session_token = state.get("session_token")
    repo_info = provision_managed_worker_repo(
        github_token=github_token, repo_name=repo or None
    )
    worker_repo = repo_info["repo"]

    join_kwargs = {
        "invite_code": invite_code,
        "user_name": user,
        "github_repo": worker_repo,
    }
    if session_token and state.get("user_id"):
        join_kwargs["user_id"] = state["user_id"]
    result = TeamClient(server_url, token=session_token).join_team(**join_kwargs)
    configure_worker_repo(
        repo=worker_repo,
        github_token=github_token,
        kaggle_username=kaggle_creds.username,
        kaggle_key=kaggle_creds.key,
        team_token=result["team_token"],
        server_url=server_url,
    )
    state.update(
        {
            "server_url": server_url,
            "user_id": result["user_id"],
            "team_id": result["team_id"],
            "member_id": result["member_id"],
            "user_name": user,
            "github_repo": worker_repo,
        }
    )
    save_local_state(state)
    source = f" ({kaggle_creds.source})"
    console.print(f"[green]Joined team[/green] {result['team_id']}")
    console.print(f"[green]Dedicated worker ready[/green] {worker_repo}{source}")


@team_app.command("status")
def team_status() -> None:
    """Show local team runner status."""
    from ezscreen.team.client import TeamClient, load_local_state
    from ezscreen.team.local_credentials import local_kaggle_summary

    state = load_local_state()
    local_kaggle = local_kaggle_summary()
    if local_kaggle.get("configured"):
        console.print(
            f"Local Kaggle: configured for {local_kaggle.get('username', '')} at {local_kaggle.get('path', '')}"
        )
    else:
        console.print("Local Kaggle: not configured")
    if not state.get("server_url") or not state.get("user_id"):
        console.print(
            "[yellow]No local team state. Use ezscreen team create or join.[/yellow]"
        )
        return
    status = TeamClient(state["server_url"]).status(state["user_id"])
    if not status.get("team"):
        console.print("[yellow]Not in an active team.[/yellow]")
        return
    team = status["team"]
    console.print(f"[bold]{team['name']}[/bold]  {team['team_id']}")
    for member in status.get("members", []):
        if not (member.get("github_repo") or "").strip():
            state_label = "disconnected"
        elif not member.get("github_installation_id"):
            state_label = "app pending"
        else:
            state_label = "paused" if member.get("paused") else "enabled"
        console.print(
            f"  {member['display_name']}  {member.get('github_repo') or 'no repo'}  "
            f"{member.get('kaggle_username') or 'no kaggle user'}  {state_label}"
        )


@team_app.command("pause-worker")
def team_pause_worker(
    paused: bool = typer.Option(
        True, "--paused/--enabled", help="Pause or re-enable this member's worker"
    ),
) -> None:
    """Pause or re-enable the local member worker."""
    from ezscreen.team.client import TeamClient, load_local_state

    state = load_local_state()
    if not state.get("server_url") or not state.get("user_id"):
        console.print(
            "[yellow]No local team state. Use ezscreen team create or join.[/yellow]"
        )
        raise typer.Exit(1)
    result = TeamClient(state["server_url"]).request(
        "POST",
        "/v1/team/worker/pause",
        {"user_id": state["user_id"], "paused": paused},
    )
    console.print(f"Worker {result['member_id']}: {'paused' if paused else 'enabled'}")


@team_app.command("connect-worker")
def team_connect_worker(
    repo: str = typer.Option(
        "",
        "--repo",
        help="Optional dedicated worker repo override. Defaults to an ezscreen-managed private repo.",
    ),
    server_url: str = typer.Option(
        None,
        "--server-url",
        help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url.",
    ),
    team_account: str = typer.Option(
        None,
        "--team-account",
        help="Saved ezscreen team account name to use for this worker's Kaggle key.",
    ),
    kaggle_username: str = typer.Option(
        None,
        "--kaggle-username",
        help="Worker Kaggle username; defaults to local kaggle.json username",
    ),
    kaggle_key: str = typer.Option(
        None,
        "--kaggle-key",
        help="Worker Kaggle API key; defaults to KAGGLE_KEY or local kaggle.json",
    ),
    use_local_kaggle: bool = typer.Option(
        False,
        "--use-local-kaggle/--no-use-local-kaggle",
        help="Read existing ezscreen Kaggle credentials if no key is provided",
    ),
    github_token: str = typer.Option(
        None,
        "--github-token",
        help="GitHub token with repo and Actions secrets write access. Defaults to GITHUB_TOKEN.",
    ),
) -> None:
    """Attach a dedicated private GitHub worker repo to the current member and seed its secrets."""
    from ezscreen.team.client import (
        TeamClient,
        default_server_url,
        load_local_state,
        save_local_state,
    )
    from ezscreen.team.github import (
        configure_worker_repo,
        provision_managed_worker_repo,
    )
    from ezscreen.team.local_credentials import resolve_worker_kaggle_credentials

    state = load_local_state()
    server_url = server_url or state.get("server_url") or default_server_url()
    session_token = state.get("session_token")
    if not session_token:
        console.print(
            "[red]Sign in first with `ezscreen team login` or the Team screen.[/red]"
        )
        raise typer.Exit(1)
    github_token = github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        console.print(
            "[red]GitHub token is required via --github-token or GITHUB_TOKEN.[/red]"
        )
        raise typer.Exit(1)

    kaggle_creds = resolve_worker_kaggle_credentials(
        team_account_name=team_account,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        use_local=use_local_kaggle,
    )
    repo_info = provision_managed_worker_repo(
        github_token=github_token, repo_name=repo or None
    )
    worker_repo = repo_info["repo"]
    client = TeamClient(server_url, token=session_token)
    result = client.connect_worker(
        github_repo=worker_repo, kaggle_username=kaggle_creds.username
    )
    configure_worker_repo(
        repo=worker_repo,
        github_token=github_token,
        kaggle_username=kaggle_creds.username,
        kaggle_key=kaggle_creds.key,
        team_token=result["team_token"],
        server_url=server_url,
    )
    state["server_url"] = server_url
    state["github_repo"] = worker_repo
    save_local_state(state)
    source = f" ({kaggle_creds.source})"
    action = "created" if repo_info.get("created") else "synced"
    console.print(f"[green]Worker connected[/green] {worker_repo}{source} [{action}]")
    if not result.get("github_installation_id"):
        install_url = str(result.get("app_install_url") or "").strip()
        if install_url:
            console.print(
                f"Install the ezscreen GitHub App for this repo: {install_url}"
            )
        else:
            console.print(
                "Install the ezscreen GitHub App on this repo to enable server-side dispatch."
            )


@team_app.command("disconnect-worker")
def team_disconnect_worker() -> None:
    """Disconnect the current member worker without leaving the team."""
    from ezscreen.team.client import TeamClient, load_local_state, save_local_state

    state = load_local_state()
    if not state.get("server_url") or not state.get("session_token"):
        console.print(
            "[yellow]Sign in first, then open Team or use team login.[/yellow]"
        )
        raise typer.Exit(1)
    client = TeamClient(state["server_url"], token=state["session_token"])
    result = client.disconnect_worker(user_id=state.get("user_id"))
    state["github_repo"] = ""
    save_local_state(state)
    console.print(f"[green]Worker disconnected[/green] {result['member_id']}")
