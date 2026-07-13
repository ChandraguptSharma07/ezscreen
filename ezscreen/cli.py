from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from ezscreen import version_check

app = typer.Typer(
    name="ezscreen",
    help="GPU-accelerated virtual screening â€” powered by Kaggle GPUs.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

console = Console()
team_app = typer.Typer(help="Manage team runners and worker control-plane state.")
app.add_typer(team_app, name="team")

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
        None, "--update", "-u",
        help="Update a specific key: 'Kaggle credentials' | 'NIM API key' | 'Both'",
    ),
) -> None:
    """Set up or update Kaggle and NIM credentials."""
    from ezscreen.commands import auth as _auth
    _auth.invoke(update=update)


@app.command()
def validate(
    receptor: Path = typer.Argument(..., help="Path to receptor PDB file"),
    hits:   Path = typer.Argument(..., help="Path to hits SDF or CSV file"),
    output: Path = typer.Option(
        Path("validation_out"), "--output", "-o", help="Output directory"
    ),
) -> None:
    """Run Stage 2 hit validation with DiffDock-L via NVIDIA NIM."""
    from ezscreen.commands import validate as _validate
    _validate.invoke(receptor_path=receptor, hits_path=hits, output_dir=output)


@app.command()
def admet(
    input_file:  Path = typer.Argument(..., help="Input SDF file"),
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
            f"[red]Run {run_id} failed â€” start a fresh run with: ezscreen[/red]"
        )
        raise typer.Exit(1)

    incomplete = checkpoint.get_incomplete_shards(run_id)
    if not incomplete:
        console.print(
            f"[dim]Run {run_id} has no incomplete shards â€” already done.[/dim]"
        )
        return

    console.print(Panel(
        f"[bold]{run_id}[/bold]  status: {run['status']}\n"
        f"{len(incomplete)} shard(s) still incomplete",
        title="[bold]Resume[/bold]",
    ))
    console.print(
        "[dim]Re-submit uses the same run.py flow with resume context.[/dim]"
    )
    console.print(
        "[yellow]Full resume is planned for v1.1 â€” launch ezscreen to start fresh.[/yellow]"  # noqa: E501
    )


@app.command()
def download(
    run_id: str = typer.Argument(..., help="Run ID to download results for (e.g. ezs-4f2a8c)"),  # noqa: E501
) -> None:
    """Download results for a completed Kaggle run (if download failed after a run)."""
    from ezscreen import auth as _auth
    from ezscreen.backends.kaggle import runner as kaggle_runner
    from ezscreen.commands import view as _view

    creds    = _auth.load_credentials()
    kaggle_path = _auth.get_kaggle_json_path(creds)
    import json
    username = (
        json.loads(kaggle_path.read_text())["username"]
        if kaggle_path and kaggle_path.exists()
        else None
    )
    if not username:
        console.print("[red]Kaggle credentials not found â€” run: ezscreen auth[/red]")
        raise typer.Exit(1)
    kernel_ref = f"{username}/{run_id}"
    work_dir   = Path.home() / ".ezscreen" / "runs" / run_id

    console.print(f"  [dim]Downloading results for {kernel_ref}...[/dim]")
    output_dir = kaggle_runner._download_output(kernel_ref, work_dir)
    console.print(f"  [green]âœ“ Results â†’ {output_dir}[/green]")
    _view.invoke(results_dir=output_dir)


@app.command()
def clean(
    run_id: str = typer.Argument(..., help="Run ID to clean (e.g. ezs-4f2a8c)"),
) -> None:
    """Delete Kaggle dataset and kernel artifacts for a run."""
    import questionary

    from ezscreen import auth as _auth
    from ezscreen.backends.kaggle import runner as kaggle_runner

    creds    = _auth.load_credentials()
    username = Path(creds.get("kaggle_json_path", "")).stem or "user"

    confirmed = questionary.confirm(
        f"Delete all Kaggle artifacts for {run_id}?", default=False
    ).ask()
    if not confirmed:
        return

    kaggle_runner.clean_run(run_id, username)
    console.print(f"  [green]âœ“ Cleaned {run_id}[/green]")
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
    server_url: str = typer.Option(None, "--server-url", help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url."),
    poll_seconds: int = typer.Option(180, "--poll-seconds", help="How long to wait for GitHub authorization."),
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
    console.print(f"Open GitHub and enter code: [bold]{start.get('user_code', '')}[/bold]")
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
        console.print(f"[green]Signed in[/green] {state.get('github_login') or state['user_name']}")
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
        console.print("[yellow]No ezscreen account session is currently signed in.[/yellow]")
        return
    user_name = state.get("user_name") or state.get("github_login") or "account"
    TeamClient(state.get("server_url") or default_server_url(), token=session_token).logout()
    kept = clear_local_session()
    console.print(f"[green]Signed out[/green] {user_name}")
    if kept.get("server_url"):
        console.print(f"Team server kept at {kept['server_url']}")

@team_app.command("create")
def team_create(
    server_url: str = typer.Option(None, "--server-url", help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url."),
    owner: str = typer.Option(..., "--owner", help="Owner display name"),
    team_name: str = typer.Option(..., "--name", help="Team name"),
    github_repo: str = typer.Option("", "--repo", help="Optional dedicated worker repo override. Defaults to an ezscreen-managed private repo."),
    team_account: str = typer.Option(None, "--team-account", help="Saved ezscreen team account name to use for this worker's Kaggle key."),
    kaggle_username: str = typer.Option(None, "--kaggle-username", help="Worker Kaggle username; defaults to local kaggle.json username"),
    kaggle_key: str = typer.Option(None, "--kaggle-key", help="Worker Kaggle API key; defaults to KAGGLE_KEY or local kaggle.json"),
    use_local_kaggle: bool = typer.Option(False, "--use-local-kaggle/--no-use-local-kaggle", help="Read existing ezscreen Kaggle credentials if no key is provided"),
    github_token: str = typer.Option(None, "--github-token", help="GitHub token with repo and Actions secrets write access. Defaults to GITHUB_TOKEN."),
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

        repo_info = provision_managed_worker_repo(github_token=github_token, repo_name=github_repo or None)
        worker_repo = repo_info["repo"]
    elif github_repo:
        worker_repo = github_repo

    result = TeamClient(server_url, token=session_token).create_team(
        owner_name=owner, team_name=team_name, github_repo=worker_repo, owner_user_id=owner_user_id
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
        console.print("Set GITHUB_TOKEN, then run `ezscreen team connect-worker` to provision a dedicated private worker repo.")


@team_app.command("join")
def team_join(
    server_url: str = typer.Option(None, "--server-url", help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url."),
    invite_code: str = typer.Option(..., "--invite", help="Team invite code"),
    user: str = typer.Option(..., "--user", help="Member display name"),
    repo: str = typer.Option("", "--repo", help="Optional dedicated worker repo override. Defaults to an ezscreen-managed private repo."),
    team_account: str = typer.Option(None, "--team-account", help="Saved ezscreen team account name to use for this worker's Kaggle key."),
    kaggle_username: str = typer.Option(None, "--kaggle-username", help="Worker Kaggle username; defaults to local kaggle.json username"),
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
        console.print("[red]GitHub token is required via --github-token or GITHUB_TOKEN.[/red]")
        raise typer.Exit(1)

    kaggle_creds = resolve_worker_kaggle_credentials(
        team_account_name=team_account,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        use_local=use_local_kaggle,
    )
    state = load_local_state()
    session_token = state.get("session_token")
    repo_info = provision_managed_worker_repo(github_token=github_token, repo_name=repo or None)
    worker_repo = repo_info["repo"]

    join_kwargs = {"invite_code": invite_code, "user_name": user, "github_repo": worker_repo}
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
        console.print(f"Local Kaggle: configured for {local_kaggle.get('username', '')} at {local_kaggle.get('path', '')}")
    else:
        console.print("Local Kaggle: not configured")
    if not state.get("server_url") or not state.get("user_id"):
        console.print("[yellow]No local team state. Use ezscreen team create or join.[/yellow]")
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
    paused: bool = typer.Option(True, "--paused/--enabled", help="Pause or re-enable this member's worker"),
) -> None:
    """Pause or re-enable the local member worker."""
    from ezscreen.team.client import TeamClient, load_local_state

    state = load_local_state()
    if not state.get("server_url") or not state.get("user_id"):
        console.print("[yellow]No local team state. Use ezscreen team create or join.[/yellow]")
        raise typer.Exit(1)
    result = TeamClient(state["server_url"]).request(
        "POST",
        "/v1/team/worker/pause",
        {"user_id": state["user_id"], "paused": paused},
    )
    console.print(f"Worker {result['member_id']}: {'paused' if paused else 'enabled'}")


@team_app.command("connect-worker")
def team_connect_worker(
    repo: str = typer.Option("", "--repo", help="Optional dedicated worker repo override. Defaults to an ezscreen-managed private repo."),
    server_url: str = typer.Option(None, "--server-url", help="Team server URL. Defaults to EZSCREEN_TEAM_SERVER_URL or config team.server_url."),
    team_account: str = typer.Option(None, "--team-account", help="Saved ezscreen team account name to use for this worker's Kaggle key."),
    kaggle_username: str = typer.Option(None, "--kaggle-username", help="Worker Kaggle username; defaults to local kaggle.json username"),
    kaggle_key: str = typer.Option(None, "--kaggle-key", help="Worker Kaggle API key; defaults to KAGGLE_KEY or local kaggle.json"),
    use_local_kaggle: bool = typer.Option(False, "--use-local-kaggle/--no-use-local-kaggle", help="Read existing ezscreen Kaggle credentials if no key is provided"),
    github_token: str = typer.Option(None, "--github-token", help="GitHub token with repo and Actions secrets write access. Defaults to GITHUB_TOKEN."),
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
        console.print("[red]Sign in first with `ezscreen team login` or the Team screen.[/red]")
        raise typer.Exit(1)
    github_token = github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        console.print("[red]GitHub token is required via --github-token or GITHUB_TOKEN.[/red]")
        raise typer.Exit(1)

    kaggle_creds = resolve_worker_kaggle_credentials(
        team_account_name=team_account,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        use_local=use_local_kaggle,
    )
    repo_info = provision_managed_worker_repo(github_token=github_token, repo_name=repo or None)
    worker_repo = repo_info["repo"]
    client = TeamClient(server_url, token=session_token)
    result = client.connect_worker(github_repo=worker_repo, kaggle_username=kaggle_creds.username)
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
            console.print(f"Install the ezscreen GitHub App for this repo: {install_url}")
        else:
            console.print("Install the ezscreen GitHub App on this repo to enable server-side dispatch.")


@team_app.command("disconnect-worker")
def team_disconnect_worker() -> None:
    """Disconnect the current member worker without leaving the team."""
    from ezscreen.team.client import TeamClient, load_local_state, save_local_state

    state = load_local_state()
    if not state.get("server_url") or not state.get("session_token"):
        console.print("[yellow]Sign in first, then open Team or use team login.[/yellow]")
        raise typer.Exit(1)
    client = TeamClient(state["server_url"], token=state["session_token"])
    result = client.disconnect_worker(user_id=state.get("user_id"))
    state["github_repo"] = ""
    save_local_state(state)
    console.print(f"[green]Worker disconnected[/green] {result['member_id']}")

