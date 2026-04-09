from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ezscreen import auth, checkpoint, config
from ezscreen.admet.filter import filter_library
from ezscreen.backends.kaggle import runner as kaggle_runner
from ezscreen.pocket import detect as pocket
from ezscreen.prep import ligands as ligand_prep
from ezscreen.prep import receptor as rec_prep
from ezscreen.state import BACK, make_context

console = Console()

# ---------------------------------------------------------------------------
# Search depth presets (Section 7.1)
# ---------------------------------------------------------------------------

_PRESETS = {
    "Fast       · triage only · misses ~25% best poses":
        {"search_mode": "fast"},
    "Balanced ★ · standard VS · good for rigid pockets":
        {"search_mode": "balance"},
    "Thorough   · flexible ligands · induced-fit targets":
        {"search_mode": "detail"},
    "Expert     · set all parameters manually": None,
}

# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _run_id() -> str:
    return "ezs-" + secrets.token_hex(3)


def _work_dir(run_id: str) -> Path:
    return Path.home() / ".ezscreen" / "runs" / run_id


# ---------------------------------------------------------------------------
# Step 1 — Receptor input
# ---------------------------------------------------------------------------

def _step_receptor(ctx: dict) -> dict | object:
    raw = questionary.text(
        "Receptor — PDB ID (e.g. 7L11) or path to a .pdb file:",
        default=ctx.get("receptor_raw", ""),
    ).ask()
    if raw is None:
        return BACK

    raw = raw.strip()
    work_dir = ctx["work_dir"]

    if len(raw) == 4 and raw.isalnum():
        console.print(f"  [dim]Fetching {raw.upper()} from RCSB...[/dim]")
        pdb_path = rec_prep.fetch_pdb(raw.upper(), work_dir / "receptor_raw")
        ctx["pdb_source"] = "rcsb"
        ctx["pdb_id"]     = raw.upper()
    else:
        pdb_path = Path(raw).expanduser()
        if not pdb_path.exists():
            console.print(f"[red]File not found: {pdb_path}[/red]")
            return _step_receptor(ctx)
        ctx["pdb_source"] = "local"
        ctx["pdb_id"]     = None

    is_af, af_ver = rec_prep.detect_alphafold(pdb_path)
    chains         = rec_prep.get_chains(pdb_path)

    ctx.update({
        "receptor_raw": raw,
        "pdb_path":     pdb_path,
        "chains":       chains,
        "is_alphafold": is_af,
        "af_version":   af_ver,
    })
    return ctx


# ---------------------------------------------------------------------------
# Step 2 — Chain selection
# ---------------------------------------------------------------------------

def _step_chains(ctx: dict) -> dict | object:
    chains = ctx["chains"]
    result = rec_prep.prompt_chain_selection(chains)
    if result is BACK:
        return BACK
    ctx["selected_chains"] = result
    if len(result) > 1:
        console.print(
            f"  [yellow]Multi-chain receptor selected ({', '.join(result)}) — "
            "box calculation uses combined chain coordinates.[/yellow]"
        )
    return ctx


# ---------------------------------------------------------------------------
# Step 3 — AlphaFold warning
# ---------------------------------------------------------------------------

def _step_alphafold_warn(ctx: dict) -> dict | object:
    if not ctx.get("is_alphafold"):
        return ctx

    ver_label = {"af2": "AlphaFold 2", "af3": "AlphaFold 3"}.get(
        ctx.get("af_version"), "AlphaFold"
    )
    console.print(Panel(
        f"[yellow bold]⚠  {ver_label} structure detected[/yellow bold]\n\n"
        "  Pocket prediction is less reliable on predicted models.\n"
        "  P2Rank will use [bold]--profile alphafold[/bold] automatically.\n"
        "  Co-crystal ligand detection is not applicable.",
        title="[bold yellow]AlphaFold Warning[/bold yellow]",
    ))

    choice = questionary.select(
        "How would you like to proceed?",
        choices=["Continue with this structure", "Abort — pick a different structure", "← Back"],
    ).ask()

    if choice is None or choice == "← Back":
        return BACK
    if "Abort" in choice:
        console.print("[dim]Aborted. Re-run ezscreen run to start fresh.[/dim]")
        raise SystemExit(0)

    ctx["af_warning_accepted"] = True
    return ctx


# ---------------------------------------------------------------------------
# Step 4 — Binding site method
# ---------------------------------------------------------------------------

def _step_binding_site(ctx: dict) -> dict | object:
    pdb_path = ctx["pdb_path"]
    is_af    = ctx.get("is_alphafold", False)
    chains   = ctx["selected_chains"]

    # Co-crystal: only if not AF
    cocrystal_ligands = [] if is_af else pocket.find_cocrystal_ligands(pdb_path)
    co_label = (
        f"Co-crystal ligand ({', '.join(l['resname'] for l in cocrystal_ligands)})"
        if cocrystal_ligands else None
    )

    choices = []
    if co_label:
        choices.append(co_label)
    choices += [
        "Define by active site residues",
        "P2Rank pocket prediction (shows top 3)",
        "Blind docking — whole protein  ⚠",
        "← Back",
    ]

    method = questionary.select("Binding site method:", choices=choices).ask()
    if method is None or method == "← Back":
        return BACK

    box: dict[str, Any] = {}

    if co_label and method == co_label:
        if len(cocrystal_ligands) == 1:
            lig = cocrystal_ligands[0]
        else:
            pick = questionary.select(
                "Multiple ligands found — which to use as reference?",
                choices=[f"{l['resname']} (chain {l['chain']})" for l in cocrystal_ligands],
            ).ask()
            idx = [f"{l['resname']} (chain {l['chain']})" for l in cocrystal_ligands].index(pick)
            lig = cocrystal_ligands[idx]
        box = pocket.box_from_cocrystal(lig)

    elif "residues" in method:
        raw = questionary.text("Enter residue numbers (comma-separated, e.g. 42,45,78):").ask()
        if raw is None:
            return BACK
        res_ids = [int(r.strip()) for r in raw.split(",") if r.strip().isdigit()]
        box = pocket.box_from_residues(pdb_path, res_ids, chains)

    elif "P2Rank" in method:
        console.print("  [dim]Running P2Rank...[/dim]")
        pockets = pocket.run_p2rank(
            pdb_path, ctx["work_dir"] / "p2rank", alphafold=is_af
        )
        if not pockets:
            console.print("[yellow]P2Rank returned no pockets — falling back to blind.[/yellow]")
            box = pocket.box_blind(pdb_path)
        else:
            pocket_choices = [
                f"#{p['rank']}  score {p['score']:.2f}  prob {p['probability']:.2f}"
                for p in pockets
            ] + ["← Back"]
            pick = questionary.select("Select pocket (never take top-1 blindly):", choices=pocket_choices).ask()
            if pick is None or pick == "← Back":
                return BACK
            selected = pockets[int(pick[1]) - 1]
            box = {**selected}

    else:  # Blind
        console.print(Panel(
            "[yellow]Blind docking uses the entire protein — slow, noisy, "
            "and misses buried pockets.[/yellow]",
            title="[bold yellow]⚠  Blind Docking[/bold yellow]",
        ))
        confirmed = questionary.confirm("Continue with blind docking?", default=False).ask()
        if not confirmed:
            return _step_binding_site(ctx)
        box = pocket.box_blind(pdb_path)

    volume_warnings = pocket.validate_box(box)
    for w in volume_warnings:
        console.print(f"  [{'red' if w['severity'] == 'high' else 'yellow'}]⚠  {w['message']}[/]")

    console.print(
        f"  Box: center {box['center']}  size {box['size']}  "
        f"volume {box.get('volume_angstrom3', 0):.0f} Å³"
    )
    ctx["box"] = box
    return ctx


# ---------------------------------------------------------------------------
# Step 5 — Ligand library
# ---------------------------------------------------------------------------

def _step_ligands(ctx: dict) -> dict | object:
    choice = questionary.select(
        "Ligand library source:",
        choices=[
            "Local file or folder (SDF / SMILES)",
            "ZINC druglike — not available in v1",
            "ChEMBL — not available in v1",
            "← Back",
        ],
    ).ask()

    if choice is None or choice == "← Back":
        return BACK
    if "not available" in choice:
        console.print("[dim]Library downloads are planned for v2. Use a local file for now.[/dim]")
        return _step_ligands(ctx)

    raw = questionary.text(
        "Path to SDF file or folder:",
        default=ctx.get("ligand_raw", ""),
    ).ask()
    if raw is None:
        return BACK

    path = Path(raw.strip()).expanduser()
    if not path.exists():
        console.print(f"[red]Not found: {path}[/red]")
        return _step_ligands(ctx)

    ctx["ligand_raw"]  = raw
    ctx["ligand_path"] = path
    return ctx


# ---------------------------------------------------------------------------
# Step 6 — ADMET pre-filter
# ---------------------------------------------------------------------------

def _step_admet(ctx: dict) -> dict | object:
    cfg = config.load()
    default = cfg["run"].get("admet_pre_filter", True)

    choice = questionary.select(
        "ADMET pre-filter?",
        choices=[
            "Yes — remove obvious failures before docking (recommended)",
            "No — dock everything",
            "← Back",
        ],
        default="Yes — remove obvious failures before docking (recommended)" if default else "No — dock everything",
    ).ask()

    if choice is None or choice == "← Back":
        return BACK

    ctx["admet_pre_filter"] = choice.startswith("Yes")
    return ctx


# ---------------------------------------------------------------------------
# Step 7 — Search depth
# ---------------------------------------------------------------------------

def _step_search_depth(ctx: dict) -> dict | object:
    choices = list(_PRESETS.keys()) + ["← Back"]
    pick = questionary.select("Search depth:", choices=choices).ask()

    if pick is None or pick == "← Back":
        return BACK

    if _PRESETS[pick] is not None:
        params = dict(_PRESETS[pick])
    else:
        # Expert mode
        def _ask_int(prompt, default):
            v = questionary.text(prompt, default=str(default)).ask()
            return int(v) if v and v.isdigit() else default

        mode_pick = questionary.select(
            "Base search mode:", choices=["fast", "balance", "detail"]
        ).ask() or "balance"
        params = {"search_mode": mode_pick}

    ctx["search_params"] = params
    ctx["search_label"]  = pick.split("·")[0].strip()
    return ctx


# ---------------------------------------------------------------------------
# Step 8 — Confirmation panel
# ---------------------------------------------------------------------------

def _step_confirm(ctx: dict) -> dict | object:
    p = ctx["search_params"]
    box = ctx["box"]

    summary = Text()
    summary.append("Receptor     ", style="bold dim")
    summary.append(f"{ctx.get('pdb_id') or ctx['pdb_path'].name}  chains {', '.join(ctx['selected_chains'])}\n")
    summary.append("Binding site ", style="bold dim")
    summary.append(f"{box.get('method', '')}  {box['center']}  {box['size']} Å\n")
    summary.append("Ligands      ", style="bold dim")
    summary.append(f"{ctx['ligand_path']}\n")
    summary.append("ADMET filter ", style="bold dim")
    summary.append(f"{'yes' if ctx['admet_pre_filter'] else 'no'}\n")
    summary.append("Search depth ", style="bold dim")
    summary.append(
        f"{ctx['search_label']}  "
        f"mode {p.get('search_mode') or 'expert'}"
        + (f" · exh {p['exhaustiveness']} · modes {p['num_modes']}" if 'exhaustiveness' in p else "")
        + "\n"
    )

    if ctx.get("is_alphafold"):
        summary.append("\n⚠  AlphaFold structure — P2Rank AF profile active", style="bold yellow")

    console.print(Panel(summary, title="[bold]Ready to submit[/bold]"))

    choice = questionary.select(
        "What would you like to do?",
        choices=["Submit job to Kaggle", "← Change something", "Abort"],
    ).ask()

    if choice is None or "Change" in choice:
        return BACK
    if "Abort" in choice:
        raise SystemExit(0)

    return ctx


# ---------------------------------------------------------------------------
# Steps 9–11 — Prep + Submit + Poll + Post-completion
# ---------------------------------------------------------------------------

def _run_prep_and_submit(ctx: dict) -> None:
    run_id   = ctx["run_id"]
    work_dir = ctx["work_dir"]
    cfg      = config.load()
    try:
        import json
        with open(auth.load_credentials().get("kaggle_json_path", "")) as f:
            kaggle_username = json.load(f).get("username", "user")
    except Exception:
        kaggle_username = "user"

    # Receptor prep
    console.print("\n[bold]Prepping receptor...[/bold]")
    rec_result = rec_prep.prep_receptor(
        pdb_path=ctx["pdb_path"],
        chains=ctx["selected_chains"],
        output_dir=work_dir / "receptor",
        ph=cfg["run"].get("default_ph", 7.4),
    )
    receptor_pdbqt = rec_result["pdbqt_path"]

    # ADMET filter
    ligand_input = ctx["ligand_path"]
    if ctx["admet_pre_filter"]:
        console.print("[bold]Running ADMET pre-filter...[/bold]")
        admet_out = work_dir / "admet_filtered.sdf"
        admet_summary = filter_library(str(ligand_input), str(admet_out))
        if admet_summary["total_input"] == 0:
            from ezscreen.errors import LigandPrepError
            raise LigandPrepError(
                "ADMET filter could not parse any molecules from the input file — "
                "check that the file is valid SDF or SMILES."
            )
        if admet_summary["admet_removed"] == admet_summary["total_input"]:
            from ezscreen.errors import LigandPrepError
            raise LigandPrepError(
                f"ADMET filter removed all {admet_summary['total_input']} molecules — "
                "no compounds passed. Relax filter settings or check your library."
            )
        ligand_input = admet_out

    # Ligand prep
    console.print("[bold]Prepping ligands...[/bold]")
    lig_result = ligand_prep.prep_ligands(
        input_path=ligand_input,
        output_dir=work_dir / "shards",
        ph=cfg["run"].get("default_ph", 7.4),
    )
    shard_paths = lig_result["shard_paths"]
    if not shard_paths:
        from ezscreen.errors import LigandPrepError
        report = lig_result["report"]
        raise LigandPrepError(
            f"Ligand prep produced 0 dockable compounds "
            f"({report['prep_failed']} failed: {report['prep_failures']}). "
            "Check your library or lower ADMET filter stringency."
        )

    # Checkpoint
    checkpoint.init_db()
    checkpoint.create_run(run_id, ctx, lig_result["report"]["total_input"])
    for i, sp in enumerate(shard_paths):
        checkpoint.add_shard(run_id, i, lig_result["report"]["total_input"] // len(shard_paths))

    # Render notebook
    import jinja2
    template_path = Path(__file__).parent.parent / "backends" / "kaggle" / "templates" / "vina_shard.ipynb.j2"
    env = jinja2.Environment(
        variable_start_string="<<", variable_end_string=">>",
        block_start_string="<%",    block_end_string="%>",
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
    )
    tmpl = env.get_template(template_path.name)

    from ezscreen import __version__
    box = ctx["box"]
    notebook_src = tmpl.render(
        ezscreen_version=__version__,
        run_id=run_id,
        engine="unidock",
        mode="hybrid",
        box_center=box["center"],
        box_size=box["size"],
        shard_index=0,
        total_shards=len(shard_paths),
        ph=config.get("run.default_ph"),
        search_mode=ctx["search_params"].get("search_mode", "balance"),
        enumerate_tautomers=False,
        shard_filename=shard_paths[0].name,
    )
    notebook_path = work_dir / "notebook.ipynb"
    notebook_path.write_text(notebook_src, encoding="utf-8")

    # Submit
    console.print("[bold]Submitting to Kaggle...[/bold]")
    result = kaggle_runner.run_screening_job(
        run_id=run_id,
        receptor_pdbqt=receptor_pdbqt,
        shard_paths=shard_paths,
        notebook_path=notebook_path,
        username=kaggle_username,
        work_dir=work_dir,
        retry_limit=cfg["run"].get("shard_retry_limit", 3),
    )

    if result["status"] == "complete":
        checkpoint.mark_run_complete(run_id)
        console.print(f"\n[bold green]✓ Run complete![/bold green]  Results → {result['output_dir']}")
        _post_completion(run_id, Path(result["output_dir"]), ctx)
    else:
        checkpoint.mark_run_failed(run_id)
        console.print(f"\n[red]✗ Run {result['status']}: {result.get('error_type', '')}[/red]")
        console.print(f"  Resume with: [bold]ezscreen resume {run_id}[/bold]")


def _post_completion(_run_id: str, output_dir: Path, ctx: dict) -> None:
    choice = questionary.select(
        "What next?",
        choices=[
            "View results in terminal + 3D viewer",
            "Run ADMET on results",
            "Stage 2 validation (DiffDock-L via NIM)",
            "Exit",
        ],
    ).ask()

    if choice is None or choice == "Exit":
        return
    if "View" in choice:
        from ezscreen.commands import view
        view.invoke(output_dir)
    elif "ADMET" in choice:
        from ezscreen.commands import admet
        admet.invoke(output_dir / "poses.sdf", output_dir / "poses_admet.sdf")
    elif "validation" in choice:
        from ezscreen.commands import validate
        validate.invoke(
            receptor_path=ctx["pdb_path"],
            hits_path=output_dir / "poses.sdf",
            output_dir=output_dir,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def invoke() -> None:
    """Full interactive virtual screening decision tree."""
    if not auth.has_kaggle_credentials():
        console.print("[yellow]No Kaggle credentials found.[/yellow]")
        setup = questionary.confirm("Set up credentials now?", default=True).ask()
        if setup:
            auth.run_wizard()
        if not auth.has_kaggle_credentials():
            console.print("[red]✗ Kaggle credentials required. Run ezscreen auth.[/red]")
            return

    run_id   = _run_id()
    work_dir = _work_dir(run_id)

    ctx = make_context()
    ctx["run_id"]   = run_id
    ctx["work_dir"] = work_dir

    steps = [
        _step_receptor,
        _step_chains,
        _step_alphafold_warn,
        _step_binding_site,
        _step_ligands,
        _step_admet,
        _step_search_depth,
        _step_confirm,
    ]

    cursor = 0
    while cursor < len(steps):
        try:
            result = steps[cursor](ctx)
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            return

        if result is BACK:
            cursor = max(0, cursor - 1)
        else:
            ctx    = result
            cursor += 1

    _run_prep_and_submit(ctx)
