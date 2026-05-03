from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from ezscreen import config
from ezscreen.backends.kaggle.kernel import push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done

console = Console()
_DATASET_POLL_INTERVAL = 10
_DATASET_TIMEOUT = 300


def _wait_for_dataset(dataset_ref: str) -> None:
    import kaggle
    kaggle.api.authenticate()
    elapsed = 0
    console.print("  [dim]Waiting for PLIP dataset to be ready...[/dim]", end="")
    while elapsed < _DATASET_TIMEOUT:
        try:
            if kaggle.api.dataset_status(dataset_ref) == "ready":
                console.print(f" ready ({elapsed}s)")
                return
        except Exception:
            pass
        time.sleep(_DATASET_POLL_INTERVAL)
        elapsed += _DATASET_POLL_INTERVAL
        console.print(".", end="")
    console.print(f" [yellow]timed out after {elapsed}s — proceeding[/yellow]")


def _upload_plip_dataset(
    run_id: str,
    receptor_pdb: Path,
    poses_sdf: Path,
    scores_csv: Path,
    username: str,
    work_dir: Path,
) -> str:
    import kaggle
    kaggle.api.authenticate()

    slug = f"ezscreen-plip-{run_id}"
    dataset_dir = work_dir / f"plip_dataset_{run_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(receptor_pdb, dataset_dir / "receptor_prep.pdb")
    shutil.copy2(poses_sdf,    dataset_dir / "plip_poses.sdf")
    shutil.copy2(scores_csv,   dataset_dir / "scores_top_n.csv")

    meta = {
        "title": f"ezscreen-plip {run_id}",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        kaggle.api.dataset_create_new(str(dataset_dir), public=False, quiet=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("already", "exists", "409", "403")):
            try:
                kaggle.api.dataset_create_version(
                    str(dataset_dir), version_notes="ezscreen retry", quiet=True
                )
            except Exception as exc2:
                raise RuntimeError(f"PLIP dataset update failed: {exc2}") from exc2
        else:
            raise RuntimeError(f"PLIP dataset upload failed: {exc}") from exc

    return f"{username}/{slug}"


def _slugify_dataset(dataset_ref: str) -> str:
    return dataset_ref.split("/")[-1]


def _render_plip_notebook(
    run_id: str,
    top_n: int,
    dataset_slug: str,
    template_dir: Path,
    out_path: Path,
) -> None:
    import jinja2

    from ezscreen import __version__

    template_path = template_dir / "plip_analysis.ipynb.j2"
    env = jinja2.Environment(
        variable_start_string="<<",
        variable_end_string=">>",
        block_start_string="<%",
        block_end_string="%>",
        loader=jinja2.FileSystemLoader(str(template_dir)),
    )
    rendered = env.get_template(template_path.name).render(
        ezscreen_version=__version__,
        run_id=run_id,
        top_n=top_n,
        dataset_slug=dataset_slug,
    )
    out_path.write_text(rendered)


def _download_plip_output(kernel_ref: str, work_dir: Path) -> Path:
    import kaggle
    kaggle.api.authenticate()
    out_dir = work_dir / "plip_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        kaggle.api.kernels_output(kernel_ref, path=str(out_dir))
    except Exception as exc:
        raise RuntimeError(f"Failed to download PLIP output: {exc}") from exc
    return out_dir


def run_plip_analysis(run_id: str, work_dir: Path) -> dict[str, Any]:
    """Upload top-N poses + receptor, run PLIP on Kaggle CPU, download interactions.json.

    Returns dict with keys: status, interactions_path|None, error|None.
    """
    output_dir = work_dir / "output"
    cfg = config.load()
    top_n = int(cfg.get("results", {}).get("interaction_top_n", 20))

    # Locate receptor PDB
    resume_json = work_dir / "resume.json"
    receptor_pdb: Path | None = None
    username = ""
    if resume_json.exists():
        info = json.loads(resume_json.read_text())
        if info.get("receptor_pdb"):
            receptor_pdb = Path(info["receptor_pdb"])
        username = info.get("username", "")

    # Fallback: look for receptor_prep.pdb beside the PDBQT
    if receptor_pdb is None or not receptor_pdb.exists():
        candidate = work_dir / "receptor" / "receptor_prep.pdb"
        if candidate.exists():
            receptor_pdb = candidate

    if receptor_pdb is None or not receptor_pdb.exists():
        return {
            "status": "failed",
            "interactions_path": None,
            "error": "receptor_prep.pdb not found — this run predates v1.9.0",
        }

    scores_csv = output_dir / "scores.csv"
    poses_sdf  = output_dir / "poses.sdf"

    if not scores_csv.exists():
        return {"status": "failed", "interactions_path": None, "error": "scores.csv not found"}
    if not poses_sdf.exists():
        return {"status": "failed", "interactions_path": None, "error": "poses.sdf not found — run at least one Kaggle shard first"}

    # Slice top-N
    with scores_csv.open() as f:
        all_rows = list(csv.DictReader(f))
    top_rows = all_rows[:top_n]
    top_ids  = {r["ligand"] for r in top_rows}

    plip_dir = work_dir / "plip_prep"
    plip_dir.mkdir(parents=True, exist_ok=True)

    scores_top_n = plip_dir / "scores_top_n.csv"
    with scores_top_n.open("w", newline="") as f:
        if top_rows:
            writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)

    # Extract matching poses from poses.sdf
    from rdkit import Chem
    plip_poses = plip_dir / "plip_poses.sdf"
    supplier   = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
    writer_sdf = Chem.SDWriter(str(plip_poses))
    written    = 0
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if name in top_ids:
            writer_sdf.write(mol)
            written += 1
    writer_sdf.close()
    console.print(f"  [dim]{written}/{len(top_ids)} poses extracted for PLIP[/dim]")

    # Get Kaggle username — try env var then credentials file
    if not username:
        import os
        username = os.environ.get("KAGGLE_USERNAME", "")
    if not username:
        for cred_path in [
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / ".ezscreen" / "kaggle.json",
        ]:
            if cred_path.exists():
                try:
                    username = json.loads(cred_path.read_text()).get("username", "")
                    if username:
                        break
                except Exception:
                    pass
    if not username:
        return {"status": "failed", "interactions_path": None, "error": "Kaggle username not found — run auth setup"}

    # Upload dataset
    console.print("  [dim]Uploading PLIP dataset...[/dim]")
    dataset_ref = _upload_plip_dataset(
        run_id=run_id,
        receptor_pdb=receptor_pdb,
        poses_sdf=plip_poses,
        scores_csv=scores_top_n,
        username=username,
        work_dir=work_dir,
    )
    console.print(f"  [dim]PLIP dataset: {dataset_ref}[/dim]")

    _wait_for_dataset(dataset_ref)
    console.print("  [dim]Waiting 90s for dataset files to sync...[/dim]")
    time.sleep(90)

    # Render and push notebook
    template_dir = Path(__file__).parent / "templates"
    plip_nb = plip_dir / "plip_analysis.ipynb"
    _render_plip_notebook(
        run_id=run_id,
        top_n=top_n,
        dataset_slug=_slugify_dataset(dataset_ref),
        template_dir=template_dir,
        out_path=plip_nb,
    )

    plip_run_id = f"{run_id}-plip"
    console.print("  [dim]Submitting PLIP notebook to Kaggle (CPU)...[/dim]")
    kernel_ref = push_kernel(
        run_id=plip_run_id,
        notebook_path=plip_nb,
        dataset_ref=dataset_ref,
        username=username,
        work_dir=work_dir,
        accelerator="none",
    )
    console.print(f"  [dim]PLIP kernel: {kernel_ref}[/dim]")

    result = poll_until_done(kernel_ref=kernel_ref, run_id=plip_run_id)

    if result["status"] != "complete":
        return {
            "status": result["status"],
            "interactions_path": None,
            "error": f"PLIP kernel {result['status']} ({result.get('error_type')})",
        }

    out_dir = _download_plip_output(kernel_ref, work_dir)

    # Find interactions.json in downloaded output
    found = list(out_dir.rglob("interactions.json"))
    if not found:
        return {"status": "failed", "interactions_path": None, "error": "interactions.json not in kernel output"}

    dest = output_dir / "interactions_top_n.json"
    shutil.copy2(found[0], dest)
    console.print(f"  [green]Interactions saved → {dest.name}[/green]")
    return {"status": "complete", "interactions_path": dest, "error": None}
