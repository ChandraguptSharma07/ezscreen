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
    console.print("  [dim]Waiting for GNINA dataset to be ready...[/dim]", end="")
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


def _upload_gnina_dataset(
    run_id: str,
    receptor_pdb: Path,
    poses_sdf: Path,
    scores_csv: Path,
    username: str,
    work_dir: Path,
) -> str:
    import kaggle
    kaggle.api.authenticate()

    slug = f"ezscreen-gnina-{run_id}"
    dataset_dir = work_dir / f"gnina_dataset_{run_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(receptor_pdb, dataset_dir / "receptor_prep.pdb")
    shutil.copy2(poses_sdf,    dataset_dir / "gnina_poses.sdf")
    shutil.copy2(scores_csv,   dataset_dir / "scores_top_n.csv")

    meta = {
        "title": f"ezscreen-gnina {run_id}",
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
                raise RuntimeError(f"GNINA dataset update failed: {exc2}") from exc2
        else:
            raise RuntimeError(f"GNINA dataset upload failed: {exc}") from exc

    return f"{username}/{slug}"


def _slugify_dataset(dataset_ref: str) -> str:
    return dataset_ref.split("/")[-1]


def _render_gnina_notebook(
    run_id: str,
    top_n: int,
    dataset_slug: str,
    template_dir: Path,
    out_path: Path,
) -> None:
    import jinja2

    from ezscreen import __version__

    template_path = template_dir / "gnina_rescore.ipynb.j2"
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


def _download_gnina_output(kernel_ref: str, work_dir: Path) -> Path:
    import kaggle
    kaggle.api.authenticate()
    out_dir = work_dir / "gnina_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        kaggle.api.kernels_output(kernel_ref, path=str(out_dir))
    except Exception as exc:
        raise RuntimeError(f"Failed to download GNINA output: {exc}") from exc
    return out_dir


def run_gnina_rescore(
    run_id: str, work_dir: Path, accelerator: str = "nvidiaTeslaP100"
) -> dict[str, Any]:
    """CNN-rescore the top-N existing poses on a Kaggle GPU kernel, no re-docking.

    Mirrors plip_runner: upload receptor + poses, push a GPU kernel running
    `gnina --score_only`, poll, download cnn_scores.csv into the run output so the
    next merge left-joins CNNscore/CNNaffinity. Returns dict with keys:
    status, cnn_scores_path|None, error|None.
    """
    output_dir = work_dir / "output"
    cfg = config.load()
    top_n = int(cfg.get("results", {}).get("interaction_top_n", 20))

    # Locate receptor PDB (resume.json → receptor/receptor_prep.pdb fallback)
    resume_json = work_dir / "resume.json"
    receptor_pdb: Path | None = None
    username = ""
    if resume_json.exists():
        info = json.loads(resume_json.read_text())
        if info.get("receptor_pdb"):
            receptor_pdb = Path(info["receptor_pdb"])
        username = info.get("username", "")

    if receptor_pdb is None or not receptor_pdb.exists():
        candidate = work_dir / "receptor" / "receptor_prep.pdb"
        if candidate.exists():
            receptor_pdb = candidate

    if receptor_pdb is None or not receptor_pdb.exists():
        return {
            "status": "failed",
            "cnn_scores_path": None,
            "error": "receptor_prep.pdb not found — this run predates v1.9.0",
        }

    scores_csv = output_dir / "scores.csv"
    poses_sdf  = output_dir / "poses.sdf"

    if not scores_csv.exists():
        return {"status": "failed", "cnn_scores_path": None, "error": "scores.csv not found"}
    if not poses_sdf.exists():
        return {"status": "failed", "cnn_scores_path": None, "error": "poses.sdf not found — run at least one Kaggle shard first"}
    if poses_sdf.stat().st_size == 0:
        return {
            "status": "failed",
            "cnn_scores_path": None,
            "error": "poses.sdf is empty — the docking kernel wrote no 3D poses to rescore",
        }

    # Slice top-N by score
    with scores_csv.open() as f:
        all_rows = list(csv.DictReader(f))
    top_rows = all_rows[:top_n]
    top_ids  = {r["ligand"] for r in top_rows}

    gnina_dir = work_dir / "gnina_prep"
    gnina_dir.mkdir(parents=True, exist_ok=True)

    scores_top_n = gnina_dir / "scores_top_n.csv"
    with scores_top_n.open("w", newline="") as f:
        if top_rows:
            writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_rows)

    # Extract matching poses from poses.sdf
    from rdkit import Chem
    gnina_poses = gnina_dir / "gnina_poses.sdf"
    try:
        supplier = Chem.SDMolSupplier(str(poses_sdf), removeHs=False)
    except OSError as exc:
        return {"status": "failed", "cnn_scores_path": None, "error": f"poses.sdf is unreadable: {exc}"}
    writer_sdf = Chem.SDWriter(str(gnina_poses))
    written    = 0
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if name in top_ids:
            writer_sdf.write(mol)
            written += 1
    writer_sdf.close()
    console.print(f"  [dim]{written}/{len(top_ids)} poses extracted for GNINA[/dim]")

    if written == 0:
        return {"status": "failed", "cnn_scores_path": None, "error": "no top-N poses found in poses.sdf"}

    # Resolve Kaggle username — env var then credentials file
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
        return {"status": "failed", "cnn_scores_path": None, "error": "Kaggle username not found — run auth setup"}

    console.print("  [dim]Uploading GNINA dataset...[/dim]")
    dataset_ref = _upload_gnina_dataset(
        run_id=run_id,
        receptor_pdb=receptor_pdb,
        poses_sdf=gnina_poses,
        scores_csv=scores_top_n,
        username=username,
        work_dir=work_dir,
    )
    console.print(f"  [dim]GNINA dataset: {dataset_ref}[/dim]")

    _wait_for_dataset(dataset_ref)
    console.print("  [dim]Waiting 90s for dataset files to sync...[/dim]")
    time.sleep(90)

    template_dir = Path(__file__).parent / "templates"
    gnina_nb = gnina_dir / "gnina_rescore.ipynb"
    _render_gnina_notebook(
        run_id=run_id,
        top_n=top_n,
        dataset_slug=_slugify_dataset(dataset_ref),
        template_dir=template_dir,
        out_path=gnina_nb,
    )

    gnina_run_id = f"{run_id}-gnina"
    console.print("  [dim]Submitting GNINA notebook to Kaggle (GPU)...[/dim]")
    kernel_ref = push_kernel(
        run_id=gnina_run_id,
        notebook_path=gnina_nb,
        dataset_ref=dataset_ref,
        username=username,
        work_dir=work_dir,
        accelerator=accelerator,
    )
    console.print(f"  [dim]GNINA kernel: {kernel_ref}[/dim]")

    result = poll_until_done(kernel_ref=kernel_ref, run_id=gnina_run_id)

    if result["status"] != "complete":
        return {
            "status": result["status"],
            "cnn_scores_path": None,
            "error": f"GNINA kernel {result['status']} ({result.get('error_type')})",
        }

    out_dir = _download_gnina_output(kernel_ref, work_dir)

    found = list(out_dir.rglob("cnn_scores.csv"))
    if not found:
        return {"status": "failed", "cnn_scores_path": None, "error": "cnn_scores.csv not in kernel output"}

    dest = output_dir / "cnn_scores.csv"
    shutil.copy2(found[0], dest)
    console.print(f"  [green]CNN scores saved → {dest.name}[/green]")
    return {"status": "complete", "cnn_scores_path": dest, "error": None}
