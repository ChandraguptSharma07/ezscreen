from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from ezscreen.backends.kaggle.kernel import push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done
from ezscreen.results.run_layout import resolve_output_dir

console = Console()
_DATASET_POLL_INTERVAL = 10
_DATASET_TIMEOUT = 300

MD_PRESETS: dict[str, dict[str, Any]] = {
    "setup": {
        "label": "Setup check",
        "steps": 0,
        "timestep_fs": 2.0,
        "report_interval": 1,
        "description": "parameterize, solvate, minimize; no dynamics",
    },
    "smoke": {
        "label": "Smoke",
        "steps": 5_000,
        "timestep_fs": 2.0,
        "report_interval": 250,
        "description": "10 ps sanity check",
    },
    "quick": {
        "label": "Quick",
        "steps": 50_000,
        "timestep_fs": 2.0,
        "report_interval": 1_000,
        "description": "100 ps stability check",
    },
    "standard": {
        "label": "Standard",
        "steps": 500_000,
        "timestep_fs": 2.0,
        "report_interval": 5_000,
        "description": "1 ns production run",
    },
}


def _json_source_literal(value: str) -> str:
    return json.dumps(json.dumps(value))[1:-1]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-") or "ligand"


def _first_ligand_id(scores_csv: Path) -> str | None:
    with scores_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[0].get("ligand") or rows[0].get("name")


def _ligand_exists(scores_csv: Path, ligand_id: str) -> bool:
    with scores_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if ligand_id in {row.get("ligand", ""), row.get("name", "")}:
                return True
    return False


def _find_pose_sdf(run_dir: Path, pose_source: str = "unidock") -> Path | None:
    candidates = [resolve_output_dir(run_dir) / "poses.sdf"]
    if pose_source == "gnina":
        candidates.append(run_dir / "consensus_gnina" / "output" / "poses.sdf")
        candidates.append(run_dir / "consensus_unidock" / "output" / "poses.sdf")
    else:
        candidates.append(run_dir / "consensus_unidock" / "output" / "poses.sdf")
        candidates.append(run_dir / "consensus_gnina" / "output" / "poses.sdf")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _kaggle_username(run_dir: Path) -> str:
    resume_json = run_dir / "resume.json"
    if resume_json.exists():
        try:
            username = json.loads(resume_json.read_text()).get("username", "")
            if username:
                return username
        except Exception:
            pass
    try:
        from ezscreen import auth

        creds = auth.load_credentials()
        kj = auth.get_kaggle_json_path(creds)
        if kj and kj.exists():
            return json.loads(kj.read_text()).get("username", "")
    except Exception:
        pass
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if cred_path.exists():
        try:
            return json.loads(cred_path.read_text()).get("username", "")
        except Exception:
            pass
    return ""


def _resolve_kaggle_credentials(kaggle_json_path: Path | None = None) -> tuple[str, Path | None]:
    try:
        from ezscreen import auth

        path = auth.get_kaggle_json_path()
        if kaggle_json_path:
            path = Path(kaggle_json_path).expanduser()
        if path and path.exists():
            data = json.loads(path.read_text())
            os.environ["KAGGLE_USERNAME"] = data["username"]
            os.environ["KAGGLE_KEY"] = data["key"]
            return data["username"], path
    except Exception:
        pass
    return "", None


def _wait_for_dataset(dataset_ref: str) -> None:
    import kaggle

    kaggle.api.authenticate()
    elapsed = 0
    console.print("  [dim]Waiting for MD dataset to be ready...[/dim]", end="")
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
    console.print(f" [yellow]timed out after {elapsed}s - proceeding[/yellow]")


def _upload_md_dataset(
    md_run_id: str,
    receptor_pdb: Path,
    poses_sdf: Path,
    scores_csv: Path,
    username: str,
    work_dir: Path,
) -> str:
    import kaggle

    kaggle.api.authenticate()
    dataset_dir = work_dir / f"dataset_{md_run_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(receptor_pdb, dataset_dir / "receptor_prep.pdb")
    shutil.copy2(poses_sdf, dataset_dir / "poses.sdf")
    shutil.copy2(scores_csv, dataset_dir / "scores.csv")

    slug = f"ezscreen-{md_run_id}"
    meta = {
        "title": f"ezscreen {md_run_id}",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        kaggle.api.dataset_create_new(str(dataset_dir), public=False, quiet=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("already", "exists", "409", "403")):
            kaggle.api.dataset_create_version(
                str(dataset_dir), version_notes="ezscreen md retry", quiet=True
            )
        else:
            raise RuntimeError(f"MD dataset upload failed: {exc}") from exc
    return f"{username}/{slug}"


def render_md_notebook(
    run_id: str,
    ligand_id: str,
    preset: str = "smoke",
    pose_source: str = "unidock",
    temperature_k: float = 300.0,
    padding_nm: float = 1.0,
) -> str:
    import jinja2

    preset_cfg = MD_PRESETS.get(preset, MD_PRESETS["smoke"])
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        variable_start_string="<<",
        variable_end_string=">>",
        block_start_string="<%",
        block_end_string="%>",
        loader=jinja2.FileSystemLoader(str(template_dir)),
    )
    return env.get_template("md_openmm.ipynb.j2").render(
        run_id_source=_json_source_literal(run_id),
        ligand_id_source=_json_source_literal(ligand_id),
        preset_source=_json_source_literal(preset),
        pose_source_source=_json_source_literal(pose_source),
        steps=int(preset_cfg["steps"]),
        timestep_fs=float(preset_cfg["timestep_fs"]),
        report_interval=int(preset_cfg["report_interval"]),
        temperature_k=float(temperature_k),
        padding_nm=float(padding_nm),
    )


def _download_md_output(kernel_ref: str, work_dir: Path) -> Path:
    import kaggle

    kaggle.api.authenticate()
    out_dir = work_dir / "kaggle_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        kaggle.api.kernels_output(kernel_ref, path=str(out_dir))
    except UnicodeEncodeError:
        if list(out_dir.rglob("md_summary.json")):
            return out_dir
        raise
    return out_dir


def run_md_simulation(
    run_id: str,
    ligand_id: str | None = None,
    preset: str = "smoke",
    pose_source: str = "unidock",
    accelerator: str = "nvidiaTeslaT4",
    run_dir: Path | None = None,
    kaggle_json_path: Path | None = None,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir or (Path.home() / ".ezscreen" / "runs" / run_id)
    if not run_dir.exists():
        return {"status": "failed", "output_dir": None, "error": f"run not found: {run_id}"}

    receptor_pdb = run_dir / "receptor" / "receptor_prep.pdb"
    scores_csv = resolve_output_dir(run_dir) / "scores.csv"
    poses_sdf = _find_pose_sdf(run_dir, pose_source=pose_source)
    if not receptor_pdb.exists():
        return {"status": "failed", "output_dir": None, "error": "receptor_prep.pdb not found"}
    if not scores_csv.exists():
        return {"status": "failed", "output_dir": None, "error": "scores.csv not found"}
    if poses_sdf is None:
        return {"status": "failed", "output_dir": None, "error": "poses.sdf not found"}
    if progress_cb:
        progress_cb("resolved inputs", 0.10)

    ligand_id = ligand_id or _first_ligand_id(scores_csv)
    if not ligand_id:
        return {"status": "failed", "output_dir": None, "error": "no ligand rows in scores.csv"}
    if not _ligand_exists(scores_csv, ligand_id):
        return {"status": "failed", "output_dir": None, "error": f"ligand not found in scores.csv: {ligand_id}"}

    username, account_path = _resolve_kaggle_credentials(kaggle_json_path)
    if not username:
        username = _kaggle_username(run_dir)
    if not username:
        return {"status": "failed", "output_dir": None, "error": "Kaggle username not found"}
    if progress_cb:
        progress_cb(f"using Kaggle account: {username}", 0.20)

    safe_lig = _safe_slug(ligand_id)
    md_run_id = f"{run_id}-md-{safe_lig}"
    work_dir = run_dir / "md" / safe_lig
    work_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"  [dim]Uploading MD dataset for {ligand_id}...[/dim]")
    if progress_cb:
        progress_cb("uploading dataset", 0.25)
    dataset_ref = _upload_md_dataset(
        md_run_id=md_run_id,
        receptor_pdb=receptor_pdb,
        poses_sdf=poses_sdf,
        scores_csv=scores_csv,
        username=username,
        work_dir=work_dir,
    )
    _wait_for_dataset(dataset_ref)
    if progress_cb:
        progress_cb("dataset ready", 0.40)
    console.print("  [dim]Waiting 60s for dataset files to sync...[/dim]")
    time.sleep(60)

    notebook_path = work_dir / "md_openmm.ipynb"
    notebook_path.write_text(
        render_md_notebook(
            run_id=run_id,
            ligand_id=ligand_id,
            preset=preset,
            pose_source=pose_source,
        ),
        encoding="utf-8",
    )
    if progress_cb:
        progress_cb("rendered notebook", 0.55)

    kernel_ref = push_kernel(
        run_id=md_run_id,
        notebook_path=notebook_path,
        dataset_ref=dataset_ref,
        username=username,
        work_dir=work_dir,
        accelerator=accelerator,
    )
    if progress_cb:
        progress_cb("submitted kernel", 0.70)
    result = poll_until_done(kernel_ref=kernel_ref, run_id=md_run_id)
    if result["status"] != "complete":
        return {
            "status": result["status"],
            "output_dir": None,
            "error": f"MD kernel {result['status']} ({result.get('error_type')})",
        }

    if progress_cb:
        progress_cb("kernel complete", 0.90)
    out_dir = _download_md_output(kernel_ref, work_dir)
    found = list(out_dir.rglob("md_summary.json"))
    if not found:
        return {"status": "failed", "output_dir": out_dir, "error": "md_summary.json not found"}

    final_dir = work_dir / "output"
    final_dir.mkdir(exist_ok=True)
    for path in found[0].parent.iterdir():
        if path.is_file():
            shutil.copy2(path, final_dir / path.name)
    if progress_cb:
        progress_cb("downloaded results", 1.0)
    return {"status": "complete", "output_dir": final_dir, "error": None}


