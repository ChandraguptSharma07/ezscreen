from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from ezscreen.backends.kaggle.kernel import push_kernel
from ezscreen.backends.kaggle.poller import poll_until_done
from ezscreen.backends.kaggle.runner import _wait_for_dataset, _wait_for_dataset_sync
from ezscreen.results.run_layout import resolve_output_dir

console = Console()

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


def _load_md_summary(summary_path: Path) -> dict[str, Any]:
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return {}


def _read_state_rows(state_csv: Path) -> list[dict[str, str]]:
    if not state_csv.exists():
        return []
    with state_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


_STATE_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "Temperature (K)"),
    "potentialEnergy": ("potentialEnergy", "Potential Energy (kJ/mole)"),
    "kineticEnergy": ("kineticEnergy", "Kinetic Energy (kJ/mole)"),
    "totalEnergy": ("totalEnergy", "Total Energy (kJ/mole)"),
    "step": ("step", "Step"),
    "time": ("time", "Time (ps)"),
}


def _series_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    aliases = _STATE_HEADER_ALIASES.get(key, (key,))
    for row in rows:
        raw = ""
        for alias in aliases:
            raw = str(row.get(alias, "")).strip()
            if raw:
                break
        if not raw:
            continue
        raw = raw.rstrip("%")
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _line_points(values: list[float], width: int = 360, height: int = 120) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"0,{height // 2} {width},{height // 2}"
    lo = min(values)
    hi = max(values)
    span = hi - lo or 1.0
    pts: list[str] = []
    for idx, value in enumerate(values):
        x = idx * width / max(len(values) - 1, 1)
        y = height - ((value - lo) / span) * height
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _state_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"frames": len(rows)}
    for key in ("temperature", "potentialEnergy", "kineticEnergy", "totalEnergy"):
        vals = _series_values(rows, key)
        if not vals:
            continue
        metrics[key] = {
            "first": vals[0],
            "last": vals[-1],
            "min": min(vals),
            "max": max(vals),
        }
    return metrics


def _write_md_report(final_dir: Path, summary: dict[str, Any]) -> None:
    rows = _read_state_rows(final_dir / "state.csv")
    metrics = _state_metrics(rows)
    charts = [
        ("Temperature", "temperature", "K"),
        ("Potential energy", "potentialEnergy", "kJ/mol"),
        ("Total energy", "totalEnergy", "kJ/mol"),
    ]

    def field(label: str, value: Any) -> str:
        return f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>"

    summary_rows = "".join([
        field("MD run", summary.get("md_run_id", "")),
        field("Screening run", summary.get("run_id", "")),
        field("Ligand", summary.get("ligand_id", "")),
        field("Preset", summary.get("preset", "")),
        field("Pose source", summary.get("pose_source", "")),
        field("Platform", summary.get("platform", "")),
        field("Simulated ps", summary.get("simulated_ps", "")),
        field("Potential energy (kJ/mol)", summary.get("potential_energy_kj_mol", "")),
        field("Kinetic energy (kJ/mol)", summary.get("kinetic_energy_kj_mol", "")),
        field("Deleted overlap waters", summary.get("deleted_overlap_waters", "")),
        field("Protein chain splits", summary.get("protein_chain_splits", "")),
    ])

    metric_cards: list[str] = []
    for label, key, unit in charts:
        vals = _series_values(rows, key)
        if not vals:
            continue
        card = (
            "<section class='chart-card'>"
            f"<h3>{html.escape(label)}</h3>"
            f"<div class='metric-sub'>last {vals[-1]:.3f} {html.escape(unit)} | min {min(vals):.3f} | max {max(vals):.3f}</div>"
            "<svg viewBox='0 0 360 120' preserveAspectRatio='none'>"
            f"<polyline points='{_line_points(vals)}'></polyline>"
            "</svg>"
            "</section>"
        )
        metric_cards.append(card)

    file_items: list[str] = []
    for name in [
        "complex_initial.pdb",
        "solvated.pdb",
        "minimized.pdb",
        "final.pdb",
        "trajectory.dcd",
        "state.csv",
        "ligand_pose.sdf",
        "md_summary.json",
        "md_metrics.json",
    ]:
        path = final_dir / name
        if path.exists():
            file_items.append(f"<li>{html.escape(name)}</li>")

    report = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>ezscreen MD report {html.escape(str(summary.get('md_run_id', '')))}</title>
<style>
body {{ background:#0d1117; color:#f0f6fc; font-family:Segoe UI, sans-serif; margin:0; padding:24px; }}
h1,h2,h3 {{ color:#79c0ff; margin:0 0 12px; }}
.grid {{ display:grid; grid-template-columns: minmax(280px, 420px) 1fr; gap:24px; align-items:start; }}
.panel {{ background:#161b22; border:1px solid #30363d; padding:16px; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; padding:6px 0; border-bottom:1px solid #30363d; }}
th {{ color:#8b949e; width:48%; font-weight:600; }}
.charts {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px; margin-top:24px; }}
.chart-card {{ background:#161b22; border:1px solid #30363d; padding:16px; }}
.metric-sub {{ color:#8b949e; margin-bottom:10px; }}
svg {{ width:100%; height:160px; background:#0d1117; border:1px solid #30363d; }}
polyline {{ fill:none; stroke:#3fb950; stroke-width:2; }}
ul {{ margin:0; padding-left:18px; color:#f0f6fc; }}
pre {{ white-space:pre-wrap; word-break:break-word; }}
</style>
</head>
<body>
<h1>ezscreen MD report</h1>
<div class='grid'>
  <section class='panel'>
    <h2>Summary</h2>
    <table>{summary_rows}</table>
  </section>
  <section class='panel'>
    <h2>Artifacts</h2>
    <ul>{''.join(file_items)}</ul>
    <h3 style='margin-top:16px'>Raw summary</h3>
    <pre>{html.escape(json.dumps(summary, indent=2))}</pre>
  </section>
</div>
<section class='charts'>
  {''.join(metric_cards) if metric_cards else "<div class='panel'><h2>No state.csv time-series was produced for this run.</h2></div>"}
</section>
</body>
</html>
"""
    (final_dir / "md_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (final_dir / "md_report.html").write_text(report, encoding="utf-8")


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
    final_dir = work_dir / "output"
    work_dir.mkdir(parents=True, exist_ok=True)

    from ezscreen import checkpoint

    checkpoint.init_db()
    if checkpoint.get_run(run_id) is None:
        checkpoint.create_run(run_id, {}, 0)
        checkpoint.mark_run_complete(run_id)
    checkpoint.create_md_run(
        md_run_id=md_run_id,
        run_id=run_id,
        ligand_id=ligand_id,
        preset=preset,
        pose_source=pose_source,
        accelerator=accelerator,
        output_dir=final_dir,
    )

    try:
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
        _wait_for_dataset_sync(dataset_ref, label="MD dataset files")
        if progress_cb:
            progress_cb("dataset ready", 0.40)

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
            error = f"MD kernel {result['status']} ({result.get('error_type')})"
            checkpoint.mark_md_run_failed(md_run_id, error, output_dir=final_dir)
            return {
                "status": result["status"],
                "output_dir": None,
                "error": error,
                "md_run_id": md_run_id,
            }

        if progress_cb:
            progress_cb("kernel complete", 0.90)
        out_dir = _download_md_output(kernel_ref, work_dir)
        found = list(out_dir.rglob("md_summary.json"))
        if not found:
            checkpoint.mark_md_run_failed(md_run_id, "md_summary.json not found", output_dir=out_dir)
            return {"status": "failed", "output_dir": out_dir, "error": "md_summary.json not found", "md_run_id": md_run_id}

        final_dir.mkdir(exist_ok=True)
        for path in found[0].parent.iterdir():
            if path.is_file():
                shutil.copy2(path, final_dir / path.name)

        if preset != "setup" and not (final_dir / "state.csv").exists():
            checkpoint.mark_md_run_failed(md_run_id, "state.csv not found for MD run", output_dir=final_dir)
            return {"status": "failed", "output_dir": final_dir, "error": "state.csv not found for MD run", "md_run_id": md_run_id}

        summary = _load_md_summary(final_dir / "md_summary.json")
        summary["md_run_id"] = md_run_id
        summary["run_id"] = run_id
        summary["ligand_id"] = ligand_id
        summary["preset"] = preset
        summary["pose_source"] = pose_source
        summary["accelerator"] = accelerator
        (final_dir / "md_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_md_report(final_dir, summary)
        checkpoint.mark_md_run_complete(md_run_id, final_dir, summary=summary)
        if progress_cb:
            progress_cb("downloaded results", 1.0)
        return {"status": "complete", "output_dir": final_dir, "error": None, "md_run_id": md_run_id}
    except Exception as exc:
        checkpoint.mark_md_run_failed(md_run_id, str(exc), output_dir=final_dir)
        raise

