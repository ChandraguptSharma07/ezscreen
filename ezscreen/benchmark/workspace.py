from __future__ import annotations

import json
import shutil
from pathlib import Path

_DEFAULT_METRICS_SUMMARY = {
    "metric_contract_version": 2,
    'ef20': None,
    'bedroc': None,
    'bedroc_alpha': 20.0,
    'coverage': None,
    'first_active_rank': None,
    "status": "pending",
    "generated_at": "",
    "actives_path": "",
    "scores_csv": "",
    "report_path": "",
    "ef1": None,
    "ef5": None,
    "auc_roc": None,
    "redock_rmsd": None,
    "n_actives": 0,
    "n_decoys": 0,
    "total_screened": 0,
    "label_diagnostics": {},
}


def benchmark_root() -> Path:
    return Path.cwd() / "benchmarks"


def templates_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _copy_template(src_name: str, dest: Path) -> Path:
    src = templates_dir() / src_name
    if not src.exists():
        raise FileNotFoundError(f"Benchmark template missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(src, dest)
    return dest


def _write_json_if_missing(dest: Path, payload: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def create_campaign_scaffold(benchmark_id: str, *, root: Path | None = None) -> Path:
    benchmark_id = benchmark_id.strip()
    if not benchmark_id:
        raise ValueError("benchmark_id is required")

    base = (root or benchmark_root()) / benchmark_id
    (base / "inputs" / "actives").mkdir(parents=True, exist_ok=True)
    (base / "inputs" / "decoys").mkdir(parents=True, exist_ok=True)

    _copy_template("campaign_manifest.template.json", base / "campaign_manifest.json")
    _copy_template("targets.template.json", base / "inputs" / "targets.json")
    return base


def create_cell_scaffold(
    benchmark_id: str,
    target: str,
    regime: str,
    run_label: str,
    *,
    root: Path | None = None,
) -> Path:
    benchmark_id = benchmark_id.strip()
    target = target.strip()
    regime = regime.strip()
    run_label = run_label.strip()
    if not all((benchmark_id, target, regime, run_label)):
        raise ValueError("benchmark_id, target, regime, and run_label are required")

    campaign_dir = create_campaign_scaffold(benchmark_id, root=root)
    cell = campaign_dir / target / regime / run_label
    for name in ("inputs", "raw_outputs", "merged_outputs", "logs", "reports"):
        (cell / name).mkdir(parents=True, exist_ok=True)

    _copy_template("cell_manifest.template.json", cell / "cell_manifest.json")
    _copy_template("provenance.template.json", cell / "provenance.json")
    _copy_template("failure_summary.template.json", cell / "failure_summary.json")
    _copy_template("notes.template.md", cell / "notes.md")
    _write_json_if_missing(cell / "metrics_summary.json", _DEFAULT_METRICS_SUMMARY)
    return cell
