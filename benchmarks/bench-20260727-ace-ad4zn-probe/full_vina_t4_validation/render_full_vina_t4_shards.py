"""Render ACE UniDock Vina-only T4 validation shards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "full_vina_vinardo_port"
sys.path.insert(0, str(BASE))
import render_full_vina_vinardo_shards as base  # noqa: E402

EXPECTED_GPU_COUNT = 2
BOOTSTRAP_SEED = 20260801

CAMPAIGN_PROTOCOL = {
    "schema_version": 1,
    "campaign": "full_vina_t4_validation",
    "scientific_question": "Does the unchanged automated ACE UniDock Vina-mode workflow achieve published Vina screening quality?",
    "target": "ACE",
    "pdb_id": "1O86",
    "dataset": "DUD Release 2",
    "records": {"total": 1846, "actives": 49, "decoys": 1797},
    "protocol": {
        "preparation": "AutoDockTools first; Meeko only after genuine ADT failure",
        "prepared_identity": "must exactly match frozen AD4 attempt 04 prepared hashes",
        "box_center": [40.7317, 32.8822, 47.3635],
        "box_size": [36.8194, 41.835, 43.7048],
        "autobox_add_angstrom": 15,
        "exhaustiveness": 32,
        "scoring": "vina",
        "grid_spacing_angstrom": 0.375,
        "ranking": "raw UniDock Vina score ascending",
    },
    "hardware": {
        "required_gpu_count": 2,
        "required_model": "Tesla T4",
        "fail_fast": True,
    },
    "unidock": {
        "repository": "https://github.com/dptech-corp/Uni-Dock",
        "commit": base.UNIDOCK_COMMIT,
        "grid_point_cap": base.GRID_POINT_CAP,
        "source_change": "capacity limit only; no intended scoring protocol change",
    },
    "published": {"roc_auc": 0.431, "bedroc20": 0.159},
    "predefined_similarity_margins_pipeline_minus_published": {
        "roc_auc": -0.02,
        "bedroc20": -0.02,
    },
    "bootstrap": {
        "method": "stratified nonparametric",
        "replicates": 10000,
        "seed": BOOTSTRAP_SEED,
        "confidence": 0.95,
    },
    "prohibitions": [
        "no UniDock optimization",
        "no label tuning",
        "no post-hoc ranking selection",
    ],
}
EXPECTED_GPU_TOKEN = "T4"

HARDWARE_GATE = r"""
import json, pathlib, subprocess, time

_gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"],
    capture_output=True, text=True,
)
_gpu_list = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
if _gpu_query.returncode != 0:
    raise RuntimeError(f"Tesla T4 required; nvidia-smi failed: {_gpu_query.stderr[-500:]}")
_gpu_rows = [line.strip() for line in _gpu_query.stdout.splitlines() if line.strip()]
_gpu_models = [row.split(",")[1].strip() for row in _gpu_rows if len(row.split(",")) >= 2]
if len(_gpu_rows) != 2 or len(_gpu_models) != 2 or any("T4" not in model for model in _gpu_models):
    raise RuntimeError(f"Tesla T4x2 required before preparation; observed rows={_gpu_rows!r}")
HARDWARE_PROVENANCE = {
    "gate_passed": True,
    "expected_gpu_count": 2,
    "expected_model_token": "T4",
    "gpu_count": len(_gpu_rows),
    "gpu_models": _gpu_models,
    "query_stdout": _gpu_query.stdout,
    "list_stdout": _gpu_list.stdout,
    "captured_epoch": time.time(),
}
pathlib.Path("/kaggle/working/hardware_manifest.json").write_text(
    json.dumps(HARDWARE_PROVENANCE, indent=2) + "\n"
)
print(HARDWARE_PROVENANCE)
"""

SOURCE_BUILD_PROVENANCE = rf'''
import hashlib, json, pathlib, shutil, subprocess

def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _tree_sha256(root):
    digest = hashlib.sha256()
    root = pathlib.Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

def _capture(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {{"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}}

_repo = pathlib.Path("/tmp/unidock")
_source = _repo / "unidock/src"
_binary = pathlib.Path(shutil.which("unidock") or "")
_commit = subprocess.run(
    ["git", "-C", str(_repo), "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
if _commit != "{base.UNIDOCK_COMMIT}":
    raise RuntimeError(f"UniDock commit drift: {{_commit}}")
if not _binary.is_file():
    raise RuntimeError("UniDock binary missing after source build")
SOURCE_BUILD_PROVENANCE = {{
    "official_repository": "https://github.com/dptech-corp/Uni-Dock",
    "git_commit": _commit,
    "source_tree_sha256": _tree_sha256(_source),
    "binary_path": str(_binary),
    "binary_sha256": _sha256_file(_binary),
    "binary_bytes": _binary.stat().st_size,
    "grid_point_cap": {base.GRID_POINT_CAP},
    "capacity_only_change": True,
    "patched_files": {{
        "unidock/src/cuda/kernel.h": _sha256_file(_repo / "unidock/src/cuda/kernel.h"),
        "unidock/src/lib/vina.cpp": _sha256_file(_repo / "unidock/src/lib/vina.cpp"),
    }},
    "git_status": _capture(["git", "-C", str(_repo), "status", "--short"]),
    "git_diff": _capture(["git", "-C", str(_repo), "diff", "--", "unidock/src/cuda/kernel.h", "unidock/src/lib/vina.cpp"]),
    "nvcc_version": _capture(["nvcc", "--version"]),
    "cmake_version": _capture(["cmake", "--version"]),
    "unidock_help": _capture([str(_binary), "--help"]),
}}
pathlib.Path("/kaggle/working/source_build_manifest.json").write_text(
    json.dumps(SOURCE_BUILD_PROVENANCE, indent=2) + "\n"
)
print({{key: SOURCE_BUILD_PROVENANCE[key] for key in (
    "git_commit", "source_tree_sha256", "binary_sha256", "grid_point_cap"
)}})
'''

INPUT_PROVENANCE = r"""
_source_manifests = list(pathlib.Path("/kaggle/input").rglob("source_manifest.json"))
if len(_source_manifests) != 1:
    raise RuntimeError(f"expected one source_manifest.json, found {len(_source_manifests)}")
INPUT_PROVENANCE = {
    "target": "ACE",
    "pdb_id": "1O86",
    "dataset": "DUD Release 2",
    "source_manifest_sha256": sha256(_source_manifests[0]),
    "receptor_sha256": sha256(RECEPTOR_PDBQT),
    "expected_receptor_sha256": EXPECTED_RECEPTOR_SHA256,
    "prepared_manifest_sha256": sha256(pathlib.Path("/kaggle/working/prepared_manifest.csv")),
    "prepared_identity_check_sha256": sha256(pathlib.Path("/kaggle/working/prepared_identity_check.json")),
    "prepared_reference": f"ACE AD4 attempt 04 shard {SHARD_INDEX:02d}",
    "box_center": [40.7317, 32.8822, 47.3635],
    "box_size": [36.8194, 41.835, 43.7048],
    "grid_spacing": 0.375,
    "scoring": "vina",
    "exhaustiveness": 32,
    "lower_is_better": True,
}
pathlib.Path("/kaggle/working/input_manifest.json").write_text(
    json.dumps(INPUT_PROVENANCE, indent=2) + "\n"
)
print(INPUT_PROVENANCE)
"""


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"{label}: expected one patch point, found {source.count(old)}"
        )
    return source.replace(old, new, 1)


def render() -> None:
    original_here = base.HERE
    try:
        base.HERE = HERE
        base.render()
    finally:
        base.HERE = original_here

    notebooks = HERE / "notebooks"
    for shard in range(base.SHARDS):
        old_path = notebooks / f"ace_vina_vinardo_shard_{shard:02d}.ipynb"
        new_path = notebooks / f"ace_vina_t4_shard_{shard:02d}.ipynb"
        notebook = json.loads(old_path.read_text(encoding="utf-8"))
        notebook["cells"].insert(1, base.ad4.cell(HARDWARE_GATE))
        notebook["cells"].insert(3, base.ad4.cell(SOURCE_BUILD_PROVENANCE))
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            source = source.replace(
                'for scoring in ("vina", "vinardo"):', 'for scoring in ("vina",):'
            )
            source = source.replace(
                "UniDock Vina and Vinardo accelerated ports; not Vina-equivalent",
                "UniDock Vina-mode T4 validation; not AutoDock Vina-equivalent",
            )
            source = source.replace(
                "# ACE Vina/Vinardo shard",
                "# ACE Vina-only T4 validation shard",
            )
            if "EXPECTED_RECEPTOR_SHA256" in source:
                source += "\n" + INPUT_PROVENANCE
            if "output_names = [" in source:
                source = _replace_once(
                    source,
                    '"prep_messages.jsonl", "prepared_identity_check.json"]',
                    '"prep_messages.jsonl", "prepared_identity_check.json",\n'
                    '                "hardware_manifest.json", "source_build_manifest.json",\n'
                    '                "input_manifest.json"]',
                    "output manifest",
                )
                source = _replace_once(
                    source,
                    '"grid_point_cap": 2097152,\n}',
                    '"grid_point_cap": 2097152,\n'
                    '    "hardware": HARDWARE_PROVENANCE,\n'
                    '    "source_build": {key: SOURCE_BUILD_PROVENANCE[key] for key in ('
                    '"git_commit", "source_tree_sha256", "binary_sha256", "grid_point_cap")},\n'
                    '    "input_manifest_sha256": sha256(pathlib.Path("/kaggle/working/input_manifest.json")),\n}',
                    "runtime provenance",
                )
            cell["source"] = source.splitlines(keepends=True)

        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if source.strip():
                compile(source, f"ace_vina_t4_{shard:02d}.py", "exec")
        new_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        old_path.unlink()

    plan_path = HERE / "shard_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.update(
        modes=["vina"],
        hardware={"gpu_count": EXPECTED_GPU_COUNT, "model_token": EXPECTED_GPU_TOKEN},
        campaign="full_vina_t4_validation",
    )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (HERE / "campaign_protocol.json").write_text(
        json.dumps(CAMPAIGN_PROTOCOL, indent=2) + "\n", encoding="utf-8"
    )
    quota = {
        "schema_version": 1,
        "account_count": 6,
        "estimated_gpu_hours_per_account": 20.0,
        "default_usable_hours_per_account": 16.5,
        "maximum_planned_gpu_hours": 100.0,
        "reserve_gpu_hours": 20.0,
        "maximum_concurrent_notebooks_per_account": 2,
        "maximum_concurrent_gpu_notebooks_per_account": 2,
        "maximum_concurrent_cpu_notebooks_per_account": 10,
        "allocations_gpu_hours": {"t4_validation": 24.0},
        "planned_gpu_hours": 0.0,
        "observed_gpu_hours": 0.0,
        "accounts": {},
    }
    quota_path = HERE / "quota_budget.json"
    if not quota_path.exists():
        quota_path.write_text(json.dumps(quota, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    render()
