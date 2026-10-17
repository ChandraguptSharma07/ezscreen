"""Fail-closed Kaggle T4 runtime for frozen Uni-Dock/canonical-Vina development docking."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import time

WORK = pathlib.Path("/kaggle/working")
INPUT = pathlib.Path(os.environ["SWIFTSCREEN_DOCKING_INPUT"])
CONTRACT = json.loads(pathlib.Path(os.environ["SWIFTSCREEN_DOCKING_CONTRACT"]).read_text())
LOG_TAIL_LIMIT = 2000


def digest(path: pathlib.Path) -> str:
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tree_digest(root: pathlib.Path) -> str:
    result = hashlib.sha256()
    root = pathlib.Path(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result.update(path.relative_to(root).as_posix().encode() + b"\0")
        result.update(path.read_bytes() + b"\0")
    return result.hexdigest()


commands: list[dict] = []


def run(argv, *, cwd=None, timeout=7200, check=True, target="runtime"):
    started = time.time()
    process = subprocess.run(
        [str(item) for item in argv],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    commands.append({
        "target": target,
        "argv": [str(item) for item in argv],
        "cwd": str(cwd) if cwd is not None else None,
        "timeout_seconds": timeout,
        "return_code": process.returncode,
        "elapsed_seconds": time.time() - started,
        "output_tail": (process.stdout or "")[-LOG_TAIL_LIMIT:],
    })
    if check and process.returncode:
        raise RuntimeError(
            f"command failed for {target}: {process.returncode}: "
            f"{(process.stdout or '')[-500:]}"
        )
    return process


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if (
    CONTRACT["execution"]["launch_authorized"] is not True
    or CONTRACT["execution"]["holdout_execution_authorized"] is not False
    or CONTRACT["execution"]["adaptive_outcome_inspection_permitted"] is not False
    or CONTRACT["target_id"] not in CONTRACT["execution"]["development_target_ids"]
    or CONTRACT["arm_id"] not in {"B0_frozen_baseline", "S1_budget_matched_replicates"}
    or CONTRACT["engine"]["name"] != "Uni-Dock"
    or CONTRACT["engine"]["scoring"] != "vina"
):
    raise RuntimeError("development-only Uni-Dock/Vina execution gate failed")

query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
)
gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
gpu_models = [row.split(",")[1].strip() for row in gpu_rows if len(row.split(",")) >= 2]
if query.returncode or not gpu_models or any("T4" not in model for model in gpu_models):
    raise RuntimeError(f"Kaggle Tesla T4 required; observed {gpu_models!r}")
(WORK / "hardware_manifest.json").write_text(json.dumps({
    "gate_passed": True,
    "gpu_count": len(gpu_models),
    "gpu_models": gpu_models,
    "query_stdout": query.stdout,
}, indent=2) + "\n")

input_contract_path = INPUT / "docking_input_contract.json"
if digest(input_contract_path) != CONTRACT["input_contract_sha256"]:
    raise RuntimeError("docking input contract hash mismatch")
input_contract = json.loads(input_contract_path.read_text())
if input_contract["target_id"] != CONTRACT["target_id"]:
    raise RuntimeError("docking input target mismatch")
if input_contract.get("contains_labels") is not False:
    raise RuntimeError("docking transport must be label-free")
for relative, expected in input_contract["members_sha256"].items():
    path = INPUT / relative
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError(f"docking input member mismatch: {relative}")

identity_path = INPUT / "identity_manifest.csv"
with identity_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    headers = [str(value).lower() for value in (reader.fieldnames or [])]
    if any(token in headers for token in ("label", "active", "class")):
        raise RuntimeError("labels leaked into docking identity manifest")
    identities = list(reader)
if len(identities) != CONTRACT["source_count"]:
    raise RuntimeError("docking source count mismatch")
seen_keys: set[str] = set()
ligands = []
for row in identities:
    source_key = row["source_key"]
    sample_index = int(row["sample_index"])
    if source_key != f"{row['source_id']}@{sample_index}" or source_key in seen_keys:
        raise RuntimeError(f"docking source identity mismatch: {source_key}")
    if int(row["state_index"]) != 0 or row["preparation_status"] != "prepared":
        raise RuntimeError(f"B0 docking input state mismatch: {source_key}")
    ligand_path = INPUT / row["pdbqt"]
    if not ligand_path.is_file() or digest(ligand_path) != row["prepared_sha256"]:
        raise RuntimeError(f"prepared ligand hash mismatch: {source_key}")
    seen_keys.add(source_key)
    ligands.append((row, ligand_path))

receptor = INPUT / "receptor.pdbqt"
if digest(receptor) != CONTRACT["receptor_sha256"]:
    raise RuntimeError("prepared receptor hash mismatch")

run(["apt-get", "update", "-qq"], timeout=1200, target="apt_update")
run(
    ["apt-get", "install", "-y", "-qq", "--no-install-recommends", "libboost-all-dev"],
    timeout=1800,
    target="apt_boost",
)
repo = pathlib.Path("/kaggle/temp/unidock-frozen")
if repo.exists():
    shutil.rmtree(repo)
run(
    ["git", "clone", "https://github.com/dptech-corp/Uni-Dock", str(repo)],
    timeout=1800,
    target="unidock_clone",
)
run(
    ["git", "-C", str(repo), "checkout", "--detach", CONTRACT["engine"]["git_commit"]],
    timeout=300,
    target="unidock_checkout",
)
grid_header = repo / "unidock/src/cuda/kernel.h"
vina_source = repo / "unidock/src/lib/vina.cpp"
grid_text = grid_header.read_text()
vina_text = vina_source.read_text()
old_cap = str(CONTRACT["engine"]["capacity_patch"]["original_grid_points"])
new_cap = str(CONTRACT["engine"]["capacity_patch"]["max_grid_points"])
if (grid_text.count(old_cap), vina_text.count(old_cap)) != (7, 1):
    raise RuntimeError("unexpected Uni-Dock capacity patch sites")
grid_header.write_text(grid_text.replace(old_cap, new_cap))
vina_source.write_text(vina_text.replace(old_cap, new_cap).replace("81^3", "128^3"))
patch = CONTRACT["engine"]["capacity_patch"]
if (
    digest(grid_header) != patch["kernel_h_sha256"]
    or digest(vina_source) != patch["vina_cpp_sha256"]
    or tree_digest(repo / "unidock/src") != patch["source_tree_sha256"]
):
    raise RuntimeError("Uni-Dock frozen capacity-only source hash mismatch")
build = pathlib.Path("/kaggle/temp/unidock-build")
if build.exists():
    shutil.rmtree(build)
run(
    [
        "cmake", "-S", str(repo / "unidock"), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=60;75",
    ],
    timeout=1800,
    target="unidock_configure",
)
run(
    ["cmake", "--build", str(build), f"-j{max(1, os.cpu_count() or 1)}"],
    timeout=3600,
    target="unidock_build",
)
binary = build / "unidock"
if not binary.is_file() or binary.stat().st_size < 500000:
    raise RuntimeError("Uni-Dock binary missing after frozen source build")
commit = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
if commit != CONTRACT["engine"]["git_commit"]:
    raise RuntimeError("Uni-Dock commit drift")
(WORK / "docking_executable_manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "engine": "unidock",
    "scoring": "vina",
    "git_commit": commit,
    "source_tree_sha256": tree_digest(repo / "unidock/src"),
    "capacity_only_change": True,
    "grid_point_cap": patch["max_grid_points"],
    "binary": {"sha256": digest(binary), "bytes": binary.stat().st_size},
    "patched_files": {
        "unidock/src/cuda/kernel.h": digest(grid_header),
        "unidock/src/lib/vina.cpp": digest(vina_source),
    },
    "git_status_short": subprocess.run(
        ["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True
    ).stdout,
}, indent=2) + "\n")

(WORK / "docking_inputs_manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "target_id": CONTRACT["target_id"],
    "arm_id": CONTRACT["arm_id"],
    "contains_labels": False,
    "source_count": len(identities),
    "identity_manifest_sha256": digest(identity_path),
    "receptor": {"sha256": digest(receptor), "bytes": receptor.stat().st_size},
    "ligand_bundle_sha256": CONTRACT["ligand_bundle_sha256"],
    "box": CONTRACT["box"],
}, indent=2) + "\n")

raw_rows: list[dict] = []
failure_rows: list[dict] = []
result_pattern = __import__("re").compile(r"REMARK VINA RESULT:\s+(\S+)")
common = CONTRACT["engine"]["common_arguments"]
for search_index, search in enumerate(CONTRACT["searches"]):
    docked = pathlib.Path(f"/kaggle/temp/docked-{search_index:02d}")
    if docked.exists():
        shutil.rmtree(docked)
    docked.mkdir()
    index_path = WORK / f"ligand_index_{search_index:02d}.txt"
    index_path.write_text("\n".join(str(path) for _, path in ligands) + "\n")
    center = CONTRACT["box"]["center"]
    size = CONTRACT["box"]["size"]
    argv = [
        binary,
        "--receptor", receptor,
        "--ligand_index", index_path,
        "--dir", docked,
        "--center_x", center[0], "--center_y", center[1], "--center_z", center[2],
        "--size_x", size[0], "--size_y", size[1], "--size_z", size[2],
        "--spacing", common["spacing"],
        "--scoring", "vina",
        "--seed", search["seed"],
        "--exhaustiveness", search["exhaustiveness"],
        "--num_modes", common["num_modes"],
        "--energy_range", common["energy_range"],
        "--min_rmsd", common["min_rmsd"],
        "--max_evals", common["max_evals"],
        "--max_step", common["max_step"],
        "--cpu", common["cpu"],
        "--max_gpu_memory", common["max_gpu_memory"],
        "--verbosity", common["verbosity"],
    ]
    result = run(
        argv,
        timeout=43200,
        check=False,
        target=f"{CONTRACT['target_id']}:{CONTRACT['arm_id']}:{search_index}",
    )
    for row, ligand_path in ligands:
        output = docked / f"{ligand_path.stem}_out.pdbqt"
        match = result_pattern.search(output.read_text(errors="replace")) if output.is_file() else None
        score = None
        reason = ""
        if match:
            try:
                candidate = float(match.group(1))
                if math.isfinite(candidate):
                    score = candidate
                else:
                    reason = "non_finite_score"
            except ValueError:
                reason = "unparseable_score"
        else:
            reason = "missing_pose_or_vina_result"
        output_row = {
            "source_key": row["source_key"],
            "source_id": row["source_id"],
            "sample_index": int(row["sample_index"]),
            "state_index": 0,
            "ligand": row["ligand"],
            "search_index": search_index,
            "seed": int(search["seed"]),
            "exhaustiveness": int(search["exhaustiveness"]),
            "engine": "unidock",
            "scoring": "vina",
            "score": "" if score is None else score,
            "status": "failed" if score is None else "scored",
            "reason": reason,
            "prepared_sha256": row["prepared_sha256"],
        }
        raw_rows.append(output_row)
        if score is None:
            failure_rows.append(output_row)
    shutil.rmtree(docked)
    if result.returncode and not any(
        item["search_index"] == search_index and item["status"] == "scored"
        for item in raw_rows
    ):
        raise RuntimeError(f"Uni-Dock search {search_index} failed without any score")

expected_constituents = len(identities) * len(CONTRACT["searches"])
if len(raw_rows) != expected_constituents:
    raise RuntimeError("raw docking constituent count mismatch")
fields = [
    "source_key", "source_id", "sample_index", "state_index", "ligand",
    "search_index", "seed", "exhaustiveness", "engine", "scoring", "score",
    "status", "reason", "prepared_sha256",
]
write_csv(WORK / "raw_scores.csv", raw_rows, fields)
write_csv(WORK / "docking_failures.csv", failure_rows, fields)
(WORK / "docking_command_ledger.json").write_text(json.dumps({
    "schema_version": 1,
    "log_tail_limit": LOG_TAIL_LIMIT,
    "commands": commands,
}, indent=2) + "\n")
subprocess.run(
    [os.sys.executable, "-m", "pip", "freeze"],
    stdout=(WORK / "pip_freeze.txt").open("w"),
    check=True,
)
(WORK / "environment_manifest.json").write_text(json.dumps({
    "python_version": os.sys.version,
    "pip_freeze_sha256": digest(WORK / "pip_freeze.txt"),
    "credential_policy": "full environment never serialized",
}, indent=2) + "\n")
(WORK / "docking_runtime.json").write_text(json.dumps({
    "schema_version": 1,
    "status": "complete",
    "target_id": CONTRACT["target_id"],
    "pdb_id": CONTRACT["pdb_id"],
    "arm_id": CONTRACT["arm_id"],
    "source_count": len(identities),
    "search_count": len(CONTRACT["searches"]),
    "expected_constituents": expected_constituents,
    "scored_constituents": len(raw_rows) - len(failure_rows),
    "failed_constituents": len(failure_rows),
    "engine": "unidock",
    "scoring": "vina",
    "aggregation_performed": False,
    "enrichment_computed": False,
    "labels_present": False,
    "gpu_models": gpu_models,
}, indent=2) + "\n")
artifacts = {}
for name in (
    "raw_scores.csv", "docking_failures.csv", "hardware_manifest.json",
    "docking_executable_manifest.json", "docking_inputs_manifest.json",
    "docking_command_ledger.json", "docking_runtime.json", "pip_freeze.txt",
    "environment_manifest.json",
):
    path = WORK / name
    artifacts[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
(WORK / "docking_artifact_manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "artifacts": artifacts,
}, indent=2) + "\n")
(WORK / "done.flag").write_text("complete\n")
