"""Render the two-stage ADA/1NDW UniDock Vina-mode T4 campaign."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOKS = HERE / "notebooks"
SOURCE_MANIFEST_PATH = (
    ROOT / "benchmarks/bench-20260723-vinardo-dud28/source_manifest.json"
)
PROTOCOL_PATH = HERE / "campaign_protocol.json"
EVIDENCE_PATH = HERE / "target_evidence.json"
PREPARED_INPUT_MANIFEST = HERE / "prepared_input_manifest.json"

TOTAL, ACTIVES, DECOYS, SHARDS = 966, 39, 927, 12
UNIDOCK_COMMIT = "ebde3b4a22c8b2c7b26e523cc15109aab815be21"
GRID_POINT_CAP = 2_097_152
BOX_CENTER = [42.2931, 17.3816, 49.5368]
BOX_SIZE = [36.9618, 39.689, 38.3826]


def cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def notebook(cells: list[str], *, stage: str) -> dict:
    rendered = [cell(source) for source in cells]
    for index, item in enumerate(rendered):
        source = "".join(item["source"])
        if source.strip():
            compile(source, f"ada_{stage}_{index:02d}.py", "exec")
    return {
        "cells": rendered,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "ezscreen": {
                "campaign": "bench-20260801-ada-1ndw-vina-t4-validation",
                "stage": stage,
                "required_accelerator": "NvidiaTeslaT4",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


HARDWARE_GATE = r'''
import json, pathlib, subprocess, time

STARTED = time.time()
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
    raise RuntimeError(f"Tesla T4x2 required before work; observed rows={_gpu_rows!r}")
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
'''


COMMON_IMPORTS = r'''
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working/ada_t4")
SCRATCH = Path("/kaggle/temp/ada_t4")
LOGS = Path("/kaggle/working/logs")
for directory in (WORK, SCRATCH, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)

def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe zip member: {member.filename}")
        bundle.extractall(destination)

def run(command, *, cwd=None, timeout=7200, check=True, log_name=None):
    started = time.time()
    process = subprocess.run(
        [str(part) for part in command], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    if log_name:
        (LOGS / log_name).write_text(process.stdout or "", errors="replace")
    if check and process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {command}\n{(process.stdout or '')[-3000:]}"
        )
    process.runtime_seconds = time.time() - started
    return process

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
'''


SOURCE_VERIFY = r'''
EMBEDDED_SOURCE_MANIFEST = __SOURCE_MANIFEST__
TARGET_EVIDENCE = __TARGET_EVIDENCE__
CAMPAIGN_PROTOCOL = __CAMPAIGN_PROTOCOL__
required_roles = {
    "active_libraries", "decoy_libraries", "receptors",
    "crystallographic_ligands", "autodocktools_runtime",
}
mounted = {path.name: path for path in Path("/kaggle/input").rglob("*") if path.is_file()}
source_dir = SCRATCH / "sources"
source_dir.mkdir(exist_ok=True)
SOURCE_FILES, verified = {}, {}
for item in EMBEDDED_SOURCE_MANIFEST["sources"]:
    if item["role"] not in required_roles:
        continue
    path = mounted.get(item["filename"], source_dir / item["filename"])
    if not path.exists():
        request = urllib.request.Request(item["url"], headers={"User-Agent": "SwiftScreen-ADA-T4/1"})
        with urllib.request.urlopen(request, timeout=600) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"immutable source mismatch: {item['filename']}")
    SOURCE_FILES[item["filename"]] = path
    verified[item["role"]] = item["sha256"]
if set(verified) != required_roles:
    raise RuntimeError(f"missing source roles: {required_roles - set(verified)}")
print({"verified_source_roles": sorted(verified)})
'''


PREP_INSTALL = r'''
apt = run(["apt-get", "update", "-qq"], log_name="apt_update.log")
run(
    ["apt-get", "install", "-y", "-qq", "openbabel"],
    log_name="apt_install.log",
)
run([sys.executable, "-m", "pip", "install", "-q", "meeko", "gemmi"], log_name="pip_prep.log")
mgl_unpack = SCRATCH / "mgl_dist"
safe_extract_tar(SOURCE_FILES["mgltools_x86_64Linux2_1.5.6-patch1.tar.gz"], mgl_unpack)
mgl_dist = next(mgl_unpack.glob("mgltools_*"))
mgl_root = SCRATCH / "mgltools"
run(["bash", "install.sh", "-d", str(mgl_root)], cwd=mgl_dist, log_name="mgltools_install.log")
pythonsh = mgl_root / "bin/pythonsh"
utilities = mgl_root / "MGLToolsPckgs/AutoDockTools/Utilities24"
if not pythonsh.exists() or not (utilities / "prepare_receptor4.py").exists() or not (utilities / "prepare_ligand4.py").exists():
    raise RuntimeError("MGLTools/AutoDockTools installation incomplete")
'''


PREPARE = r'''
data_dir = SCRATCH / "dud"
for filename in (
    "dud_targets.tar.gz", "dud_target_ligands.tar.gz",
    "dud_ligands2006.tar.gz", "dud_decoys2006.tar.gz",
):
    safe_extract_tar(SOURCE_FILES[filename], data_dir)

receptor_source = data_dir / "targets/ada/rec.pdb"
crystal_ligand = data_dir / "targets/ada/xtal-lig.mol2"
if sha256(receptor_source) != TARGET_EVIDENCE["dud_release_2"]["receptor_sha256"]:
    raise RuntimeError("ADA receptor source hash mismatch")
if sha256(crystal_ligand) != TARGET_EVIDENCE["dud_release_2"]["crystal_ligand_sha256"]:
    raise RuntimeError("ADA crystal ligand hash mismatch")

receptor_normalized = WORK / "ada_receptor_element_normalized.pdb"
source_receptor_lines = receptor_source.read_text(errors="replace").splitlines()
normalized_receptor_lines = []
normalizations = []
for line_number, line in enumerate(source_receptor_lines, start=1):
    atom_name = line[12:16].strip() if len(line) >= 16 else ""
    residue_name = line[17:20].strip() if len(line) >= 20 else ""
    element = line[76:78].strip() if len(line) >= 78 else ""
    if line.startswith(("ATOM  ", "HETATM")) and atom_name == "ZN" and residue_name == "ZN" and not element:
        normalized = line.ljust(76) + "ZN"
        normalizations.append({
            "line_number": line_number, "atom_name": atom_name, "residue_name": residue_name,
            "original_line": line, "normalized_line": normalized,
            "change": "populate blank PDB element columns 77-78 with ZN",
        })
        line = normalized
    normalized_receptor_lines.append(line)
if len(normalizations) != 1:
    raise RuntimeError(f"expected one blank-element ADA ZN atom, found {len(normalizations)}")
receptor_normalized.write_text("\n".join(normalized_receptor_lines) + "\n")
if len(source_receptor_lines) != len(normalized_receptor_lines):
    raise RuntimeError("receptor normalization changed atom-line count")
normalization_manifest = {
    "classification": "PDB element-column format repair for AutoDockTools",
    "source_receptor_sha256": sha256(receptor_source),
    "normalized_receptor_sha256": sha256(receptor_normalized),
    "source_lines": len(source_receptor_lines), "normalized_lines": len(normalized_receptor_lines),
    "atoms_added": 0, "atoms_removed": 0, "coordinates_changed": 0,
    "normalizations": normalizations,
}
Path("/kaggle/working/receptor_normalization.json").write_text(
    json.dumps(normalization_manifest, indent=2) + "\n"
)

xtal_lines = crystal_ligand.read_text(errors="replace").splitlines()
atom_start = xtal_lines.index("@<TRIPOS>ATOM") + 1
atom_end = xtal_lines.index("@<TRIPOS>BOND")
coords = [tuple(map(float, line.split()[2:5])) for line in xtal_lines[atom_start:atom_end] if len(line.split()) >= 6]
mins = [min(row[i] for row in coords) for i in range(3)]
maxs = [max(row[i] for row in coords) for i in range(3)]
box_center = [(low + high) / 2 for low, high in zip(mins, maxs)]
box_size = [high - low + 30.0 for low, high in zip(mins, maxs)]
expected_center = CAMPAIGN_PROTOCOL["protocol"]["box_center"]
expected_size = CAMPAIGN_PROTOCOL["protocol"]["box_size"]
if any(abs(a - b) > 1e-4 for a, b in zip(box_center, expected_center)) or any(abs(a - b) > 1e-4 for a, b in zip(box_size, expected_size)):
    raise RuntimeError(f"ADA crystal box mismatch: {box_center=} {box_size=}")
box_manifest = {
    "target": "ADA", "pdb_id": "1NDW", "crystal_ligand_sha256": sha256(crystal_ligand),
    "atoms": len(coords), "coordinate_min": mins, "coordinate_max": maxs,
    "center": expected_center, "size": expected_size, "autobox_add_angstrom": 15.0,
}
Path("/kaggle/working/box_manifest.json").write_text(json.dumps(box_manifest, indent=2) + "\n")

def mol2_records(path: Path) -> list[bytes]:
    marker = b"@<TRIPOS>MOLECULE"
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    return [marker + part for part in payload.split(marker)[1:]]

def molecule_id(record: bytes) -> str:
    return record.decode("utf-8", errors="replace").splitlines()[1].strip()

actives = mol2_records(data_dir / "dud_ligands2006/ada_ligands.mol2.gz")
decoys = mol2_records(data_dir / "dud_decoys2006/ada_decoys.mol2.gz")
if (len(actives), len(decoys)) != (39, 927):
    raise RuntimeError(f"ADA DUD2 count mismatch: {(len(actives), len(decoys))}")
all_records = [(record, 1) for record in actives] + [(record, 0) for record in decoys]

prepared_dir = WORK / "prepared"
source_dir = WORK / "source"
prepared_dir.mkdir(exist_ok=True)
source_dir.mkdir(exist_ok=True)
receptor_pdbqt = prepared_dir / "receptor.pdbqt"
receptor_run = run(
    [pythonsh, utilities / "prepare_receptor4.py", "-r", receptor_normalized, "-o", receptor_pdbqt, "-A", "hydrogens"],
    check=False, log_name="prepare_receptor.log",
)
if receptor_run.returncode or not receptor_pdbqt.exists() or not receptor_pdbqt.stat().st_size:
    raise RuntimeError("ADA receptor AutoDockTools preparation failed")
receptor_pdbqt.write_text(
    "\n".join(line for line in receptor_pdbqt.read_text(errors="replace").splitlines() if line.strip() != "END") + "\n"
)

prepared, prep_failures, fallbacks, prep_messages = [], [], [], []
for index, (record, label) in enumerate(all_records):
    source_id = molecule_id(record)
    record_id = f"{source_id}@{index}"
    mol2 = source_dir / f"{index:04d}.mol2"
    pdbqt = prepared_dir / f"{index:04d}.pdbqt"
    mol2.write_bytes(record)
    adt = run([pythonsh, utilities / "prepare_ligand4.py", "-l", mol2, "-o", pdbqt], check=False)
    prep_messages.append({
        "record_id": record_id, "sample_index": index, "method": "autodocktools",
        "returncode": adt.returncode, "runtime_seconds": adt.runtime_seconds,
        "tail": (adt.stdout or "")[-1000:],
    })
    method = "autodocktools"
    if adt.returncode or not pdbqt.exists() or not pdbqt.stat().st_size:
        method = "meeko_fallback"
        sdf = source_dir / f"{index:04d}.sdf"
        conversion = run(["obabel", "-imol2", mol2, "-osdf", "-O", sdf], check=False)
        mk_base = ([shutil.which("mk_prepare_ligand")] if shutil.which("mk_prepare_ligand") else [sys.executable, "-m", "meeko.cli.mk_prepare_ligand"])
        meeko = run(mk_base + ["-i", sdf, "-o", pdbqt], check=False) if conversion.returncode == 0 and sdf.exists() else None
        fallbacks.append({
            "record_id": record_id, "sample_index": index, "source_id": source_id, "label": label,
            "adt_returncode": adt.returncode, "conversion_returncode": conversion.returncode,
            "meeko_returncode": None if meeko is None else meeko.returncode,
        })
        if meeko is None or meeko.returncode or not pdbqt.exists() or not pdbqt.stat().st_size:
            prep_failures.append({
                "record_id": record_id, "sample_index": index, "source_id": source_id,
                "label": label, "reason": "adt_then_meeko_failed",
            })
            continue
    pdbqt.write_text(
        "\n".join(line for line in pdbqt.read_text(errors="replace").splitlines() if line.strip() != "END") + "\n"
    )
    prepared.append({
        "record_id": record_id, "sample_index": index, "source_id": source_id,
        "label": label, "prep_method": method, "source_sha256": sha256(mol2),
        "prepared_sha256": sha256(pdbqt), "pdbqt": f"ligands/{index:04d}.pdbqt",
    })

prepared_manifest = Path("/kaggle/working/prepared_manifest.csv")
prep_failures_path = Path("/kaggle/working/prep_failures.csv")
fallbacks_path = Path("/kaggle/working/fallbacks.csv")
write_csv(prepared_manifest, prepared, ["record_id", "sample_index", "source_id", "label", "prep_method", "source_sha256", "prepared_sha256", "pdbqt"])
write_csv(prep_failures_path, prep_failures, ["record_id", "sample_index", "source_id", "label", "reason"])
write_csv(fallbacks_path, fallbacks, ["record_id", "sample_index", "source_id", "label", "adt_returncode", "conversion_returncode", "meeko_returncode"])
with Path("/kaggle/working/prep_messages.jsonl").open("w", encoding="utf-8") as handle:
    for row in prep_messages:
        handle.write(json.dumps(row) + "\n")

input_manifest = {
    "target": "ADA", "pdb_id": "1NDW", "dataset": "DUD Release 2",
    "source_archive_hashes": verified, "source_receptor_sha256": sha256(receptor_source),
    "normalized_receptor_sha256": sha256(receptor_normalized),
    "receptor_normalization_manifest_sha256": sha256(Path("/kaggle/working/receptor_normalization.json")),
    "crystal_ligand_sha256": sha256(crystal_ligand), "receptor_pdbqt_sha256": sha256(receptor_pdbqt),
    "prepared_manifest_sha256": sha256(prepared_manifest), "records": {"total": 966, "actives": 39, "decoys": 927},
    "prepared": len(prepared), "preparation_failures": len(prep_failures), "fallbacks": len(fallbacks),
    "record_identity": "source_id@sample_index", "box": box_manifest,
}
Path("/kaggle/working/input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")

bundle_path = Path("/kaggle/working/prepared_bundle.zip")
with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(receptor_pdbqt, "receptor.pdbqt")
    for path, name in (
        (prepared_manifest, "prepared_manifest.csv"),
        (prep_failures_path, "prep_failures.csv"),
        (fallbacks_path, "fallbacks.csv"),
        (Path("/kaggle/working/input_manifest.json"), "input_manifest.json"),
        (Path("/kaggle/working/box_manifest.json"), "box_manifest.json"),
        (Path("/kaggle/working/receptor_normalization.json"), "receptor_normalization.json"),
    ):
        archive.write(path, name)
    for row in prepared:
        archive.write(prepared_dir / f"{int(row['sample_index']):04d}.pdbqt", row["pdbqt"])

runtime = {
    "stage": "preparation", "target": "ADA", "pdb_id": "1NDW", "intended": 966,
    "prepared": len(prepared), "preparation_failures": len(prep_failures), "fallbacks": len(fallbacks),
    "total_seconds": time.time() - STARTED, "gpu_count": HARDWARE_PROVENANCE["gpu_count"],
    "gpu_models": HARDWARE_PROVENANCE["gpu_models"], "prepared_bundle_sha256": sha256(bundle_path),
}
Path("/kaggle/working/runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
if len(prepared) / 966 < 0.98:
    raise RuntimeError(f"preparation coverage below 0.98: {len(prepared)}/966")
with zipfile.ZipFile("/kaggle/working/output.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in (
        "prepared_bundle.zip", "prepared_manifest.csv", "prep_failures.csv", "fallbacks.csv",
        "prep_messages.jsonl", "input_manifest.json", "box_manifest.json", "hardware_manifest.json",
        "receptor_normalization.json",
        "runtime.json",
    ):
        archive.write(Path("/kaggle/working") / name, name)
    for log in sorted(LOGS.glob("*")):
        archive.write(log, f"logs/{log.name}")
Path("/kaggle/working/done.flag").write_text("ok\n")
print(runtime)
'''


SOURCE_BUILD = rf'''
import hashlib, json, os, pathlib, shutil, subprocess

def shell(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"build command failed: {{command}}\n{{result.stderr[-2000:]}}")
    return result

shell("DEBIAN_FRONTEND=noninteractive apt-get update -y")
shell("DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libboost-all-dev")
shell("pip install unidock-tools -q")
shell("rm -rf /tmp/unidock && git clone --depth 1 https://github.com/dptech-corp/Uni-Dock /tmp/unidock")
shell("git -C /tmp/unidock checkout --detach {UNIDOCK_COMMIT}")
repo = pathlib.Path("/tmp/unidock")
grid_header = repo / "unidock/src/cuda/kernel.h"
vina_source = repo / "unidock/src/lib/vina.cpp"
grid_text, vina_text = grid_header.read_text(), vina_source.read_text()
if (grid_text.count("531441"), vina_text.count("531441")) != (7, 1):
    raise RuntimeError("unexpected UniDock grid-cap patch sites")
grid_header.write_text(grid_text.replace("531441", "{GRID_POINT_CAP}"))
vina_source.write_text(vina_text.replace("531441", "{GRID_POINT_CAP}").replace("81^3", "128^3"))
shell('cmake -S /tmp/unidock/unidock -B /tmp/unidock/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="60;75"')
shell("cmake --build /tmp/unidock/build -j$(nproc)")
binary = pathlib.Path("/tmp/unidock/build/unidock")
if not binary.is_file() or binary.stat().st_size < 500000:
    raise RuntimeError("UniDock source build missing")
shell(f"ln -sf {{binary}} /usr/local/bin/unidock")

def file_sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tree_sha(root):
    digest = hashlib.sha256()
    root = pathlib.Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()

commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
if commit != "{UNIDOCK_COMMIT}":
    raise RuntimeError(f"UniDock commit drift: {{commit}}")
SOURCE_BUILD_PROVENANCE = {{
    "official_repository": "https://github.com/dptech-corp/Uni-Dock",
    "git_commit": commit, "source_tree_sha256": tree_sha(repo / "unidock/src"),
    "binary_sha256": file_sha(binary), "binary_bytes": binary.stat().st_size,
    "grid_point_cap": {GRID_POINT_CAP}, "capacity_only_change": True,
    "patched_files": {{
        "unidock/src/cuda/kernel.h": file_sha(grid_header),
        "unidock/src/lib/vina.cpp": file_sha(vina_source),
    }},
    "git_status": subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True, text=True).stdout,
    "git_diff": subprocess.run(["git", "-C", str(repo), "diff", "--", "unidock/src/cuda/kernel.h", "unidock/src/lib/vina.cpp"], capture_output=True, text=True).stdout,
    "nvcc_version": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout,
    "cmake_version": subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout,
}}
Path("/kaggle/working/source_build_manifest.json").write_text(json.dumps(SOURCE_BUILD_PROVENANCE, indent=2) + "\n")
print({{key: SOURCE_BUILD_PROVENANCE[key] for key in ("git_commit", "source_tree_sha256", "binary_sha256", "grid_point_cap")}})
'''


DOCK_INPUT = r'''
EXPECTED_PREPARED_INPUT = __PREPARED_INPUT__
SHARD_INDEX = __SHARD_INDEX__
SHARD_START = __SHARD_START__
SHARD_END = __SHARD_END__

bundles = list(Path("/kaggle/input").rglob("prepared_bundle.kaggle"))
if len(bundles) != 1:
    raise RuntimeError(f"expected one prepared_bundle.kaggle, found {len(bundles)}")
bundle = bundles[0]
if sha256(bundle) != EXPECTED_PREPARED_INPUT["prepared_bundle_sha256"]:
    raise RuntimeError("frozen ADA prepared bundle hash mismatch")
input_dir = WORK / "input"
safe_extract_zip(bundle, input_dir)
prepared_manifest = input_dir / "prepared_manifest.csv"
receptor = input_dir / "receptor.pdbqt"
if sha256(prepared_manifest) != EXPECTED_PREPARED_INPUT["prepared_manifest_sha256"]:
    raise RuntimeError("frozen ADA prepared manifest hash mismatch")
if sha256(receptor) != EXPECTED_PREPARED_INPUT["receptor_pdbqt_sha256"]:
    raise RuntimeError("frozen ADA receptor PDBQT hash mismatch")
with prepared_manifest.open(newline="", encoding="utf-8") as handle:
    all_prepared = list(csv.DictReader(handle))
with (input_dir / "prep_failures.csv").open(newline="", encoding="utf-8") as handle:
    all_prep_failures = list(csv.DictReader(handle))
if len(all_prepared) + len(all_prep_failures) != 966:
    raise RuntimeError("frozen preparation accounting mismatch")
for row in all_prepared:
    index = int(row["sample_index"])
    if row["record_id"] != f"{row['source_id']}@{index}":
        raise RuntimeError(f"record identity mismatch at {index}")
    ligand = input_dir / row["pdbqt"]
    if not ligand.is_file() or sha256(ligand) != row["prepared_sha256"]:
        raise RuntimeError(f"prepared ligand hash mismatch at {index}")
prepared = [row for row in all_prepared if SHARD_START <= int(row["sample_index"]) < SHARD_END]
prep_failures = [row for row in all_prep_failures if SHARD_START <= int(row["sample_index"]) < SHARD_END]
if len(prepared) + len(prep_failures) != SHARD_END - SHARD_START:
    raise RuntimeError("shard prepared-input accounting mismatch")
pdbqt_files = [str(input_dir / row["pdbqt"]) for row in prepared]
input_manifest = {
    "target": "ADA", "pdb_id": "1NDW", "shard_index": SHARD_INDEX,
    "range": [SHARD_START, SHARD_END], "intended": SHARD_END - SHARD_START,
    "prepared": len(prepared), "preparation_failures": len(prep_failures),
    "prepared_bundle_sha256": sha256(bundle), "prepared_manifest_sha256": sha256(prepared_manifest),
    "receptor_pdbqt_sha256": sha256(receptor), "box_center": [42.2931, 17.3816, 49.5368],
    "box_size": [36.9618, 39.689, 38.3826], "grid_spacing": 0.375,
    "scoring": "vina", "exhaustiveness": 32, "lower_is_better": True,
}
Path("/kaggle/working/input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
write_csv(Path("/kaggle/working/prep_failures_assigned.csv"), prep_failures, ["record_id", "sample_index", "source_id", "label", "reason"])
'''


DOCK = r'''
docked = Path("/kaggle/working/docked")
docked.mkdir(exist_ok=True)
index_file = Path("/kaggle/working/ligand_index.txt")
index_file.write_text("\n".join(pdbqt_files) + "\n")
command = [
    "/usr/local/bin/unidock", "--receptor", str(receptor), "--ligand_index", str(index_file),
    "--dir", str(docked), "--center_x", "42.2931", "--center_y", "17.3816",
    "--center_z", "49.5368", "--size_x", "36.9618", "--size_y", "39.689",
    "--size_z", "38.3826", "--spacing", "0.375", "--scoring", "vina",
    "--exhaustiveness", "32",
]
print("Command: " + " ".join(command))
dock_started = time.time()
result = subprocess.run(command, capture_output=True, text=True)
dock_seconds = time.time() - dock_started
Path("/kaggle/working/unidock_vina.log").write_text(
    "Command: " + " ".join(command) + "\n\nSTDOUT\n" + (result.stdout or "") + "\nSTDERR\n" + (result.stderr or "")
)
poses = list(docked.glob("*_out.pdbqt"))
if result.returncode and not poses:
    raise RuntimeError(f"UniDock Vina failed without poses: {result.returncode}")

result_re = re.compile(r"REMARK VINA RESULT:\s+(\S+)\s+(\S+)\s+(\S+)")
scores, docking_failures = [], []
for row in prepared:
    index = int(row["sample_index"])
    pose = docked / f"{index:04d}_out.pdbqt"
    match = result_re.search(pose.read_text(errors="replace")) if pose.exists() else None
    if not match:
        docking_failures.append({
            "record_id": row["record_id"], "sample_index": index, "source_id": row["source_id"],
            "label": row["label"], "raw_score": "", "reason": "missing_pose_or_result",
        })
        continue
    try:
        score = float(match.group(1))
    except ValueError:
        score = math.nan
    if not math.isfinite(score):
        docking_failures.append({
            "record_id": row["record_id"], "sample_index": index, "source_id": row["source_id"],
            "label": row["label"], "raw_score": match.group(1), "reason": "non_finite_score",
        })
        continue
    scores.append({
        "record_id": row["record_id"], "sample_index": index, "source_id": row["source_id"],
        "label": row["label"], "score": score, "prep_method": row["prep_method"],
    })
scores.sort(key=lambda row: (row["score"], row["sample_index"]))
write_csv(Path("/kaggle/working/scores_vina.csv"), scores, ["record_id", "sample_index", "source_id", "label", "score", "prep_method"])
write_csv(Path("/kaggle/working/failed_vina.csv"), docking_failures, ["record_id", "sample_index", "source_id", "label", "raw_score", "reason"])
if len(scores) + len(docking_failures) + len(prep_failures) != SHARD_END - SHARD_START:
    raise RuntimeError("docking shard accounting mismatch")
with zipfile.ZipFile("/kaggle/working/docked_vina.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for pose in sorted(poses):
        archive.write(pose, pose.name)
runtime = {
    "stage": "docking", "target": "ADA", "pdb_id": "1NDW", "shard_index": SHARD_INDEX,
    "range": [SHARD_START, SHARD_END], "intended": SHARD_END - SHARD_START,
    "prepared": len(prepared), "preparation_failures": len(prep_failures),
    "scored": len(scores), "docking_failures": len(docking_failures),
    "fallbacks": sum(row["prep_method"] == "meeko_fallback" for row in prepared),
    "docking_seconds": dock_seconds, "total_seconds": time.time() - STARTED,
    "gpu_count": HARDWARE_PROVENANCE["gpu_count"], "gpu_models": HARDWARE_PROVENANCE["gpu_models"],
    "lower_is_better": True, "box_center": [42.2931, 17.3816, 49.5368],
    "box_size": [36.9618, 39.689, 38.3826], "grid_spacing": 0.375,
    "scoring": "vina", "exhaustiveness": 32, "grid_point_cap": 2097152,
    "unidock_commit": SOURCE_BUILD_PROVENANCE["git_commit"],
}
Path("/kaggle/working/runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
with zipfile.ZipFile("/kaggle/working/output.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in (
        "scores_vina.csv", "failed_vina.csv", "prep_failures_assigned.csv", "unidock_vina.log",
        "docked_vina.zip", "runtime.json", "hardware_manifest.json", "source_build_manifest.json",
        "input_manifest.json",
    ):
        archive.write(Path("/kaggle/working") / name, name)
Path("/kaggle/working/done.flag").write_text("ok\n")
print(runtime)
'''


def inject(source: str, replacements: dict[str, object]) -> str:
    for token, value in replacements.items():
        source = source.replace(token, repr(value))
    return source


def render_preparation() -> Path:
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    verify = inject(
        SOURCE_VERIFY,
        {
            "__SOURCE_MANIFEST__": source_manifest,
            "__TARGET_EVIDENCE__": evidence,
            "__CAMPAIGN_PROTOCOL__": protocol,
        },
    )
    cells = [
        "# ADA/1NDW full-library AutoDockTools preparation on Tesla T4x2",
        HARDWARE_GATE,
        COMMON_IMPORTS,
        verify,
        PREP_INSTALL,
        PREPARE,
    ]
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / "ada_1ndw_prepare_t4.ipynb"
    path.write_text(json.dumps(notebook(cells, stage="preparation"), indent=1) + "\n", encoding="utf-8")
    return path


def render_docking() -> list[Path]:
    if not PREPARED_INPUT_MANIFEST.is_file():
        return []
    prepared_input = json.loads(PREPARED_INPUT_MANIFEST.read_text(encoding="utf-8"))
    bounds = [(i * TOTAL // SHARDS, (i + 1) * TOTAL // SHARDS) for i in range(SHARDS)]
    paths = []
    for shard, (start, end) in enumerate(bounds):
        input_cell = inject(
            DOCK_INPUT,
            {
                "__PREPARED_INPUT__": prepared_input,
                "__SHARD_INDEX__": shard,
                "__SHARD_START__": start,
                "__SHARD_END__": end,
            },
        )
        cells = [
            f"# ADA/1NDW UniDock Vina-mode T4 shard {shard:02d}: [{start},{end})",
            HARDWARE_GATE,
            COMMON_IMPORTS,
            SOURCE_BUILD,
            input_cell,
            DOCK,
        ]
        path = NOTEBOOKS / f"ada_1ndw_vina_t4_shard_{shard:02d}.ipynb"
        path.write_text(json.dumps(notebook(cells, stage="docking"), indent=1) + "\n", encoding="utf-8")
        paths.append(path)
    plan = {
        "schema_version": 1,
        "target": "ADA",
        "pdb_id": "1NDW",
        "total": TOTAL,
        "shards": SHARDS,
        "bounds": [
            {"shard": shard, "start": start, "end": end, "count": end - start}
            for shard, (start, end) in enumerate(bounds)
        ],
        "hardware": {"gpu_count": 2, "model_token": "T4"},
        "scoring": "vina",
        "exhaustiveness": 32,
        "spacing": 0.375,
    }
    (HERE / "shard_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    prep = render_preparation()
    docking = render_docking()
    print(json.dumps({"preparation": str(prep), "docking_notebooks": len(docking)}, indent=2))


if __name__ == "__main__":
    main()
