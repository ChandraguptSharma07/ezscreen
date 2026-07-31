"""Render ACE UniDock Vina/Vinardo GPU shards with AD4-byte-identical prep."""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AD4_DIR = HERE.parent / "full_ad4_port"
sys.path.insert(0, str(AD4_DIR))
import render_full_ad4_shards as ad4  # noqa: E402

TOTAL, SHARDS = 1846, 12

DOCK = r'''
import json, pathlib, shutil, subprocess, time

BOX_CENTER = [40.7317, 32.8822, 47.3635]
BOX_SIZE = [36.8194, 41.835, 43.7048]
EXHAUSTIVENESS = 32
_unidock = shutil.which("unidock")
if not _unidock: raise RuntimeError("unidock binary missing")

MODE_RUNS = {}
for scoring in ("vina", "vinardo"):
    docked = pathlib.Path("/kaggle/working/docked") / scoring
    docked.mkdir(parents=True, exist_ok=True)
    index_file = pathlib.Path(f"/kaggle/working/ligand_index_{scoring}.txt")
    index_file.write_text("\n".join(pdbqt_files) + "\n")
    command = [
        _unidock, "--receptor", str(RECEPTOR_PDBQT),
        "--ligand_index", str(index_file), "--dir", str(docked),
        "--center_x", str(BOX_CENTER[0]), "--center_y", str(BOX_CENTER[1]),
        "--center_z", str(BOX_CENTER[2]), "--size_x", str(BOX_SIZE[0]),
        "--size_y", str(BOX_SIZE[1]), "--size_z", str(BOX_SIZE[2]),
        "--scoring", scoring, "--exhaustiveness", str(EXHAUSTIVENESS),
    ]
    print("Command: " + " ".join(command))
    mode_started = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    seconds = time.time() - mode_started
    poses = list(docked.glob("*_out.pdbqt"))
    pathlib.Path(f"/kaggle/working/unidock_{scoring}.log").write_text(
        "Command: " + " ".join(command) + "\n\nSTDOUT\n" +
        (result.stdout or "") + "\nSTDERR\n" + (result.stderr or "")
    )
    MODE_RUNS[scoring] = {
        "command": command, "returncode": result.returncode,
        "runtime_seconds": seconds, "poses": len(poses),
    }
    print({"scoring": scoring, "returncode": result.returncode,
           "poses": len(poses), "runtime_seconds": seconds})
    if result.returncode and not poses:
        raise RuntimeError(f"{scoring} docking failed without poses")
'''

RESULTS = r'''
import math, re, zipfile

result_re = re.compile(r"REMARK VINA RESULT:\s+(\S+)\s+(\S+)\s+(\S+)")
by_index = {int(row["sample_index"]): row for row in prepared}
mode_audit = {}
output_names = ["prepared_manifest.csv", "prep_failures.csv", "fallbacks.csv",
                "prep_messages.jsonl", "prepared_identity_check.json"]
for scoring in ("vina", "vinardo"):
    scores, failed = [], []
    docked = pathlib.Path("/kaggle/working/docked") / scoring
    for index, meta in by_index.items():
        pose = docked / f"{index:04d}_out.pdbqt"
        match = result_re.search(pose.read_text(errors="replace")) if pose.exists() else None
        if not match:
            failed.append({"sample_index": index, "source_id": meta["source_id"],
                           "label": meta["label"], "raw_score": "",
                           "reason": "missing_pose_or_result"})
            continue
        try: score = float(match.group(1))
        except ValueError: score = math.nan
        if not math.isfinite(score):
            failed.append({"sample_index": index, "source_id": meta["source_id"],
                           "label": meta["label"], "raw_score": match.group(1),
                           "reason": "non_finite_score"})
            continue
        scores.append({"sample_index": index, "source_id": meta["source_id"],
                       "label": meta["label"], "score": score,
                       "prep_method": meta["prep_method"]})
    scores.sort(key=lambda row: (row["score"], row["sample_index"]))
    write_csv(f"scores_{scoring}.csv", scores,
              ["sample_index", "source_id", "label", "score", "prep_method"])
    write_csv(f"failed_{scoring}.csv", failed,
              ["sample_index", "source_id", "label", "raw_score", "reason"])
    if len(scores) + len(failed) + len(prep_failures) != len(selected):
        raise RuntimeError(f"{scoring} accounting mismatch")
    pose_zip = f"docked_{scoring}.zip"
    with zipfile.ZipFile(pathlib.Path("/kaggle/working") / pose_zip, "w",
                         zipfile.ZIP_DEFLATED) as archive:
        for pose in sorted(docked.glob("*_out.pdbqt")): archive.write(pose, pose.name)
    mode_audit[scoring] = {**MODE_RUNS[scoring], "scored": len(scores),
                           "failures": len(failed)}
    output_names += [f"scores_{scoring}.csv", f"failed_{scoring}.csv",
                     f"unidock_{scoring}.log", pose_zip]

runtime = {
    "shard_index": SHARD_INDEX, "range": [SHARD_START, SHARD_END],
    "intended": len(selected), "prepared": len(prepared),
    "prep_failures": len(prep_failures), "fallbacks": len(fallbacks),
    "prepared_identity_passed": True, "modes": mode_audit,
    "total_seconds": time.time() - STARTED, "lower_is_better": True,
    "protocol": "UniDock Vina and Vinardo accelerated ports; not Vina-equivalent",
    "box_center": BOX_CENTER, "box_size": BOX_SIZE,
    "exhaustiveness": EXHAUSTIVENESS,
}
pathlib.Path("/kaggle/working/runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
output_names.append("runtime.json")
with zipfile.ZipFile("/kaggle/working/output.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in output_names:
        path = pathlib.Path("/kaggle/working") / name
        if path.exists(): archive.write(path, name)
pathlib.Path("/kaggle/working/done.flag").write_text("ok\n")
print(runtime)
'''


def expected_prepared(shard: int) -> list[dict[str, str]]:
    output = AD4_DIR / "work" / f"ace-1o86-ad4-full-s{shard:02d}" / "output" / "output.zip"
    if not output.is_file():
        raise FileNotFoundError(output)
    with zipfile.ZipFile(output) as archive:
        with archive.open("prepared_manifest.csv") as handle:
            rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    return [
        {key: row[key] for key in (
            "sample_index", "source_id", "label", "prep_method",
            "source_sha256", "prepared_sha256",
        )}
        for row in rows
    ]


def render() -> None:
    ace = json.loads(ad4.ACE_NB.read_text(encoding="utf-8"))
    ad4_notebook = json.loads(ad4.AD4_NB.read_text(encoding="utf-8"))
    imports = "".join(ace["cells"][1]["source"])
    manifest = "".join(ace["cells"][3]["source"])
    verify = "".join(ace["cells"][4]["source"])
    verify = verify.replace(
        'for item in manifest["sources"]:',
        'for item in [x for x in manifest["sources"] if x["role"] in {"active_libraries", "decoy_libraries", "autodocktools_runtime"}]:',
    )
    install = "".join(ad4_notebook["cells"][1]["source"]).replace(
        "_note_optional_tool('obabel', 'used only for fallback ligand-format conversion')",
        "_apt_install('openbabel')\n_require_tools('obabel')",
    )
    out = HERE / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    bounds = [(i * TOTAL // SHARDS, (i + 1) * TOTAL // SHARDS) for i in range(SHARDS)]
    for shard, (start, end) in enumerate(bounds):
        expected = expected_prepared(shard)
        if len(expected) != end - start:
            raise RuntimeError(f"shard {shard}: AD4 manifest count mismatch")
        setup = imports.replace("N_SAMPLE = 100", f"N_SAMPLE = {end - start}")
        setup = setup.replace(
            'WORK = Path("/kaggle/working/vinardo_preflight")',
            'WORK = Path("/kaggle/working/ace_vina_vinardo_full")',
        ).replace(
            'SCRATCH = Path("/kaggle/temp/vinardo_preflight")',
            'SCRATCH = Path("/kaggle/temp/ace_vina_vinardo_full")',
        )
        setup += (
            f"\nSHARD_INDEX={shard}\nSHARD_START={start}\nSHARD_END={end}"
            "\nSTARTED=time.time()\n"
        )
        receptor_setup = r'''
import pathlib
_candidates = list(pathlib.Path("/kaggle/input").rglob("receptor.pdbqt"))
if len(_candidates) != 1: raise RuntimeError(f"expected exact receptor once, found {len(_candidates)}")
RECEPTOR_PDBQT = _candidates[0]
EXPECTED_RECEPTOR_SHA256 = "7c3e148c6c094ffc5bb5d10ddb7184983f0248590712a1ed7c82f3a752375585"
if sha256(RECEPTOR_PDBQT) != EXPECTED_RECEPTOR_SHA256:
    raise RuntimeError("exact ADT receptor hash mismatch")
'''
        identity = (
            "EXPECTED_PREPARED = " + repr(expected) + "\n" + r'''
expected_by_index = {int(row["sample_index"]): row for row in EXPECTED_PREPARED}
actual_by_index = {int(row["sample_index"]): row for row in prepared}
if set(actual_by_index) != set(expected_by_index):
    raise RuntimeError("prepared index set differs from AD4 campaign")
identity_fields = ("source_id", "label", "prep_method", "source_sha256", "prepared_sha256")
mismatches = []
for index, expected_row in expected_by_index.items():
    actual_row = actual_by_index[index]
    for field in identity_fields:
        if str(actual_row[field]) != str(expected_row[field]):
            mismatches.append({"sample_index": index, "field": field,
                               "expected": expected_row[field], "actual": actual_row[field]})
if mismatches: raise RuntimeError(f"prepared bytes differ from AD4 campaign: {mismatches[:3]}")
identity_check = {"passed": True, "records": len(actual_by_index),
                  "fields": list(identity_fields),
                  "reference": f"ACE AD4 attempt 04 shard {SHARD_INDEX:02d}"}
Path("/kaggle/working/prepared_identity_check.json").write_text(
    json.dumps(identity_check, indent=2) + "\n")
print(identity_check)
'''
        )
        sources = [
            f"# ACE Vina/Vinardo shard {shard:02d}: [{start},{end})\n",
            install, setup, manifest, verify, ad4.INSTALL_ADT, ad4.PREP,
            identity, receptor_setup, DOCK, RESULTS,
        ]
        cells = [ad4.cell(source) for source in sources]
        for item in cells:
            source = "".join(item["source"])
            if source.strip():
                compile(source, f"ace_vina_vinardo_{shard:02d}.py", "exec")
        notebook = {
            "cells": cells, "metadata": ad4_notebook["metadata"],
            "nbformat": ad4_notebook["nbformat"],
            "nbformat_minor": ad4_notebook["nbformat_minor"],
        }
        (out / f"ace_vina_vinardo_shard_{shard:02d}.ipynb").write_text(
            json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
        )
    plan = {
        "total": TOTAL, "shards": SHARDS,
        "modes": ["vina", "vinardo"], "exhaustiveness": 32,
        "bounds": [
            {"shard": i, "start": a, "end": b, "count": b - a}
            for i, (a, b) in enumerate(bounds)
        ],
    }
    (HERE / "shard_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    render()
