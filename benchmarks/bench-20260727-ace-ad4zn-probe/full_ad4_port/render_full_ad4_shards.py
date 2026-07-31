"""Render 12 ACE DUD2 UniDock AD4 GPU shards with GPU-side ADT prep."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ACE_NB = ROOT / "benchmarks/bench-20260726-ace-1o86/notebooks/ace_smoke.ipynb"
AD4_NB = ROOT / "benchmarks/bench-20260727-ace-ad4zn-probe/work/unidock_ad4_smoke_020_kernel/notebook.ipynb"
TOTAL, SHARDS = 1846, 12


def cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


INSTALL_ADT = r'''
def run(command, *, cwd=None, timeout=7200, check=True):
    p = subprocess.run([str(x) for x in command], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {command}\n{(p.stdout or '')[-3000:]}")
    return p

mgl_unpack = SCRATCH / "mgl_dist"
safe_extract(SOURCE_FILES["mgltools_x86_64Linux2_1.5.6-patch1.tar.gz"], mgl_unpack)
mgl_dist = next(mgl_unpack.glob("mgltools_*"))
mgl_root = SCRATCH / "mgltools"
run(["bash", "install.sh", "-d", str(mgl_root)], cwd=mgl_dist)
pythonsh = mgl_root / "bin/pythonsh"
utilities = mgl_root / "MGLToolsPckgs/AutoDockTools/Utilities24"
if not pythonsh.exists() or not (utilities / "prepare_ligand4.py").exists():
    raise RuntimeError("MGLTools/ADT install incomplete")
'''

PREP = r'''
data_dir = SCRATCH / "dud"
safe_extract(SOURCE_FILES["dud_ligands2006.tar.gz"], data_dir)
safe_extract(SOURCE_FILES["dud_decoys2006.tar.gz"], data_dir)

def mol2_records(path):
    marker = b"@<TRIPOS>MOLECULE"
    with gzip.open(path, "rb") as handle: payload = handle.read()
    return [marker + part for part in payload.split(marker)[1:]]

def molecule_id(record):
    return record.decode("utf-8", errors="replace").splitlines()[1].strip()

actives = mol2_records(data_dir / "dud_ligands2006/ace_ligands.mol2.gz")
decoys = mol2_records(data_dir / "dud_decoys2006/ace_decoys.mol2.gz")
if (len(actives), len(decoys)) != (49, 1797):
    raise RuntimeError(f"ACE DUD2 count mismatch: {(len(actives), len(decoys))}")
all_records = [(record, 1) for record in actives] + [(record, 0) for record in decoys]
selected = list(enumerate(all_records))[SHARD_START:SHARD_END]
if len(selected) != SHARD_END - SHARD_START: raise RuntimeError("shard slice mismatch")

prep_dir, source_dir = WORK / "prepared", WORK / "source"
prep_dir.mkdir(exist_ok=True); source_dir.mkdir(exist_ok=True)
prepared, prep_failures, fallbacks, prep_messages = [], [], [], []
for index, (record, label) in selected:
    source_id = molecule_id(record)
    mol2, pdbqt = source_dir / f"{index:04d}.mol2", prep_dir / f"{index:04d}.pdbqt"
    mol2.write_bytes(record)
    adt = run([pythonsh, utilities / "prepare_ligand4.py", "-l", mol2, "-o", pdbqt], check=False)
    method = "autodocktools"
    prep_messages.append({"sample_index": index, "method": method, "returncode": adt.returncode,
                          "tail": (adt.stdout or "")[-1000:]})
    if adt.returncode or not pdbqt.exists() or not pdbqt.stat().st_size:
        method = "meeko_fallback"
        sdf = source_dir / f"{index:04d}.sdf"
        convert = run(["obabel", "-imol2", mol2, "-osdf", "-O", sdf], check=False)
        mk_base = ([shutil.which("mk_prepare_ligand")] if shutil.which("mk_prepare_ligand")
                   else [sys.executable, "-m", "meeko.cli.mk_prepare_ligand"])
        mk = run(mk_base + ["-i", sdf, "-o", pdbqt], check=False) if convert.returncode == 0 and sdf.exists() else None
        fallbacks.append({"sample_index": index, "source_id": source_id, "label": label,
                          "adt_returncode": adt.returncode, "conversion_returncode": convert.returncode,
                          "meeko_returncode": None if mk is None else mk.returncode})
        if mk is None or mk.returncode or not pdbqt.exists() or not pdbqt.stat().st_size:
            prep_failures.append({"sample_index": index, "source_id": source_id,
                                  "label": label, "reason": "adt_then_meeko_failed"})
            continue
    clean = "\n".join(x for x in pdbqt.read_text(errors="replace").splitlines() if x.strip() != "END") + "\n"
    pdbqt.write_text(clean)
    prepared.append({"sample_index": index, "source_id": source_id, "label": label,
                     "prep_method": method, "source_sha256": sha256(mol2),
                     "prepared_sha256": sha256(pdbqt), "pdbqt": str(pdbqt)})

def write_csv(name, rows, fields):
    with (Path("/kaggle/working") / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

write_csv("prepared_manifest.csv", prepared, ["sample_index","source_id","label","prep_method","source_sha256","prepared_sha256","pdbqt"])
write_csv("prep_failures.csv", prep_failures, ["sample_index","source_id","label","reason"])
write_csv("fallbacks.csv", fallbacks, ["sample_index","source_id","label","adt_returncode","conversion_returncode","meeko_returncode"])
with open("/kaggle/working/prep_messages.jsonl", "w") as handle:
    for row in prep_messages: handle.write(json.dumps(row) + "\n")
pdbqt_files = [row["pdbqt"] for row in prepared]
print({"intended": len(selected), "prepared": len(prepared), "prep_failures": len(prep_failures), "fallbacks": len(fallbacks)})
'''

RESULTS = r'''
import math, zipfile
result_re = re.compile(r"REMARK VINA RESULT:\s+(\S+)\s+(\S+)\s+(\S+)")
by_index = {int(row["sample_index"]): row for row in prepared}
docked_dir = RECEPTOR_DOCKED_DIRS["default"]
scores, failed = [], []
for index, meta in by_index.items():
    pose = docked_dir / f"{index:04d}_out.pdbqt"
    match = result_re.search(pose.read_text(errors="replace")) if pose.exists() else None
    if not match:
        failed.append({"sample_index": index, "source_id": meta["source_id"], "label": meta["label"],
                       "raw_score": "", "reason": "missing_pose_or_result"})
        continue
    try: score = float(match.group(1))
    except ValueError: score = math.nan
    if not math.isfinite(score):
        failed.append({"sample_index": index, "source_id": meta["source_id"], "label": meta["label"],
                       "raw_score": match.group(1), "reason": "non_finite_score"})
        continue
    scores.append({"sample_index": index, "source_id": meta["source_id"], "label": meta["label"],
                   "score": score, "prep_method": meta["prep_method"]})
scores.sort(key=lambda row: (row["score"], row["sample_index"]))
write_csv("scores.csv", scores, ["sample_index","source_id","label","score","prep_method"])
write_csv("failed_docking.csv", failed, ["sample_index","source_id","label","raw_score","reason"])
accounted = len(scores) + len(failed) + len(prep_failures)
if accounted != len(selected): raise RuntimeError(f"accounting mismatch {accounted}/{len(selected)}")
runtime = {"shard_index": SHARD_INDEX, "range": [SHARD_START, SHARD_END], "intended": len(selected),
           "prepared": len(prepared), "scored": len(scores), "prep_failures": len(prep_failures),
           "docking_failures": len(failed), "fallbacks": len(fallbacks),
           "total_seconds": time.time() - STARTED, "lower_is_better": True,
           "protocol": "UniDock AD4 accelerated port; official AD4Zn maps; 0.52 A spacing"}
Path("/kaggle/working/runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
with zipfile.ZipFile("/kaggle/working/docked_pdbqt.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for pose in sorted(docked_dir.glob("*_out.pdbqt")): archive.write(pose, pose.name)
names = ["scores.csv","prepared_manifest.csv","prep_failures.csv","fallbacks.csv","failed_docking.csv",
         "prep_messages.jsonl","ad4_map_runtime.json","runtime.json","docked_pdbqt.zip"]
with zipfile.ZipFile("/kaggle/working/output.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        path = Path("/kaggle/working") / name
        if path.exists(): archive.write(path, name)
Path("/kaggle/working/done.flag").write_text("ok\n")
print(runtime)
'''


def render() -> None:
    ace = json.loads(ACE_NB.read_text(encoding="utf-8"))
    ad4 = json.loads(AD4_NB.read_text(encoding="utf-8"))
    imports = "".join(ace["cells"][1]["source"])
    manifest = "".join(ace["cells"][3]["source"])
    verify = "".join(ace["cells"][4]["source"])
    verify = verify.replace(
        'for item in manifest["sources"]:',
        'for item in [x for x in manifest["sources"] if x["role"] in {"active_libraries", "decoy_libraries", "autodocktools_runtime"}]:'
    )
    install = "".join(ad4["cells"][1]["source"]).replace(
        "_note_optional_tool('obabel', 'used only for fallback ligand-format conversion')",
        "_apt_install('openbabel')\n_require_tools('obabel')")
    dock = "".join(ad4["cells"][6]["source"])
    dock = dock.replace(
        '_required = [f"{AD4_MAPS_PREFIX_NAME}.{_atom}.map" for _atom in ("A", "C", "HD", "N", "NA", "OA", "SA", "d", "e")]',
        '_ligand_types = {line.split()[-1] for ligand in pdbqt_files for line in pathlib.Path(ligand).read_text(errors="replace").splitlines() if line.startswith(("ATOM", "HETATM"))}\n        _required = [f"{AD4_MAPS_PREFIX_NAME}.{_atom}.map" for _atom in sorted(_ligand_types | {"d", "e"})]')
    if "_ligand_types =" not in dock: raise RuntimeError("dynamic AD4 map patch missing")
    out = HERE / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    bounds = [(i * TOTAL // SHARDS, (i + 1) * TOTAL // SHARDS) for i in range(SHARDS)]
    for shard, (start, end) in enumerate(bounds):
        setup = imports.replace("N_SAMPLE = 100", f"N_SAMPLE = {end-start}")
        setup = setup.replace('WORK = Path("/kaggle/working/vinardo_preflight")',
                              'WORK = Path("/kaggle/working/ace_ad4_full")')
        setup = setup.replace('SCRATCH = Path("/kaggle/temp/vinardo_preflight")',
                              'SCRATCH = Path("/kaggle/temp/ace_ad4_full")')
        setup += f"\nSHARD_INDEX={shard}\nSHARD_START={start}\nSHARD_END={end}\nSTARTED=time.time()\n"
        receptor_setup = """
import pathlib, zipfile
_mounted_input = pathlib.Path("/kaggle/input")
_map_zips = list(_mounted_input.rglob("ad4_map_bundle.zip"))
_mounted_manifests = list(_mounted_input.rglob("ad4_maps_manifest.json"))
if len(_map_zips) == 1 and not _mounted_manifests:
    _input_root = WORK / "maps"
    _input_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(_map_zips[0]) as _archive: _archive.extractall(_input_root)
elif not _map_zips and len(_mounted_manifests) == 1:
    _input_root = _mounted_manifests[0].parent
else:
    raise RuntimeError(f"ambiguous map mount: zip={len(_map_zips)}, manifest={len(_mounted_manifests)}")
_candidates = list(_mounted_input.rglob("receptor.pdbqt"))
if len(_candidates) != 1: raise RuntimeError(f"expected exact receptor once, found {len(_candidates)}")
RECEPTOR_PDBQT = _candidates[0]
RECEPTOR_PDBQTS = {"default": RECEPTOR_PDBQT}
def _variant_slug(name): return str(name)
"""
        sources = [f"# ACE full AD4 shard {shard:02d}: [{start},{end})\n", install, setup,
                   manifest, verify, INSTALL_ADT, PREP, receptor_setup, dock, RESULTS]
        cells = [cell(source) for source in sources]
        for item in cells:
            source = "".join(item["source"])
            if source.strip(): compile(source, f"ace_ad4_shard_{shard:02d}.py", "exec")
        notebook = {"cells": cells, "metadata": ad4["metadata"],
                    "nbformat": ad4["nbformat"], "nbformat_minor": ad4["nbformat_minor"]}
        (out / f"ace_ad4_full_shard_{shard:02d}.ipynb").write_text(
            json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    plan = {"total": TOTAL, "shards": SHARDS,
            "bounds": [{"shard": i, "start": a, "end": b, "count": b-a}
                       for i,(a,b) in enumerate(bounds)]}
    (HERE / "shard_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    render()
