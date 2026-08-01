"""Validate and freeze the exact ADA AutoDockTools output produced on T4."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREP_OUTPUT = HERE / "work/ada-1ndw-prepare-t4/output/output.zip"
INPUTS = HERE / "inputs"
FROZEN_BUNDLE = INPUTS / "prepared_bundle.zip"
FROZEN_MANIFEST = HERE / "prepared_input_manifest.json"

TOTAL, ACTIVES, DECOYS = 966, 39, 927
EXPECTED_SOURCE_RECEPTOR = (
    "dff0543204ea1fbf1586e718321fe4a62b8ed0637cc394eab29b74e404a82f95"
)
EXPECTED_XTAL = (
    "19df7dd93a16414d991d58719bd5b871454e8467d562ac433a8dbb879ba2e841"
)
EXPECTED_CENTER = [42.2931, 17.3816, 49.5368]
EXPECTED_SIZE = [36.9618, 39.689, 38.3826]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))


def main() -> None:
    if not PREP_OUTPUT.is_file():
        raise FileNotFoundError(PREP_OUTPUT)
    with zipfile.ZipFile(PREP_OUTPUT) as outer:
        required = {
            "prepared_bundle.zip",
            "prepared_manifest.csv",
            "prep_failures.csv",
            "fallbacks.csv",
            "input_manifest.json",
            "box_manifest.json",
            "receptor_normalization.json",
            "hardware_manifest.json",
            "runtime.json",
        }
        missing = required - set(outer.namelist())
        if missing:
            raise RuntimeError(f"preparation output missing: {sorted(missing)}")
        hardware = json.loads(outer.read("hardware_manifest.json"))
        if (
            not hardware.get("gate_passed")
            or hardware.get("gpu_count") != 2
            or any("T4" not in model for model in hardware.get("gpu_models", []))
        ):
            raise RuntimeError(f"preparation hardware gate failed: {hardware}")
        input_manifest = json.loads(outer.read("input_manifest.json"))
        box = json.loads(outer.read("box_manifest.json"))
        runtime = json.loads(outer.read("runtime.json"))
        normalization = json.loads(outer.read("receptor_normalization.json"))
        prepared = rows(outer, "prepared_manifest.csv")
        failures = rows(outer, "prep_failures.csv")
        fallbacks = rows(outer, "fallbacks.csv")
        bundle_bytes = outer.read("prepared_bundle.zip")

    if input_manifest["source_receptor_sha256"] != EXPECTED_SOURCE_RECEPTOR:
        raise RuntimeError("source receptor hash mismatch")
    if input_manifest["crystal_ligand_sha256"] != EXPECTED_XTAL:
        raise RuntimeError("crystal ligand hash mismatch")
    if box["center"] != EXPECTED_CENTER or box["size"] != EXPECTED_SIZE:
        raise RuntimeError("frozen ADA box mismatch")
    if (
        normalization.get("source_receptor_sha256") != EXPECTED_SOURCE_RECEPTOR
        or normalization.get("atoms_added") != 0
        or normalization.get("atoms_removed") != 0
        or normalization.get("coordinates_changed") != 0
        or len(normalization.get("normalizations", [])) != 1
        or normalization["normalizations"][0].get("atom_name") != "ZN"
    ):
        raise RuntimeError("ADA receptor normalization audit failed")
    if len(prepared) + len(failures) != TOTAL:
        raise RuntimeError("preparation accounting mismatch")
    labels = Counter(int(row["label"]) for row in prepared + failures)
    if labels != Counter({0: DECOYS, 1: ACTIVES}):
        raise RuntimeError(f"label count mismatch: {labels}")
    indices = [int(row["sample_index"]) for row in prepared + failures]
    if sorted(indices) != list(range(TOTAL)) or len(indices) != len(set(indices)):
        raise RuntimeError("sample index accounting mismatch")
    for row in prepared + failures:
        if row["record_id"] != f"{row['source_id']}@{int(row['sample_index'])}":
            raise RuntimeError(f"record identity mismatch: {row}")

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
        bundle_names = set(bundle.namelist())
        if {
            "receptor.pdbqt",
            "prepared_manifest.csv",
            "prep_failures.csv",
            "fallbacks.csv",
            "input_manifest.json",
            "box_manifest.json",
            "receptor_normalization.json",
        } - bundle_names:
            raise RuntimeError("prepared bundle audit files missing")
        receptor_bytes = bundle.read("receptor.pdbqt")
        internal_manifest = bundle.read("prepared_manifest.csv")
        for row in prepared:
            name = row["pdbqt"]
            if name not in bundle_names:
                raise RuntimeError(f"prepared ligand missing: {name}")
            if sha256_bytes(bundle.read(name)) != row["prepared_sha256"]:
                raise RuntimeError(f"prepared ligand hash mismatch: {name}")

    INPUTS.mkdir(parents=True, exist_ok=True)
    FROZEN_BUNDLE.write_bytes(bundle_bytes)
    frozen = {
        "schema_version": 1,
        "target": "ADA",
        "pdb_id": "1NDW",
        "prepared_on_required_hardware": True,
        "hardware": {
            "gpu_count": hardware["gpu_count"],
            "gpu_models": hardware["gpu_models"],
            "query_stdout": hardware["query_stdout"],
        },
        "records": {
            "intended": TOTAL,
            "actives": ACTIVES,
            "decoys": DECOYS,
            "prepared": len(prepared),
            "preparation_failures": len(failures),
            "fallbacks": len(fallbacks),
        },
        "prepared_bundle_sha256": sha256_bytes(bundle_bytes),
        "prepared_bundle_bytes": len(bundle_bytes),
        "prepared_manifest_sha256": sha256_bytes(internal_manifest),
        "receptor_pdbqt_sha256": sha256_bytes(receptor_bytes),
        "source_receptor_sha256": EXPECTED_SOURCE_RECEPTOR,
        "normalized_receptor_sha256": normalization["normalized_receptor_sha256"],
        "receptor_normalization": normalization,
        "crystal_ligand_sha256": EXPECTED_XTAL,
        "box_center": EXPECTED_CENTER,
        "box_size": EXPECTED_SIZE,
        "prep_output_zip_sha256": sha256(PREP_OUTPUT),
        "prep_runtime": runtime,
        "record_identity": "source_id@sample_index",
    }
    FROZEN_MANIFEST.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
