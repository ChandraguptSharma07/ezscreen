from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ezscreen.benchmark.early_enrichment import (
    ProtocolContractError,
    preparation_contract,
    validate_protocol_freeze,
)

PREPARATION_ARM = {
    "B0_frozen_baseline": "B0_frozen_baseline",
    "L1_meeko_single_state": "L1_meeko_single_state",
    "L2_bounded_state_ensemble": "L2_bounded_state_ensemble",
    "S1_budget_matched_replicates": "B0_frozen_baseline",
    "C1_states_plus_replicates": "L2_bounded_state_ensemble",
}
SOURCE_FIELDS = (
    "source_key",
    "source_id",
    "sample_index",
    "label",
    "source_sha256",
    "mol2",
)
PREPARED_FIELDS = (
    "source_key",
    "source_id",
    "sample_index",
    "label",
    "state_index",
    "ligand",
    "preparation_status",
    "failure_reason",
    "source_sha256",
    "prepared_sha256",
    "pdbqt",
    "prep_arm",
    "coordinate_max_drift_angstrom",
    "mmff_convergence_code",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_cell(source: str) -> dict[str, Any]:
    compile(source, "early_enrichment_preparation_cell.py", "exec")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def render_t4_preparation_notebook(contract: Mapping[str, Any]) -> str:
    """Render output-free guarded preparation notebook; never submits it."""
    frozen = json.dumps(contract, sort_keys=True)
    cells = [
        _markdown_cell(
            """# SwiftScreen UniDock/Vina early-enrichment preparation

Purpose: execute one frozen ligand-preparation arm on verified Kaggle Tesla T4
hardware. This notebook performs no docking and refuses work unless a separate
execution authorization is present in the frozen contract."""
        ),
        _code_cell(
            """from __future__ import annotations

import json
import pathlib
import subprocess

CONTRACT = json.loads(r'''"""
            + frozen
            + """''')
if CONTRACT["execution"]["launch_authorized"] is not True:
    raise RuntimeError("Preparation execution is not authorized")

query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
)
if query.returncode:
    raise RuntimeError(f"Tesla T4 required; nvidia-smi failed: {query.stderr[-500:]}")
gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
gpu_models = [
    row.split(",")[1].strip() for row in gpu_rows if len(row.split(",")) >= 2
]
if not gpu_rows or len(gpu_models) != len(gpu_rows):
    raise RuntimeError(f"GPU inventory could not be parsed: {gpu_rows!r}")
if any("T4" not in model for model in gpu_models):
    raise RuntimeError(f"Tesla T4 only; observed: {gpu_models!r}")
pathlib.Path("/kaggle/working/hardware_manifest.json").write_text(
    json.dumps(
        {
            "gate_passed": True,
            "gpu_count": len(gpu_rows),
            "gpu_models": gpu_models,
            "query_stdout": query.stdout,
        },
        indent=2,
    )
    + "\\n"
)
"""
        ),
        _code_cell(
            """import hashlib
import pathlib
import zipfile

transport = CONTRACT.get("transport")
if transport:
    archive_name = transport["filename"]
    matches = list(pathlib.Path("/kaggle/input").rglob(archive_name))
    if len(matches) == 1:
        archive_path = matches[0]
        digest = hashlib.sha256()
        with archive_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != transport["sha256"]:
            raise RuntimeError("Preparation input archive hash mismatch")
        destination = pathlib.Path(
            "/kaggle/temp/swiftscreen_early_enrichment_input"
        )
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(archive_path) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(
                        f"Unsafe preparation archive member: {member.filename}"
                    )
            bundle.extractall(destination)
    elif len(matches) == 0:
        contracts = list(
            pathlib.Path("/kaggle/input").rglob("preparation_contract.json")
        )
        if len(contracts) != 1:
            raise RuntimeError(
                f"Expected one expanded preparation contract, found {len(contracts)}"
            )
        expanded_root = contracts[0].parent
        member_hashes = transport.get("members_sha256", {})
        if not member_hashes:
            raise RuntimeError("Expanded preparation transport lacks member hashes")
        for relative_name, expected_hash in member_hashes.items():
            member_path = expanded_root / relative_name
            if not member_path.is_file():
                raise RuntimeError(
                    f"Expanded preparation member missing: {relative_name}"
                )
            digest = hashlib.sha256()
            with member_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise RuntimeError(
                    f"Expanded preparation member hash mismatch: {relative_name}"
                )
    else:
        raise RuntimeError(
            f"Expected at most one preparation input archive, found {len(matches)}"
        )
"""
        ),
        _code_cell(
            """import json
import pathlib
import subprocess
import sys

versions = CONTRACT["toolchain"]
packages = [
    f"openbabel-wheel=={versions['openbabel_wheel']}",
    f"rdkit=={versions['rdkit']}",
    f"meeko=={versions['meeko']}",
    f"gemmi=={versions['gemmi']}",
]
if CONTRACT["preparation_arm"] == "L2_bounded_state_ensemble":
    packages.extend(
        [
            f"gypsum-dl=={versions['gypsum_dl']}",
            f"dimorphite-dl=={versions['dimorphite_dl']}",
        ]
    )
install_command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-input",
    *packages,
]
install = subprocess.run(
    install_command,
    capture_output=True,
    text=True,
)
pathlib.Path("/kaggle/working/bootstrap_commands.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "commands": [
                {
                    "sequence": "bootstrap-0",
                    "argv": install_command,
                    "cwd": None,
                    "timeout_seconds": None,
                    "return_code": install.returncode,
                    "status": "complete",
                    "output_tail": (install.stdout + "\\n" + install.stderr)[-2000:],
                }
            ],
        },
        indent=2,
    )
    + "\\n"
)
if install.returncode:
    raise RuntimeError(
        f"Pinned dependency install failed: return code {install.returncode}: "
        f"{install.stderr[-500:]}"
    )
"""
        ),
        _code_cell(
            """import runpy
from pathlib import Path

runtime_candidates = []
for root in (
    Path("/kaggle/input"),
    Path("/kaggle/temp/swiftscreen_early_enrichment_input"),
):
    runtime_candidates.extend(root.rglob("early_enrichment_prep_runtime.py"))
if len(runtime_candidates) != 1:
    raise RuntimeError(
        f"Expected one preparation runtime, found {len(runtime_candidates)}"
    )
runpy.run_path(str(runtime_candidates[0]), run_name="__main__")
"""
        ),
        _markdown_cell(
            """## Expected outputs

- prepared_bundle.zip
- prepared_manifest.csv
- preparation_failures.csv
- preparation_runtime.json
- tool_versions.json
- hardware_manifest.json
- pip_freeze.txt
- environment_manifest.json
- executable_manifest.json
- command_ledger.json
- artifact_manifest.json
- archive_manifest.json

All code-cell outputs intentionally remain empty in repository artifacts."""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "swiftscreen": {
                "study_id": contract["study_id"],
                "stage": "ligand_preparation",
                "required_accelerator": "Tesla T4",
                "launch_authorized": contract["execution"]["launch_authorized"],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def pack_t4_preparation_dataset(
    output_dir: Path,
    *,
    archive_name: str = "preparation_input_bundle.zip",
) -> dict[str, Any]:
    """Pack one staged bundle for reliable private Kaggle transport."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "bundle_manifest.json"
    contract_path = output_dir / "preparation_contract.json"
    notebook_path = output_dir / "prepare_t4.ipynb"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise ProtocolContractError("staged preparation bundle is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("execution", {}).get("launch_authorized") is not True:
        raise ProtocolContractError("cannot pack an unauthorized preparation bundle")
    archive_path = output_dir / archive_name
    if archive_path.exists():
        raise ProtocolContractError("preparation transport archive already exists")
    members = [
        contract_path,
        output_dir / "intended_sources.csv",
        output_dir / "early_enrichment_prep_runtime.py",
        *sorted((output_dir / "source_mol2").glob("*.mol2")),
    ]
    runtime_filename = str(contract["toolchain"]["mgltools_runtime_filename"])
    runtime_path = output_dir / runtime_filename
    if runtime_path.is_file():
        members.append(runtime_path)
    if any(not path.is_file() for path in members):
        raise ProtocolContractError("preparation transport member is missing")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(output_dir).as_posix())
    transport = {
        "format": "zip",
        "filename": archive_path.name,
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
        "members_sha256": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in members
        },
    }
    notebook_contract = json.loads(json.dumps(contract))
    notebook_contract["transport"] = transport
    notebook_path.write_text(
        render_t4_preparation_notebook(notebook_contract), encoding="utf-8"
    )
    manifest["transport"] = transport
    manifest["files_sha256"][notebook_path.name] = _sha256(notebook_path)
    manifest["files_sha256"][archive_path.name] = transport["sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "archive": str(archive_path),
        "notebook": str(notebook_path),
        "transport": transport,
    }


def refresh_t4_preparation_transport(output_dir: Path) -> dict[str, Any]:
    """Refresh notebook transport metadata without changing archived inputs."""
    output_dir = Path(output_dir)
    archive_path = output_dir / "preparation_input_bundle.zip"
    contract_path = output_dir / "preparation_contract.json"
    notebook_path = output_dir / "prepare_t4.ipynb"
    manifest_path = output_dir / "bundle_manifest.json"
    if not all(
        path.is_file()
        for path in (archive_path, contract_path, notebook_path, manifest_path)
    ):
        raise ProtocolContractError("packed preparation bundle is incomplete")
    members_sha256: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            digest = hashlib.sha256()
            with archive.open(member) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            members_sha256[member.filename] = digest.hexdigest()
    transport = {
        "format": "zip",
        "filename": archive_path.name,
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
        "members_sha256": members_sha256,
    }
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    notebook_contract = json.loads(json.dumps(contract))
    notebook_contract["transport"] = transport
    notebook_path.write_text(
        render_t4_preparation_notebook(notebook_contract), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transport"] = transport
    manifest["files_sha256"][notebook_path.name] = _sha256(notebook_path)
    manifest["files_sha256"][archive_path.name] = transport["sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "archive": str(archive_path),
        "notebook": str(notebook_path),
        "transport": transport,
    }


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ProtocolContractError("preparation bundle requires source records")
    seen_keys: set[str] = set()
    seen_indices: set[int] = set()
    for row_number, row in enumerate(records):
        source_id = str(row.get("source_id", "")).strip()
        try:
            sample_index = int(row["sample_index"])
            label = int(row["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolContractError(
                f"source row {row_number} lacks integer sample_index/label"
            ) from exc
        source_key = str(row.get("source_key", f"{source_id}@{sample_index}"))
        if source_key != f"{source_id}@{sample_index}":
            raise ProtocolContractError(
                f"source row {row_number} identity does not equal source_id@sample_index"
            )
        if label not in (0, 1) or sample_index < 0 or not source_id:
            raise ProtocolContractError(f"invalid source row {row_number}")
        if source_key in seen_keys or sample_index in seen_indices:
            raise ProtocolContractError(f"duplicate source identity at row {row_number}")
        path = Path(str(row.get("mol2_path", "")))
        if not path.is_file():
            raise ProtocolContractError(f"missing MOL2 source at row {row_number}")
        expected = str(row.get("source_sha256", "")).lower()
        if len(expected) != 64 or _sha256(path) != expected:
            raise ProtocolContractError(f"MOL2 hash mismatch at row {row_number}")
        seen_keys.add(source_key)
        seen_indices.add(sample_index)


def stage_t4_preparation_bundle(
    document: Mapping[str, Any],
    *,
    arm_id: str,
    target_id: str,
    pdb_id: str,
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    execution_authorization: str | None = None,
    mgltools_runtime_path: Path | None = None,
) -> dict[str, Any]:
    """Stage reviewable files only. Never uploads, submits, or executes work."""
    validate_protocol_freeze(document)
    if arm_id not in PREPARATION_ARM:
        raise ProtocolContractError(f"unknown preparation arm: {arm_id}")
    prep_arm = PREPARATION_ARM[arm_id]
    _validate_records(records)
    synthetic_authorized = execution_authorization == "synthetic_runtime_validation"
    development_authorized = execution_authorization == "development_execution"
    if execution_authorization is not None and not (
        synthetic_authorized or development_authorized
    ):
        raise ProtocolContractError("unsupported preparation execution authorization")
    if synthetic_authorized and (
        target_id != "SYNTHETIC"
        or len(records) > 3
        or any(
            not str(row.get("source_id", "")).startswith("synthetic-")
            for row in records
        )
    ):
        raise ProtocolContractError(
            "synthetic authorization is limited to at most three synthetic records"
        )
    if development_authorized:
        execution = document.get("execution", {})
        if execution.get("development_launch_authorized") is not True:
            raise ProtocolContractError("development execution is not authorized")
        if execution.get("holdout_launch_authorized") is not False:
            raise ProtocolContractError("holdout execution must remain unauthorized")
        holdouts = {str(value).upper() for value in execution.get("holdout_target_ids", [])}
        excluded = {str(value).upper() for value in execution.get("excluded_target_ids", [])}
        requested_target = str(target_id).upper()
        if requested_target in holdouts or requested_target in excluded:
            raise ProtocolContractError("development authorization excludes holdout targets")
        target_specs = {
            str(item.get("target_id", "")).upper(): item
            for item in execution.get("development_target_manifest", [])
            if isinstance(item, Mapping)
        }
        target_spec = target_specs.get(requested_target)
        if target_spec is None:
            raise ProtocolContractError("target is not in frozen development manifest")
        if str(pdb_id).upper() != str(target_spec.get("pdb_id", "")).upper():
            raise ProtocolContractError("development target PDB differs from freeze")
        expected_actives = int(target_spec["actives"])
        expected_decoys = int(target_spec["decoys"])
        expected_total = expected_actives + expected_decoys
        actual_actives = sum(int(row["label"]) == 1 for row in records)
        actual_decoys = sum(int(row["label"]) == 0 for row in records)
        indices = sorted(int(row["sample_index"]) for row in records)
        if (
            len(records) != expected_total
            or actual_actives != expected_actives
            or actual_decoys != expected_decoys
            or indices != list(range(expected_total))
        ):
            raise ProtocolContractError(
                "development source counts, labels, or sample indices differ from freeze"
            )
    launch_authorized = synthetic_authorized or development_authorized
    if mgltools_runtime_path is not None and prep_arm != "B0_frozen_baseline":
        raise ProtocolContractError("MGLTools runtime is valid only for B0 preparation")
    runtime_source: Path | None = None
    if prep_arm == "B0_frozen_baseline":
        if mgltools_runtime_path is None:
            runtime_url = str(
                document["toolchain"].get("mgltools_runtime_url", "")
            ).strip()
            if launch_authorized and not runtime_url.startswith("https://"):
                raise ProtocolContractError(
                    "authorized B0 runtime requires a frozen MGLTools asset or HTTPS URL"
                )
        else:
            runtime_source = Path(mgltools_runtime_path)
            expected_name = document["toolchain"]["mgltools_runtime_filename"]
            expected_hash = document["toolchain"]["mgltools_runtime_sha256"]
            if (
                not runtime_source.is_file()
                or runtime_source.name != expected_name
                or _sha256(runtime_source) != expected_hash
            ):
                raise ProtocolContractError("MGLTools runtime filename/hash mismatch")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProtocolContractError("preparation bundle output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = output_dir / "source_mol2"
    sources_dir.mkdir()
    if runtime_source is not None:
        shutil.copy2(runtime_source, output_dir / runtime_source.name)

    intended: list[dict[str, Any]] = []
    for row in records:
        sample_index = int(row["sample_index"])
        source_id = str(row["source_id"]).strip()
        source_key = f"{source_id}@{sample_index}"
        source_path = Path(str(row["mol2_path"]))
        destination = sources_dir / f"{sample_index:06d}.mol2"
        shutil.copy2(source_path, destination)
        intended.append(
            {
                "source_key": source_key,
                "source_id": source_id,
                "sample_index": sample_index,
                "label": int(row["label"]),
                "source_sha256": _sha256(destination),
                "mol2": destination.relative_to(output_dir).as_posix(),
            }
        )
    intended.sort(key=lambda row: row["sample_index"])
    intended_path = output_dir / "intended_sources.csv"
    with intended_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(intended)

    contract = {
        "schema_version": 1,
        "study_id": document["study_id"],
        "target_id": target_id,
        "pdb_id": pdb_id,
        "requested_arm": arm_id,
        "preparation_arm": prep_arm,
        "preparation_contract": preparation_contract(document, prep_arm),
        "toolchain": document["toolchain"],
        "source_identity": "source_id@sample_index",
        "source_count": len(intended),
        "source_manifest": intended_path.name,
        "prepared_manifest_fields": list(PREPARED_FIELDS),
        "provenance_contract": document["provenance_contract"],
        "fail_closed": True,
        "fallbacks_permitted": False,
        "execution": {
            "hardware": "Kaggle Tesla T4 only",
            "launch_authorized": launch_authorized,
            "authorization_scope": execution_authorization,
        },
    }
    contract_path = output_dir / "preparation_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    runtime_source = Path(__file__).with_name(
        "early_enrichment_prep_runtime.py"
    )
    runtime_destination = output_dir / runtime_source.name
    shutil.copy2(runtime_source, runtime_destination)
    notebook_path = output_dir / "prepare_t4.ipynb"
    notebook_path.write_text(
        render_t4_preparation_notebook(contract), encoding="utf-8"
    )
    files = {
        path.relative_to(output_dir).as_posix(): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "study_id": document["study_id"],
        "stage": "ligand_preparation",
        "target_id": target_id,
        "pdb_id": pdb_id,
        "requested_arm": arm_id,
        "preparation_arm": prep_arm,
        "source_count": len(intended),
        "contains_raw_mol2": True,
        "repository_policy": "Do not stage or commit this generated bundle.",
        "launch_authorized": launch_authorized,
        "authorization_scope": execution_authorization,
        "files_sha256": files,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "bundle_manifest": str(manifest_path)}
