"""Standalone Kaggle T4 ligand-preparation runtime for frozen EF arms."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

WORKING = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/swiftscreen_early_enrichment_prep")
PREPARED = WORKING / "prepared_ligands"
FIELDS = (
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

L1_COORDINATE_TOLERANCE_ANGSTROM = 0.01
MAX_SYMMETRY_MAPPINGS = 100_000
LOG_TAIL_LIMIT = 2_000
COMMAND_LOG: list[dict[str, Any]] = []
EXECUTABLES: dict[str, dict[str, Any]] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_tail(value: Any, limit: int = LOG_TAIL_LIMIT) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)[-limit:]


def write_command_ledger() -> None:
    bootstrap_path = WORKING / "bootstrap_commands.json"
    bootstrap = []
    if bootstrap_path.is_file():
        payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        bootstrap = list(payload.get("commands", []))
    (WORKING / "command_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "log_tail_limit": LOG_TAIL_LIMIT,
                "commands": [*bootstrap, *COMMAND_LOG],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run(command: list[Any], *, cwd: Path | None = None, timeout: int = 7200):
    argv = [str(part) for part in command]
    started = time.time()
    entry: dict[str, Any] = {
        "sequence": len(COMMAND_LOG),
        "argv": argv,
        "cwd": str(cwd) if cwd is not None else None,
        "timeout_seconds": timeout,
    }
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        entry.update(
            {
                "return_code": None,
                "status": "timeout",
                "elapsed_seconds": time.time() - started,
                "output_tail": bounded_tail(exc.stdout),
            }
        )
        COMMAND_LOG.append(entry)
        write_command_ledger()
        raise
    entry.update(
        {
            "return_code": result.returncode,
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "output_tail": bounded_tail(result.stdout),
        }
    )
    COMMAND_LOG.append(entry)
    write_command_ledger()
    return result


def record_executable(name: str, path: str | Path | None, *, required: bool) -> None:
    resolved = Path(path).resolve() if path else None
    exists = bool(resolved and resolved.is_file())
    if required and not exists:
        raise RuntimeError(f"required executable missing: {name}")
    EXECUTABLES[name] = {
        "path": str(resolved) if resolved else None,
        "required": required,
        "exists": exists,
        "sha256": sha256(resolved) if exists and resolved is not None else None,
        "size_bytes": resolved.stat().st_size if exists and resolved is not None else None,
    }


def write_executable_manifest() -> None:
    (WORKING / "executable_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "executables": EXECUTABLES},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_environment_provenance(contract: dict[str, Any]) -> None:
    freeze = run([sys.executable, "-m", "pip", "freeze", "--all"], timeout=300)
    if freeze.returncode:
        raise RuntimeError(
            f"pip freeze failed: return code {freeze.returncode}: "
            f"{bounded_tail(freeze.stdout, 500)}"
        )
    freeze_path = WORKING / "pip_freeze.txt"
    freeze_path.write_text(freeze.stdout, encoding="utf-8")
    os_release: dict[str, str] = {}
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        pass
    identifiers = {
        key: os.environ[key]
        for key in (
            "KAGGLE_CONTAINER_NAME",
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_URL_BASE",
        )
        if key in os.environ
    }
    (WORKING / "environment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": {
                    "version": sys.version,
                    "implementation": platform.python_implementation(),
                    "executable": sys.executable,
                },
                "platform": platform.platform(),
                "uname": list(platform.uname()),
                "os_release": os_release,
                "container_identifiers": identifiers,
                "pip_freeze": {
                    "path": freeze_path.name,
                    "sha256": sha256(freeze_path),
                    "line_count": len(freeze.stdout.splitlines()),
                },
                "study_id": contract["study_id"],
                "target_id": contract["target_id"],
                "preparation_arm": contract["preparation_arm"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_executable("python", sys.executable, required=True)
    if contract["preparation_arm"] == "L1_meeko_single_state":
        record_executable("openbabel", shutil.which("obabel"), required=True)
    write_executable_manifest()


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def locate_bundle() -> tuple[Path, dict[str, Any]]:
    roots = (
        Path("/kaggle/input"),
        Path("/kaggle/temp/swiftscreen_early_enrichment_input"),
    )
    contracts = [
        path
        for root in roots
        for path in root.rglob("preparation_contract.json")
    ]
    if len(contracts) != 1:
        raise RuntimeError(f"expected one preparation contract, found {len(contracts)}")
    contract = json.loads(contracts[0].read_text(encoding="utf-8"))
    if contract["execution"]["launch_authorized"] is not True:
        raise RuntimeError("preparation execution is not authorized")
    if contract.get("fallbacks_permitted") is not False:
        raise RuntimeError("preparation contract permits fallback")
    hardware = WORKING / "hardware_manifest.json"
    if not hardware.is_file() or json.loads(hardware.read_text()).get("gate_passed") is not True:
        raise RuntimeError("verified Tesla T4 hardware manifest missing")
    return contracts[0].parent, contract


def verify_versions(contract: dict[str, Any]) -> dict[str, str]:
    from packaging.version import Version

    frozen = contract["toolchain"]
    distributions = {
        "openbabel_wheel": "openbabel-wheel",
        "rdkit": "rdkit",
        "meeko": "meeko",
        "gemmi": "gemmi",
    }
    if contract["preparation_arm"] == "L2_bounded_state_ensemble":
        distributions.update(
            {"gypsum_dl": "gypsum-dl", "dimorphite_dl": "dimorphite-dl"}
        )
    observed: dict[str, str] = {}
    for key, distribution in distributions.items():
        value = importlib.metadata.version(distribution)
        observed[key] = value
        if Version(value) != Version(str(frozen[key])):
            raise RuntimeError(
                f"tool version mismatch for {distribution}: {value} != {frozen[key]}"
            )
    (WORKING / "tool_versions.json").write_text(
        json.dumps({"frozen": frozen, "observed": observed}, indent=2) + "\n"
    )
    return observed


def source_rows(bundle_root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    path = bundle_root / contract["source_manifest"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != int(contract["source_count"]):
        raise RuntimeError("source manifest count mismatch")
    keys: set[str] = set()
    for row in rows:
        expected_key = f"{row['source_id']}@{int(row['sample_index'])}"
        if row["source_key"] != expected_key or expected_key in keys:
            raise RuntimeError("source identity mismatch or duplicate")
        source = bundle_root / row["mol2"]
        if not source.is_file() or sha256(source) != row["source_sha256"]:
            raise RuntimeError(f"source hash mismatch: {expected_key}")
        row["_path"] = str(source)
        keys.add(expected_key)
    return rows


def install_mgltools(bundle_root: Path, contract: dict[str, Any]) -> tuple[Path, Path]:
    filename = contract["toolchain"]["mgltools_runtime_filename"]
    matches = [
        path
        for root in (
            Path("/kaggle/input"),
            Path("/kaggle/temp/swiftscreen_early_enrichment_input"),
        )
        for path in root.rglob(filename)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"expected at most one MGLTools runtime, found {len(matches)}")
    if matches:
        runtime = matches[0]
    else:
        url = str(contract["toolchain"].get("mgltools_runtime_url", "")).strip()
        if not url.startswith("https://"):
            raise RuntimeError("frozen MGLTools HTTPS source is unavailable")
        runtime = SCRATCH / filename
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "swiftscreen-early-enrichment/1"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            with runtime.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    expected = contract["toolchain"]["mgltools_runtime_sha256"]
    if sha256(runtime) != expected:
        raise RuntimeError("MGLTools runtime hash mismatch")
    unpacked = SCRATCH / "mgl_dist"
    safe_extract_tar(runtime, unpacked)
    distribution = next(unpacked.glob("mgltools_*"))
    root = SCRATCH / "mgltools"
    result = run(["bash", "install.sh", "-d", root], cwd=distribution)
    if result.returncode:
        raise RuntimeError(f"MGLTools install failed: {result.stdout[-2000:]}")
    pythonsh = root / "bin/pythonsh"
    utility = root / "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    if not pythonsh.is_file() or not utility.is_file():
        raise RuntimeError("MGLTools/AutoDockTools installation incomplete")
    record_executable("pythonsh", pythonsh, required=True)
    record_executable("prepare_ligand4.py", utility, required=True)
    write_executable_manifest()
    return pythonsh, utility


def write_meeko(molecule, destination: Path) -> None:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation(hydrate=False).prepare(molecule)
    if len(setups) != 1:
        raise RuntimeError(f"Meeko returned {len(setups)} molecule setups")
    pdbqt, ok, message = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"Meeko PDBQT write failed: {message}")
    destination.write_text(
        "\n".join(line for line in pdbqt.splitlines() if line.strip() != "END")
        + "\n"
    )


def molecule_signature(molecule) -> tuple[Any, ...]:
    from rdkit import Chem

    elements = Counter(
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    centers = sorted(
        label
        for _index, label in Chem.FindMolChiralCenters(
            molecule, includeUnassigned=False, useLegacyImplementation=False
        )
    )
    heavy_graph_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(molecule)),
        canonical=True,
        isomericSmiles=True,
    )
    return (
        tuple(sorted(elements.items())),
        charge,
        tuple(centers),
        heavy_graph_smiles,
    )


def heavy_atom_coordinate_drift(
    source,
    converted,
    *,
    tolerance: float = L1_COORDINATE_TOLERANCE_ANGSTROM,
    max_mappings: int = MAX_SYMMETRY_MAPPINGS,
) -> float:
    """Return minimum symmetry-aware maximum heavy-atom displacement."""
    from rdkit import Chem

    if tolerance < 0 or max_mappings <= 0:
        raise ValueError("coordinate tolerance/mapping limit must be positive")
    source_heavy = Chem.RemoveHs(Chem.Mol(source))
    converted_heavy = Chem.RemoveHs(Chem.Mol(converted))
    if (
        source_heavy.GetNumAtoms() != converted_heavy.GetNumAtoms()
        or source_heavy.GetNumConformers() != 1
        or converted_heavy.GetNumConformers() != 1
    ):
        raise RuntimeError("L1 coordinate QC requires equal 3D heavy-atom graphs")
    mappings = converted_heavy.GetSubstructMatches(
        source_heavy,
        uniquify=False,
        useChirality=True,
        maxMatches=max_mappings,
    )
    if not mappings:
        raise RuntimeError("L1 coordinate QC found no stereochemical graph mapping")
    source_conformer = source_heavy.GetConformer()
    converted_conformer = converted_heavy.GetConformer()
    best = math.inf
    for mapping in mappings:
        maximum = max(
            math.dist(
                source_conformer.GetAtomPosition(source_index),
                converted_conformer.GetAtomPosition(converted_index),
            )
            for source_index, converted_index in enumerate(mapping)
        )
        best = min(best, maximum)
        if best <= tolerance:
            break
    if len(mappings) == max_mappings and best > tolerance:
        raise RuntimeError("L1 coordinate QC exceeded symmetry-mapping limit")
    return best


def require_mmff94_convergence(status: int) -> int:
    """Accept only RDKit's converged MMFF94 status code."""
    code = int(status)
    if code != 0:
        raise RuntimeError(f"MMFF94 did not converge: return code {code}")
    return code


def prepare_b0(
    rows: list[dict[str, Any]],
    bundle_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    pythonsh, utility = install_mgltools(bundle_root, contract)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        sample_index = int(row["sample_index"])
        ligand = f"lig_{sample_index:06d}_s00"
        destination = PREPARED / f"{ligand}.pdbqt"
        result = run([pythonsh, utility, "-l", row["_path"], "-o", destination])
        ok = result.returncode == 0 and destination.is_file() and destination.stat().st_size
        if ok:
            destination.write_text(
                "\n".join(
                    line
                    for line in destination.read_text(errors="replace").splitlines()
                    if line.strip() != "END"
                )
                + "\n"
            )
        if ok:
            reason = ""
        elif result.returncode:
            reason = (
                "autodocktools_failed:"
                f"return_code={result.returncode}:"
                f"output_tail={bounded_tail(result.stdout, 350)}"
            )[-500:]
        else:
            reason = "autodocktools_failed:return_code=0:output_not_created"
        prepared.append(
            prep_row(
                row,
                0,
                ligand if ok else "",
                destination if ok else None,
                reason,
                contract["preparation_arm"],
            )
        )
    return prepared


def prepare_l1(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    from rdkit import Chem

    prepared: list[dict[str, Any]] = []
    for row in rows:
        sample_index = int(row["sample_index"])
        ligand = f"lig_{sample_index:06d}_s00"
        destination = PREPARED / f"{ligand}.pdbqt"
        reason = ""
        coordinate_drift: float | None = None
        try:
            source = Chem.MolFromMol2File(
                row["_path"], sanitize=True, removeHs=False
            )
            if source is None:
                raise RuntimeError("RDKit could not parse source MOL2")
            sdf = SCRATCH / f"{ligand}.sdf"
            conversion = run(["obabel", "-imol2", row["_path"], "-osdf", "-O", sdf])
            if conversion.returncode or not sdf.is_file():
                raise RuntimeError("OpenBabel MOL2-to-SDF conversion failed")
            molecules = [
                molecule
                for molecule in Chem.SDMolSupplier(
                    str(sdf), removeHs=False, sanitize=True
                )
                if molecule is not None
            ]
            if len(molecules) != 1:
                raise RuntimeError(f"OpenBabel produced {len(molecules)} molecules")
            if molecule_signature(source) != molecule_signature(molecules[0]):
                raise RuntimeError("L1 heavy-element/charge/stereochemistry QC failed")
            coordinate_drift = heavy_atom_coordinate_drift(source, molecules[0])
            if coordinate_drift > L1_COORDINATE_TOLERANCE_ANGSTROM:
                raise RuntimeError(
                    "L1 heavy-atom coordinate drift "
                    f"{coordinate_drift:.6f} exceeds "
                    f"{L1_COORDINATE_TOLERANCE_ANGSTROM:.6f} Angstrom"
                )
            write_meeko(molecules[0], destination)
        except Exception as exc:
            reason = f"l1_fail_closed:{type(exc).__name__}:{exc}"[-500:]
        prepared.append(
            prep_row(
                row,
                0,
                ligand if not reason else "",
                destination if not reason else None,
                reason,
                contract["preparation_arm"],
                coordinate_max_drift_angstrom=coordinate_drift,
            )
        )
    return prepared


def gypsum_states(smiles: str, max_states: int) -> list[str]:
    from rdkit import Chem

    try:
        from gypsum_dl.run import main as gypsum_main
    except ImportError as exc:
        raise RuntimeError("Gypsum-DL import failed") from exc
    executable = shutil.which("gypsum-dl") or shutil.which("gypsum_dl")
    command = (
        [executable]
        if executable
        else [
            sys.executable,
            "-c",
            "from gypsum_dl.run import main; main()",
        ]
    )
    del gypsum_main
    record_executable(
        "gypsum-dl",
        executable if executable else sys.executable,
        required=True,
    )
    write_executable_manifest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "input.smi"
        output = root / "output"
        source.write_text(f"{smiles}\tlig\n")
        output.mkdir()
        result = run(
            [
                *command,
                "--source",
                source,
                "--output_folder",
                output,
                "--2d_output_only",
                "--separate_output_files",
                "--max_variants_per_compound",
                max_states,
                "--job_manager",
                "serial",
                "--min_ph",
                "6.4",
                "--max_ph",
                "8.4",
                "--skip_enumerate_chiral_mol",
                "--skip_alternate_ring_conformations",
            ]
        )
        if result.returncode:
            raise RuntimeError(f"Gypsum-DL failed: {result.stdout[-1000:]}")
        states: set[str] = set()
        for sdf in sorted(output.glob("*.sdf")):
            for molecule in Chem.SDMolSupplier(
                str(sdf), removeHs=False, sanitize=True
            ):
                if molecule is not None:
                    states.add(
                        Chem.MolToSmiles(
                            molecule, canonical=True, isomericSmiles=True
                        )
                    )
        ordered = sorted(state for state in states if state)
        if not ordered:
            raise RuntimeError("Gypsum-DL produced no valid state")
        return ordered[:max_states]


def prepare_l2(
    rows: list[dict[str, Any]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    max_states = int(
        contract["preparation_contract"]["max_states_per_source_sample"]
    )
    prepared: list[dict[str, Any]] = []
    for row in rows:
        try:
            source = Chem.MolFromMol2File(
                row["_path"], sanitize=True, removeHs=False
            )
            if source is None:
                raise RuntimeError("RDKit could not parse source MOL2")
            smiles = Chem.MolToSmiles(
                source, canonical=True, isomericSmiles=True
            )
            states = gypsum_states(smiles, max_states)
        except Exception as exc:
            prepared.append(
                prep_row(
                    row,
                    0,
                    "",
                    None,
                    f"l2_enumeration_fail_closed:{type(exc).__name__}:{exc}"[-500:],
                    contract["preparation_arm"],
                )
            )
            continue
        for state_index, state in enumerate(states):
            ligand = f"lig_{int(row['sample_index']):06d}_s{state_index:02d}"
            destination = PREPARED / f"{ligand}.pdbqt"
            reason = ""
            mmff_status: int | None = None
            try:
                molecule = Chem.MolFromSmiles(state, sanitize=True)
                if molecule is None:
                    raise RuntimeError("enumerated state did not parse")
                molecule = Chem.AddHs(molecule)
                parameters = AllChem.ETKDGv3()
                parameters.randomSeed = 42
                if AllChem.EmbedMolecule(molecule, parameters) == -1:
                    raise RuntimeError("ETKDGv3 embedding failed")
                properties = AllChem.MMFFGetMoleculeProperties(
                    molecule, mmffVariant="MMFF94"
                )
                if properties is None:
                    raise RuntimeError("MMFF94 parameters unavailable")
                mmff_status = require_mmff94_convergence(
                    AllChem.MMFFOptimizeMolecule(
                        molecule, mmffVariant="MMFF94", maxIters=200
                    )
                )
                write_meeko(molecule, destination)
            except Exception as exc:
                reason = f"l2_state_fail_closed:{type(exc).__name__}:{exc}"[-500:]
            prepared.append(
                prep_row(
                    row,
                    state_index,
                    ligand if not reason else "",
                    destination if not reason else None,
                    reason,
                    contract["preparation_arm"],
                    mmff_convergence_code=mmff_status,
                )
            )
    return prepared


def prep_row(
    source: dict[str, Any],
    state_index: int,
    ligand: str,
    destination: Path | None,
    reason: str,
    prep_arm: str,
    *,
    coordinate_max_drift_angstrom: float | None = None,
    mmff_convergence_code: int | None = None,
) -> dict[str, Any]:
    relative = (
        destination.relative_to(WORKING).as_posix() if destination is not None else ""
    )
    return {
        "source_key": source["source_key"],
        "source_id": source["source_id"],
        "sample_index": int(source["sample_index"]),
        "label": int(source["label"]),
        "state_index": state_index,
        "ligand": ligand,
        "preparation_status": "failed" if reason else "prepared",
        "failure_reason": reason,
        "source_sha256": source["source_sha256"],
        "prepared_sha256": sha256(destination) if destination is not None else "",
        "pdbqt": relative,
        "prep_arm": prep_arm,
        "coordinate_max_drift_angstrom": (
            ""
            if coordinate_max_drift_angstrom is None
            else round(float(coordinate_max_drift_angstrom), 6)
        ),
        "mmff_convergence_code": (
            "" if mmff_convergence_code is None else int(mmff_convergence_code)
        ),
    }


def artifact_entries(paths: list[Path], root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = str(path)
        entries[name] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return entries


def write_artifact_manifest(
    bundle_root: Path,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    input_paths = [
        bundle_root / "preparation_contract.json",
        bundle_root / contract["source_manifest"],
        bundle_root / "prepare_t4.ipynb",
        bundle_root / "early_enrichment_prep_runtime.py",
        *(Path(row["_path"]) for row in rows),
    ]
    output_paths = [
        WORKING / "prepared_manifest.csv",
        WORKING / "preparation_failures.csv",
        WORKING / "tool_versions.json",
        WORKING / "hardware_manifest.json",
        WORKING / "pip_freeze.txt",
        WORKING / "environment_manifest.json",
        WORKING / "executable_manifest.json",
        WORKING / "bootstrap_commands.json",
        WORKING / "command_ledger.json",
        *sorted(PREPARED.glob("*.pdbqt")),
    ]
    path = WORKING / "artifact_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_id": contract["study_id"],
                "target_id": contract["target_id"],
                "preparation_arm": contract["preparation_arm"],
                "inputs": artifact_entries(input_paths, bundle_root),
                "receptor": "not_applicable_to_ligand_preparation",
                "executables": EXECUTABLES,
                "outputs": artifact_entries(output_paths, WORKING),
                "self_hash_policy": "artifact manifest, mutable pre-archive runtime, archive, and external archive manifest are excluded to avoid recursive or stale hashes",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    started = time.time()
    COMMAND_LOG.clear()
    EXECUTABLES.clear()
    bundle_root, contract = locate_bundle()
    verify_versions(contract)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    PREPARED.mkdir(parents=True, exist_ok=True)
    write_environment_provenance(contract)
    rows = source_rows(bundle_root, contract)
    arm = contract["preparation_arm"]
    if arm == "B0_frozen_baseline":
        prepared = prepare_b0(rows, bundle_root, contract)
    elif arm == "L1_meeko_single_state":
        prepared = prepare_l1(rows, contract)
    elif arm == "L2_bounded_state_ensemble":
        prepared = prepare_l2(rows, contract)
    else:
        raise RuntimeError(f"unsupported preparation arm: {arm}")

    prepared_path = WORKING / "prepared_manifest.csv"
    failures_path = WORKING / "preparation_failures.csv"
    write_csv(prepared_path, prepared)
    failures = [row for row in prepared if row["preparation_status"] != "prepared"]
    write_csv(failures_path, failures)
    prepared_count = len(prepared) - len(failures)
    source_successes = {
        row["source_key"]
        for row in prepared
        if row["preparation_status"] == "prepared"
    }
    runtime = {
        "study_id": contract["study_id"],
        "target_id": contract["target_id"],
        "pdb_id": contract["pdb_id"],
        "requested_arm": contract["requested_arm"],
        "preparation_arm": arm,
        "intended_sources": len(rows),
        "sources_with_at_least_one_prepared_state": len(source_successes),
        "prepared_states": prepared_count,
        "failed_states": len(failures),
        "fallbacks": 0,
        "elapsed_seconds": time.time() - started,
    }
    runtime_path = WORKING / "preparation_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    write_command_ledger()
    write_executable_manifest()
    artifact_path = write_artifact_manifest(bundle_root, contract, rows)
    archived_runtime_sha256 = sha256(runtime_path)
    archive_path = WORKING / "prepared_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (
            prepared_path,
            failures_path,
            runtime_path,
            WORKING / "tool_versions.json",
            WORKING / "hardware_manifest.json",
            WORKING / "pip_freeze.txt",
            WORKING / "environment_manifest.json",
            WORKING / "executable_manifest.json",
            WORKING / "bootstrap_commands.json",
            WORKING / "command_ledger.json",
            artifact_path,
        ):
            if path.is_file():
                archive.write(path, path.name)
        for path in sorted(PREPARED.glob("*.pdbqt")):
            archive.write(path, f"ligands/{path.name}")
    runtime["prepared_bundle_sha256"] = sha256(archive_path)
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    archive_manifest_path = WORKING / "archive_manifest.json"
    archive_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prepared_bundle_sha256": sha256(archive_path),
                "prepared_bundle_size_bytes": archive_path.stat().st_size,
                "archived_pre_archive_runtime_sha256": archived_runtime_sha256,
                "final_external_runtime_sha256": sha256(runtime_path),
                "artifact_manifest_sha256": sha256(artifact_path),
                "policy": "The ZIP contains the pre-archive runtime because an archive cannot contain its own final hash; this external manifest binds both versions.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (WORKING / "done.flag").write_text("ok\n")
    print(runtime)


if __name__ == "__main__":
    main()
