from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase

PAPER_DOI = "10.48550/arXiv.2403.10478"
PAPER_REPO_COMMIT = "03fc9de3710d6d4f5ce7c0bf91eeb7bb9a29432a"
EXPECTED_ARCHIVE_SHA256 = (
    "9dc111026c4f8fc43143fd2e5b5853015336ae46460e063d7b5b81d62d0a4b9d"
)
VINA_VERSION = "1.2.5"
VINA_BINARY_FILENAME = "vina_1.2.5_linux_x86_64"
VINA_BINARY_SHA256 = (
    "fa0126a28a9ea9162d1b161dfa92bc76e632416db28ca246278ea4b2dc6860cb"
)
BIGBIND_ARCHIVE_SHA256 = (
    "de2b059846b954a8b297b7b0c78c037029f330d6c45431fea49390f0a6b8c91c"
)
BIGBIND_ARCHIVE_BYTES = 19254374400
BASELINES_ARCHIVE_BYTES = 4173711360
GNINA_VERSION = "1.3.3"
GNINA_BINARY_URL = (
    "https://github.com/gnina/gnina/releases/download/"
    "v1.3.3/gnina.cuda12.8.static"
)
GNINA_CNN_MODEL = "crossdock_default2018"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _portable_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _source_id(target_id: str, canonical_smiles: str) -> str:
    token = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:20]
    return f"{target_id}__{token}"


def _same_float_column(frame: pd.DataFrame, name: str) -> float:
    values = frame[name].dropna().astype(float).unique()
    if len(values) != 1:
        raise ValueError(f"{name} must contain one target-level value")
    return float(values[0])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _code_cell(source: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def freeze_bayesbind(
    campaign_dir: Path,
    extracted_dir: Path,
    *,
    archive_path: Path | None = None,
    random_limit: int = 1000,
    active_limit: int = 1000,
    active_cutoff: float = 5.0,
) -> dict[str, Any]:
    """Freeze paper-era BayesBind rows and a validity-corrected source view."""
    campaign_dir = campaign_dir.expanduser().resolve()
    extracted_dir = extracted_dir.expanduser().resolve()
    if (extracted_dir / "BayesBindV1.5").is_dir():
        root = extracted_dir / "BayesBindV1.5"
    else:
        root = extracted_dir
    if random_limit <= 0 or active_limit <= 0:
        raise ValueError("random_limit and active_limit must be positive")
    split_dirs = {"val": root / "val", "test": root / "test"}
    if not all(path.is_dir() for path in split_dirs.values()):
        raise FileNotFoundError("Expected BayesBindV1.5 val/ and test/ directories")

    discovered: list[tuple[str, Path]] = []
    for paper_split, split_dir in split_dirs.items():
        discovered.extend(
            (paper_split, path)
            for path in split_dir.iterdir()
            if path.is_dir()
        )
    discovered.sort(key=lambda item: item[1].name)
    if len(discovered) != 25:
        raise ValueError(f"Paper-era BayesBind must contain 25 targets, got {len(discovered)}")
    if sum(split == "val" for split, _ in discovered) != 14:
        raise ValueError("Paper-era BayesBind must contain 14 validation targets")
    if sum(split == "test" for split, _ in discovered) != 11:
        raise ValueError("Paper-era BayesBind must contain 11 test targets")

    inputs_dir = campaign_dir / "paper_inputs"
    targets: list[dict[str, Any]] = []
    totals = {
        "paper_active_rows": 0,
        "paper_random_rows": 0,
        "paper_intended_rows": 0,
        "source_compounds": 0,
        "cross_label_conflicts": 0,
        "canonicalization_failed": 0,
    }

    for position, (paper_split, target_dir) in enumerate(discovered, start=1):
        target_id = target_dir.name
        active_frame = pd.read_csv(target_dir / "actives.csv").iloc[:active_limit].copy()
        random_frame = pd.read_csv(target_dir / "random.csv").iloc[:random_limit].copy()
        active_frame = active_frame[
            active_frame["pchembl_value"].astype(float) > float(active_cutoff)
        ].copy()
        if len(random_frame) != random_limit:
            raise ValueError(
                f"{target_id} has {len(random_frame)} random rows; expected {random_limit}"
            )

        paper_rows: list[dict[str, Any]] = []
        invalid_rows: list[dict[str, Any]] = []
        for label, frame in ((1, active_frame), (0, random_frame)):
            row_class = "active" if label else "random"
            for source_index, row in frame.reset_index(drop=True).iterrows():
                raw_smiles = str(row["lig_smiles"])
                try:
                    canonical = _canonical_smiles(raw_smiles)
                    error = ""
                except ValueError as exc:
                    canonical = ""
                    error = str(exc)
                paper_row = {
                    "paper_row_id": f"{target_id}__{row_class}__{source_index:05d}",
                    "target_id": target_id,
                    "paper_class": row_class,
                    "paper_label": label,
                    "source_index": int(source_index),
                    "lig_file": str(row.get("lig_file", "")).strip(),
                    "raw_smiles": raw_smiles,
                    "canonical_smiles": canonical,
                    "canonicalization_error": error,
                }
                paper_rows.append(paper_row)
                if error:
                    invalid_rows.append(paper_row)

        by_canonical: dict[str, dict[str, Any]] = {}
        conflict_count = 0
        for row in paper_rows:
            canonical = str(row["canonical_smiles"])
            if not canonical:
                continue
            current = by_canonical.get(canonical)
            if current is None:
                by_canonical[canonical] = dict(row)
                continue
            if int(current["paper_label"]) != int(row["paper_label"]):
                conflict_count += 1
                if int(row["paper_label"]) == 1:
                    by_canonical[canonical] = dict(row)

        validity_rows: list[dict[str, Any]] = []
        for canonical, row in sorted(by_canonical.items()):
            label = int(row["paper_label"])
            validity_rows.append(
                {
                    "source_compound_id": _source_id(target_id, canonical),
                    "target_id": target_id,
                    "label": label,
                    "label_source": (
                        "measured_activity_pchembl_gt_cutoff"
                        if label
                        else "paper_random_set"
                    ),
                    "canonical_smiles": canonical,
                    "lig_file": str(row.get("lig_file", "")).strip(),
                    "cross_label_conflict_resolved": int(
                        any(
                            other["canonical_smiles"] == canonical
                            and int(other["paper_label"]) != label
                            for other in paper_rows
                        )
                    ),
                }
            )

        target_out = inputs_dir / target_id
        target_out.mkdir(parents=True, exist_ok=True)
        paper_fields = [
            "paper_row_id",
            "target_id",
            "paper_class",
            "paper_label",
            "source_index",
            "lig_file",
            "raw_smiles",
            "canonical_smiles",
            "canonicalization_error",
        ]
        validity_fields = [
            "source_compound_id",
            "target_id",
            "label",
            "label_source",
            "canonical_smiles",
            "lig_file",
            "cross_label_conflict_resolved",
        ]
        _write_csv(target_out / "paper_rows.csv", paper_rows, paper_fields)
        _write_csv(target_out / "source_compounds.csv", validity_rows, validity_fields)
        (target_out / "library.smi").write_text(
            "".join(
                f"{row['canonical_smiles']}\t{row['source_compound_id']}\n"
                for row in validity_rows
            ),
            encoding="utf-8",
        )
        (target_out / "actives.smi").write_text(
            "".join(
                f"{row['canonical_smiles']}\t{row['source_compound_id']}\n"
                for row in validity_rows
                if int(row["label"]) == 1
            ),
            encoding="utf-8",
        )

        center = [
            _same_float_column(active_frame, f"pocket_center_{axis}")
            for axis in "xyz"
        ]
        size = [
            _same_float_column(active_frame, f"pocket_size_{axis}")
            for axis in "xyz"
        ]
        (target_out / "decoys.smi").write_text(
            "".join(
                f"{row['canonical_smiles']}\t{row['source_compound_id']}{chr(10)}"
                for row in validity_rows
                if int(row["label"]) == 0
            ),
            encoding="utf-8",
        )

        target_record = {
            "position": position,
            "target_id": target_id,
            "paper_split": paper_split,
            "split": "development" if position % 2 else "held_out_validation",
            "paper_active_rows": len(active_frame),
            "paper_random_rows": len(random_frame),
            "paper_intended_rows": len(paper_rows),
            "source_compounds": len(validity_rows),
            "cross_label_conflicts": conflict_count,
            "canonicalization_failed": len(invalid_rows),
            "box_center": center,
            "box_size": size,
            "receptor_pdb": (
                target_dir / "rec.pdb"
            ).relative_to(campaign_dir.parent.parent).as_posix(),
            "receptor_hydrogenated_pdb": (
                target_dir / "rec_hs.pdb"
            ).relative_to(campaign_dir.parent.parent).as_posix(),
            "library_smi": (
                target_out / "library.smi"
            ).relative_to(campaign_dir).as_posix(),
            "actives_smi": (
                target_out / "actives.smi"
            ).relative_to(campaign_dir).as_posix(),
            "decoys_smi": (
                target_out / "decoys.smi"
            ).relative_to(campaign_dir).as_posix(),
        }
        targets.append(target_record)
        for key in totals:
            if key == "paper_active_rows":
                totals[key] += len(active_frame)
            elif key == "paper_random_rows":
                totals[key] += len(random_frame)
            elif key == "paper_intended_rows":
                totals[key] += len(paper_rows)
            elif key == "source_compounds":
                totals[key] += len(validity_rows)
            elif key == "cross_label_conflicts":
                totals[key] += conflict_count
            elif key == "canonicalization_failed":
                totals[key] += len(invalid_rows)

    archive_record: dict[str, Any] = {}
    if archive_path is not None:
        archive_path = archive_path.expanduser().resolve()
        archive_hash = _sha256(archive_path)
        if archive_hash != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(
                f"BayesBind archive SHA256 mismatch: {archive_hash}"
            )
        archive_record = {
            "path": archive_path.relative_to(campaign_dir).as_posix(),
            "sha256": archive_hash,
            "bytes": archive_path.stat().st_size,
        }

    protocol = {
        "schema_version": 1,
        "paper": {
            "title": "An Improved Metric and Benchmark for Assessing the Performance of Virtual Screening Models",
            "authors": [
                "Michael Brocidiacono",
                "Konstantin I. Popov",
                "Alexander Tropsha",
            ],
            "doi": PAPER_DOI,
            "paper_release": "arXiv:2403.10478v1",
            "paper_repository_commit": PAPER_REPO_COMMIT,
        },
        "official_input": archive_record,
        "paper_protocol": {
            "target_count": 25,
            "paper_splits": {"val": 14, "test": 11},
            "active_cutoff_pchembl": float(active_cutoff),
            "active_cutoff_operator": ">",
            "measured_rows_per_target_limit": int(active_limit),
            "random_rows_per_target": int(random_limit),
            "receptor_preparation": "Open Babel: obabel -xr -ipdb rec.pdb -opdbqt -p 7",
            "ligand_preparation": "paper code uses RDKit 3D input + Meeko; versions unreported",
            "box": "BigBind pocket center and size; ligand dimensions may enlarge box",
            "vina": "one CPU per ligand, executable version and explicit search parameters unreported",
            "gnina": "crossdock_default2018 CNN model, executable version unreported",
        },
        "frozen_reconstruction": {
            "vina_version": "1.2.5",
            "vina_version_classification": "paper-era inference; not reported",
            "vina_search_parameters": "defaults, matching released paper code",
            "failed_intended_compounds": "retain at worst rank",
            "source_aggregation": "one canonical source compound; measured-active label wins conflicts",
            "tie_handling": "0.5-credit ROC AUC and expected-actives EF boundary handling",
            "primary_split": "alphabetical target IDs; odd development, even held-out validation",
            "paper_split_retained": True,
        },
        "software_provenance": {"rdkit": rdBase.rdkitVersion},
        "counts": totals,
    }
    _write_json(campaign_dir / "targets.json", targets)
    _write_json(
        campaign_dir / "inputs" / "targets.json",
        [
            {
                "target_id": target["target_id"],
                "target_class": "BayesBind protein target",
                "primary_structure": {
                    "kind": "pdb",
                    "id": target["target_id"],
                    "path": target["receptor_pdb"],
                },
                "actives_file": target["actives_smi"],
                "decoys_file": target["decoys_smi"],
                "notes": [
                    f"paper_split={target['paper_split']}",
                    f"tuning_split={target['split']}",
                ],
            }
            for target in targets
        ],
    )
    _write_json(
        campaign_dir / "campaign_manifest.json",
        {
            "benchmark_id": campaign_dir.name,
            "goal": "Reproduce the BayesBind paper Vina-family virtual-screening baselines",
            "status": "running",
            "targets_file": "inputs/targets.json",
            "baselines": [
                "autodock_vina_1.2.5_cpu",
                "gnina_crossdock_default2018",
            ],
            "required_metrics": [
                "ef1",
                "ef5",
                "ef20",
                "auc_roc",
                "bedroc",
                "coverage",
                "first_active_rank",
                "pose_retention",
            ],
            "notes": [
                "Paper protocol frozen from arXiv:2403.10478v1 and repository commit "
                + PAPER_REPO_COMMIT,
                "Official ligand SDF bytes are absent; deterministic SMILES-to-3D reconstruction is declared.",
            ],
        },
    )
    _write_json(campaign_dir / "protocol_freeze.json", protocol)
    _write_json(
        campaign_dir / "source_manifest.json",
        {
            "schema_version": 1,
            "official_archive": archive_record,
            "paper_repository_commit": PAPER_REPO_COMMIT,
            "extracted_root": _portable_path(root, campaign_dir),
        },
    )
    return {
        "campaign_dir": str(campaign_dir),
        "target_count": len(targets),
        **totals,
        "targets_path": str(campaign_dir / "targets.json"),
        "protocol_path": str(campaign_dir / "protocol_freeze.json"),
    }

def _iter_frozen_paper_rows(campaign_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paper_root = campaign_dir / "paper_inputs"
    for paper_csv in sorted(paper_root.glob("*/paper_rows.csv")):
        with paper_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No frozen paper_rows.csv files found under {paper_root}")
    missing = [row.get("paper_row_id", "") for row in rows if not str(row.get("lig_file", "")).strip()]
    if missing:
        raise ValueError(f"Frozen paper rows missing lig_file: {missing[:5]}")
    return rows


def _normalize_member_key(name: str) -> str:
    token = name.replace("\\", "/").lstrip("/")
    parts = [part for part in token.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive path: {name}")
    if parts and parts[0] == "BigBindV1.5":
        parts = parts[1:]
    return "/".join(parts)


def _safe_output_path(root: Path, relative_name: str) -> Path:
    key = _normalize_member_key(relative_name)
    out = (root / key).resolve()
    resolved_root = root.resolve()
    if resolved_root != out and resolved_root not in out.parents:
        raise ValueError(f"Unsafe extraction target: {relative_name}")
    return out


def _sdf_has_valid_molecule(data: bytes, lig_file: str) -> None:
    supplier = Chem.ForwardSDMolSupplier(BytesIO(data), sanitize=False, removeHs=False)
    try:
        mol = next((mol for mol in supplier if mol is not None), None)
    except Exception as exc:
        raise ValueError(f"Malformed SDF for {lig_file}: {exc}") from exc
    if mol is None:
        raise ValueError(f"Malformed SDF for {lig_file}: no readable molecule")
    if mol.GetNumConformers() < 1:
        raise ValueError(f"Malformed SDF for {lig_file}: no 3D conformer")


def resolve_exact_bayesbind_ligand_members(
    campaign_dir: Path,
    *,
    inventory_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve each frozen paper lig_file to exactly one BigBind tar member."""
    campaign_dir = campaign_dir.expanduser().resolve()
    inventory_path = (
        inventory_path.expanduser().resolve()
        if inventory_path is not None
        else campaign_dir / "inputs" / "archive_manifests" / "BigBindV1.5.members.jsonl"
    )
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else campaign_dir / "inputs" / "exact_ligands"
    )
    rows = _iter_frozen_paper_rows(campaign_dir)
    required = sorted({str(row["lig_file"]).strip().replace("\\", "/") for row in rows})
    required_set = set(required)
    matches: dict[str, list[dict[str, Any]]] = {lig_file: [] for lig_file in required}
    with inventory_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            member = json.loads(line)
            if member.get("kind") != "file":
                continue
            key = _normalize_member_key(str(member.get("name", "")))
            if key in required_set:
                matches[key].append(member)
    missing = [lig_file for lig_file, found in matches.items() if not found]
    duplicates = {lig_file: found for lig_file, found in matches.items() if len(found) > 1}
    if missing:
        raise ValueError(f"Missing BigBind members for lig_file values: {missing[:10]}")
    if duplicates:
        example_lig_file, example_members = next(iter(duplicates.items()))
        names = [str(member["name"]) for member in example_members]
        raise ValueError(f"Duplicate BigBind members for {example_lig_file}: {names}")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolution_csv = output_dir / "lig_file_member_resolution.csv"
    paper_map_csv = output_dir / "paper_row_member_map.csv"
    with resolution_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["lig_file", "member_name", "member_size", "member_mtime", "output_relative_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lig_file in required:
            member = matches[lig_file][0]
            writer.writerow(
                {
                    "lig_file": lig_file,
                    "member_name": member["name"],
                    "member_size": member["size"],
                    "member_mtime": member.get("mtime", ""),
                    "output_relative_path": _normalize_member_key(lig_file),
                }
            )
    by_lig = {lig_file: matches[lig_file][0] for lig_file in required}
    with paper_map_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["paper_row_id", "target_id", "paper_class", "paper_label", "source_index", "lig_file", "member_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            lig_file = str(row["lig_file"]).strip().replace("\\", "/")
            out = {field: row.get(field, "") for field in fields[:-1]}
            out["member_name"] = by_lig[lig_file]["name"]
            writer.writerow(out)
    manifest = {
        "schema_version": 1,
        "campaign_dir": str(campaign_dir),
        "inventory_path": str(inventory_path),
        "paper_row_count": len(rows),
        "unique_lig_file_count": len(required),
        "missing_count": 0,
        "duplicate_count": 0,
        "resolution_csv": str(resolution_csv),
        "paper_row_member_map_csv": str(paper_map_csv),
        "strict_policy": "missing, duplicate, unsafe, malformed, or hash-drifting members fail closed",
    }
    manifest_path = output_dir / "lig_file_member_resolution_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def extract_exact_bayesbind_ligands(
    campaign_dir: Path,
    *,
    archive_path: Path | None = None,
    inventory_path: Path | None = None,
    output_dir: Path | None = None,
    expected_archive_sha256: str | None = BIGBIND_ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Selectively extract required BayesBind ligand SDF members from BigBind."""
    campaign_dir = campaign_dir.expanduser().resolve()
    archive_path = (
        archive_path.expanduser().resolve()
        if archive_path is not None
        else campaign_dir / "inputs" / "BigBindV1.5.tar.gz"
    )
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else campaign_dir / "inputs" / "exact_ligands"
    )
    archive_hash = _sha256(archive_path)
    if expected_archive_sha256 is not None and archive_hash != expected_archive_sha256:
        raise ValueError(f"BigBind archive SHA256 mismatch: {archive_hash}")
    resolution = resolve_exact_bayesbind_ligand_members(
        campaign_dir,
        inventory_path=inventory_path,
        output_dir=output_dir,
    )
    resolution_csv = Path(str(resolution["resolution_csv"]))
    with resolution_csv.open(newline="", encoding="utf-8") as handle:
        resolution_rows = list(csv.DictReader(handle))
    needed_by_member = {row["member_name"]: row for row in resolution_rows}
    existing_manifest = output_dir / "exact_ligand_extraction_manifest.json"
    previous_hashes: dict[str, str] = {}
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        previous_hashes = {
            str(row["lig_file"]): str(row["sha256"])
            for row in previous.get("ligands", [])
            if "lig_file" in row and "sha256" in row
        }

    extracted_rows: list[dict[str, Any]] = []
    found: set[str] = set()
    output_root = output_dir / "sdf"
    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar:
            if member.name not in needed_by_member:
                continue
            if not member.isfile():
                raise ValueError(f"Required member is not a file: {member.name}")
            info = needed_by_member[member.name]
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read required member: {member.name}")
            data = handle.read()
            if len(data) != int(info["member_size"]):
                raise ValueError(f"Member size drift for {member.name}: {len(data)} != {info['member_size']}")
            _sdf_has_valid_molecule(data, info["lig_file"])
            digest = hashlib.sha256(data).hexdigest()
            if info["lig_file"] in previous_hashes and previous_hashes[info["lig_file"]] != digest:
                raise ValueError(f"Member hash drift for {info['lig_file']}")
            out_path = _safe_output_path(output_root, info["lig_file"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            extracted_rows.append(
                {
                    "lig_file": info["lig_file"],
                    "member_name": member.name,
                    "member_size": member.size,
                    "sha256": digest,
                    "output_path": str(out_path),
                }
            )
            found.add(member.name)
    missing_members = sorted(set(needed_by_member) - found)
    if missing_members:
        raise ValueError(f"Resolved members missing during extraction: {missing_members[:10]}")
    extracted_rows.sort(key=lambda row: row["lig_file"])
    ligand_hash_csv = output_dir / "exact_ligand_hashes.csv"
    _write_csv(
        ligand_hash_csv,
        extracted_rows,
        ["lig_file", "member_name", "member_size", "sha256", "output_path"],
    )
    manifest = {
        "schema_version": 1,
        "campaign_dir": str(campaign_dir),
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "paper_row_count": int(resolution["paper_row_count"]),
        "unique_lig_file_count": int(resolution["unique_lig_file_count"]),
        "extracted_ligand_count": len(extracted_rows),
        "ligand_hashes_csv": str(ligand_hash_csv),
        "resolution_manifest": str(resolution["manifest_path"]),
        "ligands": extracted_rows,
    }
    _write_json(existing_manifest, manifest)
    return {**manifest, "manifest_path": str(existing_manifest)}


def _select_stage_rows(
    rows: list[dict[str, str]],
    source_csv: Path,
    *,
    sample_size: int | None,
    sample_seed: int,
    source_row_start: int | None,
    source_row_count: int | None,
    selection_provenance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if sample_size is not None and (source_row_start is not None or source_row_count is not None):
        raise ValueError("sample_size cannot be combined with source row range")
    if (source_row_start is None) != (source_row_count is None):
        raise ValueError("source_row_start and source_row_count must be provided together")
    if source_row_start is not None and source_row_count is not None:
        if source_row_start < 0:
            raise ValueError("source_row_start must be non-negative")
        if source_row_count < 1:
            raise ValueError("source_row_count must be positive")
        end = source_row_start + source_row_count
        if end > len(rows):
            raise ValueError(f"source row range {source_row_start}:{end} exceeds {len(rows)} rows")
        selected = list(rows[source_row_start:end])
        selection = {
            "mode": "source_row_range",
            "source_row_start": source_row_start,
            "source_row_end_exclusive": end,
            "source_row_count": len(selected),
        }
    elif sample_size is not None:
        if sample_size < 2 or sample_size > len(rows):
            raise ValueError(f"sample_size must be between 2 and {len(rows)}")
        selected = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{sample_seed}:{row['source_compound_id']}".encode("utf-8")
            ).hexdigest(),
        )[:sample_size]
        if not any(int(row["label"]) == 1 for row in selected):
            with source_csv.open(newline="", encoding="utf-8") as handle:
                first_active = next(row for row in csv.DictReader(handle) if int(row["label"]) == 1)
            selected[-1] = first_active
        selection = {
            "mode": "deterministic_hash_sample",
            "sample_size": len(selected),
            "sample_seed": sample_seed,
            "forced_active_if_needed": True,
        }
    else:
        selected = list(rows)
        selection = {
            "mode": "full_target",
            "source_row_start": 0,
            "source_row_end_exclusive": len(selected),
            "source_row_count": len(selected),
        }
    selected.sort(key=lambda row: row["source_compound_id"])
    if selection_provenance:
        selection["provenance"] = dict(selection_provenance)
    return selected, selection

def stage_vina_notebook(
    campaign_dir: Path,
    target_id: str,
    output_notebook: Path,
    *,
    run_id: str,
    sample_size: int | None = None,
    sample_seed: int = 42,
    use_exact_sdf: bool = False,
    ligand_workers: int = 1,
    source_row_start: int | None = None,
    source_row_count: int | None = None,
    selection_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage one paper-protocol Vina notebook and immutable user inputs."""
    from ezscreen.benchmark.vina_baseline import render_vina_full_redock_notebook

    campaign_dir = campaign_dir.expanduser().resolve()
    output_notebook = output_notebook.expanduser().resolve()
    targets = json.loads((campaign_dir / "targets.json").read_text(encoding="utf-8"))
    target = next(
        (row for row in targets if str(row.get("target_id")) == target_id), None
    )
    if target is None:
        raise ValueError(f"Unknown BayesBind target: {target_id}")

    source_csv = campaign_dir / "paper_inputs" / target_id / "source_compounds.csv"
    with source_csv.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    rows, row_selection = _select_stage_rows(
        all_rows,
        source_csv,
        sample_size=sample_size,
        sample_seed=sample_seed,
        source_row_start=source_row_start,
        source_row_count=source_row_count,
        selection_provenance=selection_provenance,
    )

    stage_dir = campaign_dir / "work" / run_id / "input"
    stage_dir.mkdir(parents=True, exist_ok=True)
    input_payloads: list[Path] = []
    ligand_input_mode = "sdf_manifest" if use_exact_sdf else "smiles"
    exact_sdf_manifest: Path | None = None
    exact_sdf_archive: Path | None = None
    exact_sdf_rows: list[dict[str, Any]] = []
    library = stage_dir / f"{run_id}_library.smi"
    if use_exact_sdf:
        exact_hash_csv = campaign_dir / "inputs" / "exact_ligands" / "exact_ligand_hashes.csv"
        if not exact_hash_csv.exists():
            raise FileNotFoundError("Run extract-bayesbind-ligands before exact SDF staging")
        with exact_hash_csv.open(newline="", encoding="utf-8") as handle:
            exact_by_lig = {row["lig_file"]: row for row in csv.DictReader(handle)}
        sdf_dir = stage_dir / "sdfs"
        sdf_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            lig_file = str(row.get("lig_file", "")).strip()
            exact = exact_by_lig.get(lig_file)
            if exact is None:
                raise ValueError(f"Exact SDF hash missing for {lig_file}")
            source_sdf = Path(exact["output_path"])
            if not source_sdf.exists():
                raise FileNotFoundError(f"Exact SDF file missing: {source_sdf}")
            staged_sdf = sdf_dir / f"{row['source_compound_id']}.sdf"
            shutil.copy2(source_sdf, staged_sdf)
            digest = _sha256(staged_sdf)
            if digest != exact["sha256"]:
                raise ValueError(f"Exact SDF hash drift for {lig_file}")
            exact_sdf_rows.append(
                {
                    "source_compound_id": row["source_compound_id"],
                    "target_id": row["target_id"],
                    "label": row["label"],
                    "canonical_smiles": row["canonical_smiles"],
                    "lig_file": lig_file,
                    "sdf_filename": staged_sdf.name,
                    "sdf_sha256": digest,
                }
            )
        exact_sdf_archive = stage_dir / f"{run_id}_sdfs.tar"
        with tarfile.open(exact_sdf_archive, "w") as archive:
            for staged_sdf in sorted(sdf_dir.glob("*.sdf")):
                archive.add(staged_sdf, arcname=f"sdfs/{staged_sdf.name}")
        input_payloads.append(exact_sdf_archive)
        exact_sdf_manifest = stage_dir / f"{run_id}_sdf_manifest.csv"
        _write_csv(
            exact_sdf_manifest,
            exact_sdf_rows,
            [
                "source_compound_id",
                "target_id",
                "label",
                "canonical_smiles",
                "lig_file",
                "sdf_filename",
                "sdf_sha256",
            ],
        )
        input_payloads.append(exact_sdf_manifest)
    else:
        library.write_text(
            "".join(
                f"{row['canonical_smiles']}\t{row['source_compound_id']}\n" for row in rows
            ),
            encoding="utf-8",
        )
        input_payloads.append(library)
    intended = stage_dir / f"{run_id}_intended_compounds.csv"
    _write_csv(
        intended,
        rows,
        [
            "source_compound_id",
            "target_id",
            "label",
            "label_source",
            "canonical_smiles",
            "lig_file",
            "cross_label_conflict_resolved",
        ],
    )

    repo_root = campaign_dir.parent.parent
    receptor_source = Path(str(target["receptor_pdb"]))
    if not receptor_source.is_absolute():
        receptor_source = repo_root / receptor_source
    receptor = stage_dir / f"{run_id}_rec.pdb"
    shutil.copy2(receptor_source, receptor)
    vina_source = campaign_dir / "inputs" / VINA_BINARY_FILENAME
    if _sha256(vina_source) != VINA_BINARY_SHA256:
        raise ValueError("Pinned Vina binary SHA256 mismatch")
    vina_binary = stage_dir / VINA_BINARY_FILENAME
    shutil.copy2(vina_source, vina_binary)

    rendered = render_vina_full_redock_notebook(
        run_id=run_id,
        shard_filenames=[exact_sdf_manifest.name if exact_sdf_manifest else library.name],
        box_center=list(target["box_center"]),
        box_size=list(target["box_size"]),
        prep_cfg={
            "enable_gpu_size_filter": False,
            "mmff_max_iters": 200,
            "force_field": "UFF",
            "enumerate_enabled": False,
        },
        exhaustiveness=8,
        num_modes=9,
        cpu=1,
        seed=sample_seed,
        vina_version=VINA_VERSION,
        vina_binary_filename=vina_binary.name,
        vina_binary_sha256=VINA_BINARY_SHA256,
        receptor_filename=receptor.name,
        receptor_prep="openbabel",
        center_ligand=True,
        ligand_box_padding=3.0,
        ligand_box_lexicographic=True,
        ligand_input_mode=ligand_input_mode,
        meeko_hydrate=False,
        vina_use_executable_defaults=use_exact_sdf,
        sdf_archive_filenames=[exact_sdf_archive.name] if exact_sdf_archive else None,
        ligand_workers=ligand_workers,
    )
    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    output_notebook.write_text(rendered, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "target_id": target_id,
        "sample_size": len(rows),
        "sample_seed": sample_seed,
        "active_count": sum(int(row["label"]) == 1 for row in rows),
        "random_count": sum(int(row["label"]) == 0 for row in rows),
        "row_selection": row_selection,
        "ligand_input_mode": ligand_input_mode,
        "paper_protocol": {
            "receptor_prep": "obabel -xr -ipdb rec.pdb -opdbqt -p 7",
            "vina_cpu": 1,
            "ligand_workers": int(ligand_workers),
            "vina_defaults": "executable defaults" if use_exact_sdf else {"exhaustiveness": 8, "num_modes": 9},
            "ligand_prep": (
                "released 3D SDF + Chem.AddHs(addCoords=True) + Meeko MoleculePreparation(hydrate=False)"
                if use_exact_sdf
                else "deterministic ETKDGv3/UFF reconstruction"
            ),
            "ligand_box_padding": 3.0,
            "box_tuple_selection": "Python lexicographic max, matching released code",
        },
        "declared_deviation": (
            "none: exact BigBind SDF bytes staged"
            if use_exact_sdf
            else "Official BayesBind ligand SDF bytes were not included in BayesBindV1.5; "
            "this reconstruction deterministically embeds exact frozen canonical SMILES "
            "with ETKDGv3 and UFF before Meeko conversion."
        ),
        "exact_sdf_archive": exact_sdf_archive.name if exact_sdf_archive else None,
        "exact_sdf_archive_member_count": len(exact_sdf_rows) if exact_sdf_archive else 0,
        "files_sha256": {
            intended.name: _sha256(intended),
            receptor.name: _sha256(receptor),
            vina_binary.name: _sha256(vina_binary),
            output_notebook.name: _sha256(output_notebook),
            **{path.name: _sha256(path) for path in input_payloads},
        },
    }
    manifest_path = stage_dir / "stage_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "notebook": str(output_notebook),
        "stage_dir": str(stage_dir),
        "inputs": [str(path) for path in [*input_payloads, receptor, vina_binary, intended]],
        "manifest": str(manifest_path),
        **{key: manifest[key] for key in ("run_id", "target_id", "sample_size", "active_count", "random_count")},
    }


def render_gnina_full_docking_notebook(
    *,
    run_id: str,
    manifest_filename: str,
    sdf_archive_filename: str,
    receptor_filename: str,
    box_center: list[float],
    box_size: list[float],
    gnina_workers: int = 1,
    gnina_version: str = GNINA_VERSION,
    gnina_url: str = GNINA_BINARY_URL,
    cnn_model: str = GNINA_CNN_MODEL,
) -> str:
    '''Render a Kaggle notebook for paper-comparator GNINA full docking.'''
    center_py = repr([float(v) for v in box_center])
    size_py = repr([float(v) for v in box_size])
    workers = max(1, int(gnina_workers))
    header = (
        "# AUTO-GENERATED BY ezscreen\n"
        f"# Run ID : {run_id}\n"
        f"# Engine : gnina-{gnina_version}-full-docking\n"
        f"# CNN    : {cnn_model}\n"
        "# Contract: raw receptor PDB + exact ligand SDF + released fixed box\n"
        f"print('GNINA full-docking comparator: {run_id}')\n"
    )
    install = r'''
import hashlib, json, os, pathlib, stat, subprocess, urllib.request

def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _run(args, timeout=600):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        pathlib.Path('/kaggle/working/error.json').write_text(json.dumps({
            'cell': 'install', 'cmd': args, 'returncode': result.returncode,
            'stderr': result.stderr[-4000:], 'stdout': result.stdout[-2000:],
        }, indent=2))
        raise RuntimeError(f"command failed: {args}: {result.stderr[-800:]}")
    return result

try:
    import rdkit  # noqa: F401
except ImportError:
    _run([os.sys.executable, '-m', 'pip', 'install', 'rdkit', '-q'], timeout=1200)

GNINA_VERSION = __GNINA_VERSION__
GNINA_URL = __GNINA_URL__
GNINA = pathlib.Path('/kaggle/working/gnina')
if not GNINA.exists():
    urllib.request.urlretrieve(GNINA_URL, GNINA)
    GNINA.chmod(GNINA.stat().st_mode | stat.S_IEXEC)
version = _run([str(GNINA), '--version'], timeout=120).stdout.strip()
pathlib.Path('/kaggle/working/gnina_install_manifest.json').write_text(json.dumps({
    'gnina_version_pin': GNINA_VERSION,
    'gnina_binary_url': GNINA_URL,
    'gnina_binary_path': str(GNINA),
    'gnina_binary_bytes': GNINA.stat().st_size,
    'gnina_binary_sha256': _file_sha256(GNINA),
    'gnina_version_stdout': version,
}, indent=2))
print('gnina:', version.splitlines()[0] if version else 'version unavailable')
'''
    install = install.replace('__GNINA_VERSION__', repr(gnina_version)).replace('__GNINA_URL__', repr(gnina_url))

    locate = r'''
import csv, hashlib, json, pathlib, shutil, tarfile

RUN_ID = __RUN_ID__
RECEPTOR_FILENAME = __RECEPTOR_FILENAME__
MANIFEST_FILENAME = __MANIFEST_FILENAME__
SDF_ARCHIVE_FILENAME = __SDF_ARCHIVE_FILENAME__
BOX_CENTER = __BOX_CENTER__
BOX_SIZE = __BOX_SIZE__
CNN_MODEL = __CNN_MODEL__
GNINA_WORKERS = __GNINA_WORKERS__

input_root = pathlib.Path('/kaggle/input')
preview = [str(p) for p in list(input_root.rglob('*'))[:30]]

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def _single_input(filename):
    hits = list(input_root.rglob(filename))
    if len(hits) != 1:
        pathlib.Path('/kaggle/working/error.json').write_text(json.dumps({
            'cell': 'inputs', 'message': 'input match count violation',
            'filename': filename, 'match_count': len(hits), 'preview': preview,
        }, indent=2))
        raise FileNotFoundError(f'{filename} matched {len(hits)} files')
    return hits[0]

RECEPTOR_PATH = _single_input(RECEPTOR_FILENAME)
MANIFEST_PATH = _single_input(MANIFEST_FILENAME)
EXACT_SDF_ROOT = pathlib.Path('/kaggle/working/exact_sdfs')
EXACT_SDF_ROOT.mkdir(exist_ok=True)

def _stage_exact_sdfs(archive_filename):
    hits = list(input_root.rglob(archive_filename))
    if len(hits) == 1:
        archive_path = hits[0]
        with tarfile.open(archive_path, 'r:*') as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                member_path = pathlib.PurePosixPath(member.name)
                if member_path.is_absolute() or '..' in member_path.parts:
                    raise ValueError(f'Unsafe SDF archive member: {member.name}')
                target_path = EXACT_SDF_ROOT / pathlib.Path(*member_path.parts)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f'Unreadable SDF archive member: {member.name}')
                target_path.write_bytes(source.read())
        return {'kind': 'archive', 'path': str(archive_path), 'bytes': archive_path.stat().st_size, 'sha256': _sha256(archive_path)}
    if len(hits) > 1:
        raise FileNotFoundError(f'{archive_filename} matched {len(hits)} files')
    extracted_name = pathlib.Path(archive_filename).with_suffix('').name
    roots = [p for p in input_root.rglob(extracted_name) if p.is_dir()]
    if len(roots) != 1:
        pathlib.Path('/kaggle/working/error.json').write_text(json.dumps({
            'cell': 'inputs', 'message': 'SDF archive or extracted directory match violation',
            'filename': archive_filename, 'archive_match_count': 0,
            'extracted_name': extracted_name, 'extracted_match_count': len(roots), 'preview': preview,
        }, indent=2))
        raise FileNotFoundError(f'{archive_filename} matched 0 files and {extracted_name} matched {len(roots)} dirs')
    root = roots[0]
    copied = 0
    for source_path in root.rglob('*.sdf'):
        rel = source_path.relative_to(root)
        target_path = EXACT_SDF_ROOT / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1
    return {'kind': 'kaggle_extracted_directory', 'path': str(root), 'bytes': None, 'sha256': None, 'copied_sdf_count': copied}

SDF_SOURCE = _stage_exact_sdfs(SDF_ARCHIVE_FILENAME)

rows = []
for row in csv.DictReader(MANIFEST_PATH.open(newline='', encoding='utf-8')):
    sdf_filename = row['sdf_filename']
    hits = list(EXACT_SDF_ROOT.rglob(pathlib.PurePosixPath(sdf_filename).name))
    if len(hits) != 1:
        raise FileNotFoundError(f'{sdf_filename} matched {len(hits)} extracted SDF files')
    sdf_path = hits[0]
    digest = _sha256(sdf_path)
    if digest != row.get('sdf_sha256'):
        raise ValueError(f"SDF hash drift for {row.get('source_compound_id')}: {digest}")
    rows.append({
        'ligand': row['source_compound_id'],
        'name': row['source_compound_id'],
        'smiles': row.get('canonical_smiles', ''),
        'exact_lig_file': row.get('lig_file', ''),
        'exact_sdf_sha256': digest,
        'sdf_path': str(sdf_path),
    })
if not rows:
    raise RuntimeError('no GNINA ligands staged')
with open('/kaggle/working/index.csv', 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['ligand', 'name', 'smiles', 'exact_lig_file', 'exact_sdf_sha256', 'sdf_path'])
    writer.writeheader()
    writer.writerows(rows)
pathlib.Path('/kaggle/working/input_manifest.json').write_text(json.dumps({
    'run_id': RUN_ID,
    'receptor_mode': 'raw_rec_pdb',
    'receptor_path': str(RECEPTOR_PATH),
    'receptor_bytes': RECEPTOR_PATH.stat().st_size,
    'receptor_sha256': _sha256(RECEPTOR_PATH),
    'sdf_source_kind': SDF_SOURCE['kind'],
    'sdf_source_path': SDF_SOURCE['path'],
    'sdf_archive': SDF_SOURCE['path'] if SDF_SOURCE['kind'] == 'archive' else None,
    'sdf_archive_bytes': SDF_SOURCE.get('bytes'),
    'sdf_archive_sha256': SDF_SOURCE.get('sha256'),
    'sdf_extracted_copied_count': SDF_SOURCE.get('copied_sdf_count'),
    'ligand_count': len(rows),
    'box_center': BOX_CENTER,
    'box_size': BOX_SIZE,
    'cnn_model': CNN_MODEL,
}, indent=2))
print('receptor raw PDB:', RECEPTOR_PATH)
print('ligands:', len(rows))
print('free disk GB:', round(shutil.disk_usage('/kaggle/working').free / 1e9, 2))
'''
    locate = (
        locate.replace('__RUN_ID__', repr(run_id))
        .replace('__RECEPTOR_FILENAME__', repr(receptor_filename))
        .replace('__MANIFEST_FILENAME__', repr(manifest_filename))
        .replace('__SDF_ARCHIVE_FILENAME__', repr(sdf_archive_filename))
        .replace('__BOX_CENTER__', center_py)
        .replace('__BOX_SIZE__', size_py)
        .replace('__CNN_MODEL__', repr(cnn_model))
        .replace('__GNINA_WORKERS__', repr(workers))
    )

    dock = r'''
import csv, json, math, pathlib, re, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import SDWriter

DOCKED = pathlib.Path('/kaggle/working/gnina_docked')
DOCKED.mkdir(exist_ok=True)
LOGS = pathlib.Path('/kaggle/working/gnina_logs')
LOGS.mkdir(exist_ok=True)
writer = SDWriter('/kaggle/working/poses.sdf')
score_rows = []
failed_rows = []

def _safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)[:180] or 'ligand'

def _prop(mol, name):
    return mol.GetProp(name) if mol.HasProp(name) else ''

def _float_prop(mol, name):
    value = _prop(mol, name)
    return float(value) if value not in {'', 'nan', 'None'} else math.nan

def _best_pose(out_path):
    supplier = Chem.SDMolSupplier(str(out_path), sanitize=False, removeHs=False)
    best = None
    best_score = math.inf
    for mol in supplier:
        if mol is None:
            continue
        try:
            score = _float_prop(mol, 'minimizedAffinity')
        except Exception:
            continue
        if math.isfinite(score) and score < best_score:
            best = mol
            best_score = score
    return best, best_score

def _dock_one(row):
    lig_id = row['ligand']
    stem = _safe_name(lig_id)
    out_path = DOCKED / f'{stem}.sdf'
    stdout_path = LOGS / f'{stem}.stdout.log'
    stderr_path = LOGS / f'{stem}.stderr.log'
    cmd = [
        str(GNINA), '-r', str(RECEPTOR_PATH), '-l', row['sdf_path'], '-o', str(out_path),
        '--center_x', str(BOX_CENTER[0]), '--center_y', str(BOX_CENTER[1]), '--center_z', str(BOX_CENTER[2]),
        '--size_x', str(BOX_SIZE[0]), '--size_y', str(BOX_SIZE[1]), '--size_z', str(BOX_SIZE[2]),
        '--cnn', CNN_MODEL,
    ]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception as exc:
        return {'ok': False, 'ligand': lig_id, 'name': row['name'], 'smiles': row['smiles'], 'raw_score': '', 'reason': f'gnina_exception:{type(exc).__name__}', 'message': str(exc)[-500:]}
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    seconds = round(time.perf_counter() - t0, 3)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        return {'ok': False, 'ligand': lig_id, 'name': row['name'], 'smiles': row['smiles'], 'raw_score': '', 'reason': 'gnina_failed', 'message': result.stderr[-500:] or result.stdout[-500:]}
    pose, score = _best_pose(out_path)
    if pose is None or not math.isfinite(score):
        return {'ok': False, 'ligand': lig_id, 'name': row['name'], 'smiles': row['smiles'], 'raw_score': '', 'reason': 'unparseable_minimizedAffinity', 'message': ''}
    return {'ok': True, 'ligand': lig_id, 'name': row['name'], 'smiles': row['smiles'], 'score': score, 'pose': pose, 'seconds': seconds}

start = time.perf_counter()
processed = 0
max_workers = min(max(1, int(GNINA_WORKERS)), len(rows))
print(f'GNINA workers: {max_workers}; CNN model: {CNN_MODEL}')
with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = [pool.submit(_dock_one, row) for row in rows]
    for future in as_completed(futures):
        result = future.result()
        processed += 1
        if result.get('ok'):
            mol = result['pose']
            mol.SetProp('_Name', result['name'])
            mol.SetProp('source_compound_id', result['name'])
            mol.SetProp('score', str(result['score']))
            writer.write(mol)
            score_rows.append({
                'ligand': result['ligand'], 'score': result['score'], 'name': result['name'], 'smiles': result['smiles'],
                'engine': 'gnina-local', 'seconds': result['seconds'], 'pose_retained': '1',
                'CNNscore': _prop(mol, 'CNNscore'), 'CNNaffinity': _prop(mol, 'CNNaffinity'),
            })
        else:
            failed_rows.append({k: result.get(k, '') for k in ['ligand', 'name', 'smiles', 'raw_score', 'reason']})
            if len(failed_rows) <= 3:
                print(f"{result.get('ligand', '')} failed: {result.get('message', '')}")
        if processed % 25 == 0 or processed == len(rows):
            progress = {
                'run_id': RUN_ID, 'processed': processed, 'total': len(rows), 'scored': len(score_rows),
                'failed': len(failed_rows), 'elapsed_seconds': round(time.perf_counter() - start, 3),
                'gnina_workers': max_workers, 'cnn_model': CNN_MODEL,
            }
            pathlib.Path('/kaggle/working/progress.json').write_text(json.dumps(progress, indent=2))
            print(progress)
writer.close()
score_rows.sort(key=lambda row: (float(row['score']), row['ligand']))
with open('/kaggle/working/scores.csv', 'w', newline='') as handle:
    writer_csv = csv.DictWriter(handle, fieldnames=['ligand', 'score', 'name', 'smiles', 'engine', 'seconds', 'pose_retained', 'CNNscore', 'CNNaffinity'])
    writer_csv.writeheader()
    writer_csv.writerows(score_rows)
if failed_rows:
    with open('/kaggle/working/failed_docking.csv', 'w', newline='') as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=['ligand', 'name', 'smiles', 'raw_score', 'reason'])
        writer_csv.writeheader()
        writer_csv.writerows(failed_rows)
summary = {
    'run_id': RUN_ID,
    'engine': f'gnina_{GNINA_VERSION}_full_docking',
    'receptor_mode': 'raw_rec_pdb',
    'ligand_input_mode': 'sdf_manifest',
    'cnn_model': CNN_MODEL,
    'score_col': 'score',
    'score_provenance': 'GNINA minimizedAffinity; lower is better',
    'gnina_command_options_provenance': 'explicit receptor, ligand, output, released fixed box, --cnn crossdock_default2018; remaining search options executable defaults',
    'gnina_workers': max_workers,
    'input_sdf_count': len(rows),
    'scored': len(score_rows),
    'failed_docking': len(failed_rows),
    'runtime_seconds': round(time.perf_counter() - start, 3),
}
pathlib.Path('/kaggle/working/gnina_baseline_summary.json').write_text(json.dumps(summary, indent=2))
pathlib.Path('/kaggle/working/results_meta.json').write_text(json.dumps({
    'score_type': 'gnina_minimized_affinity_kcal_mol',
    'score_col': 'score',
    'engine': 'gnina-local',
    'ligand_input_mode': 'sdf_manifest',
    'cnn_model': CNN_MODEL,
}, indent=2))
shutil.rmtree(DOCKED, ignore_errors=True)
pathlib.Path('/kaggle/working/done.flag').write_text('ok\n')
print('done:', summary)
'''
    nb = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11.0'},
        },
        'cells': [
            _code_cell(header, 'cell-01'),
            _code_cell(install, 'cell-02'),
            _code_cell(locate, 'cell-03'),
            _code_cell(dock, 'cell-04'),
        ],
    }
    return json.dumps(nb, indent=2)


def stage_gnina_notebook(
    campaign_dir: Path,
    target_id: str,
    output_notebook: Path,
    *,
    run_id: str,
    sample_size: int | None = None,
    sample_seed: int = 42,
    gnina_workers: int = 1,
    source_row_start: int | None = None,
    source_row_count: int | None = None,
    selection_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    '''Stage one BayesBind GNINA full-docking comparator notebook.'''
    campaign_dir = campaign_dir.expanduser().resolve()
    output_notebook = output_notebook.expanduser().resolve()
    targets = json.loads((campaign_dir / 'targets.json').read_text(encoding='utf-8'))
    target = next((row for row in targets if str(row.get('target_id')) == target_id), None)
    if target is None:
        raise ValueError(f'Unknown BayesBind target: {target_id}')
    source_csv = campaign_dir / 'paper_inputs' / target_id / 'source_compounds.csv'
    with source_csv.open(newline='', encoding='utf-8') as handle:
        all_rows = list(csv.DictReader(handle))
    rows, row_selection = _select_stage_rows(
        all_rows,
        source_csv,
        sample_size=sample_size,
        sample_seed=sample_seed,
        source_row_start=source_row_start,
        source_row_count=source_row_count,
        selection_provenance=selection_provenance,
    )

    exact_hash_csv = campaign_dir / 'inputs' / 'exact_ligands' / 'exact_ligand_hashes.csv'
    if not exact_hash_csv.exists():
        raise FileNotFoundError('Run extract-bayesbind-ligands before GNINA staging')
    with exact_hash_csv.open(newline='', encoding='utf-8') as handle:
        exact_by_lig = {row['lig_file']: row for row in csv.DictReader(handle)}
    stage_dir = campaign_dir / 'work' / run_id / 'input'
    stage_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = stage_dir / 'sdfs'
    sdf_dir.mkdir(parents=True, exist_ok=True)
    exact_sdf_rows: list[dict[str, Any]] = []
    for row in rows:
        lig_file = str(row.get('lig_file', '')).strip()
        exact = exact_by_lig.get(lig_file)
        if exact is None:
            raise ValueError(f'Exact SDF hash missing for {lig_file}')
        source_sdf = Path(exact['output_path'])
        if not source_sdf.exists():
            raise FileNotFoundError(f'Exact SDF file missing: {source_sdf}')
        staged_sdf = sdf_dir / f"{row['source_compound_id']}.sdf"
        shutil.copy2(source_sdf, staged_sdf)
        digest = _sha256(staged_sdf)
        if digest != exact['sha256']:
            raise ValueError(f'Exact SDF hash drift for {lig_file}')
        exact_sdf_rows.append({
            'source_compound_id': row['source_compound_id'],
            'target_id': row['target_id'],
            'label': row['label'],
            'canonical_smiles': row['canonical_smiles'],
            'lig_file': lig_file,
            'sdf_filename': staged_sdf.name,
            'sdf_sha256': digest,
        })
    sdf_archive = stage_dir / f'{run_id}_sdfs.tar'
    with tarfile.open(sdf_archive, 'w') as archive:
        for staged_sdf in sorted(sdf_dir.glob('*.sdf')):
            archive.add(staged_sdf, arcname=f'sdfs/{staged_sdf.name}')
    sdf_manifest = stage_dir / f'{run_id}_sdf_manifest.csv'
    _write_csv(
        sdf_manifest,
        exact_sdf_rows,
        ['source_compound_id', 'target_id', 'label', 'canonical_smiles', 'lig_file', 'sdf_filename', 'sdf_sha256'],
    )
    intended = stage_dir / f'{run_id}_intended_compounds.csv'
    _write_csv(
        intended,
        rows,
        ['source_compound_id', 'target_id', 'label', 'label_source', 'canonical_smiles', 'lig_file', 'cross_label_conflict_resolved'],
    )
    repo_root = campaign_dir.parent.parent
    receptor_source = Path(str(target['receptor_pdb']))
    if not receptor_source.is_absolute():
        receptor_source = repo_root / receptor_source
    receptor = stage_dir / f'{run_id}_rec.pdb'
    shutil.copy2(receptor_source, receptor)
    rendered = render_gnina_full_docking_notebook(
        run_id=run_id,
        manifest_filename=sdf_manifest.name,
        sdf_archive_filename=sdf_archive.name,
        receptor_filename=receptor.name,
        box_center=list(target['box_center']),
        box_size=list(target['box_size']),
        gnina_workers=gnina_workers,
    )
    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    output_notebook.write_text(rendered, encoding='utf-8')
    manifest = {
        'schema_version': 1,
        'run_id': run_id,
        'target_id': target_id,
        'sample_size': len(rows),
        'sample_seed': sample_seed,
        'active_count': sum(int(row['label']) == 1 for row in rows),
        'random_count': sum(int(row['label']) == 0 for row in rows),
        'row_selection': row_selection,
        'engine': 'gnina',
        'paper_protocol': {
            'receptor_prep': 'raw rec.pdb',
            'ligand_input': 'exact released BigBind SDF bytes',
            'box': 'released fixed pocket box',
            'cnn_model': GNINA_CNN_MODEL,
            'gnina_version': GNINA_VERSION,
            'gnina_defaults': 'executable defaults except explicit fixed box and --cnn crossdock_default2018',
            'gnina_workers': int(gnina_workers),
        },
        'declared_deviation': 'none: exact BigBind SDF bytes staged for GNINA comparator',
        'exact_sdf_archive': sdf_archive.name,
        'exact_sdf_archive_member_count': len(exact_sdf_rows),
        'files_sha256': {
            intended.name: _sha256(intended),
            receptor.name: _sha256(receptor),
            sdf_manifest.name: _sha256(sdf_manifest),
            sdf_archive.name: _sha256(sdf_archive),
            output_notebook.name: _sha256(output_notebook),
        },
    }
    manifest_path = stage_dir / 'stage_manifest.json'
    _write_json(manifest_path, manifest)
    return {
        'notebook': str(output_notebook),
        'stage_dir': str(stage_dir),
        'inputs': [str(path) for path in [sdf_archive, sdf_manifest, receptor, intended]],
        'manifest': str(manifest_path),
        **{key: manifest[key] for key in ('run_id', 'target_id', 'sample_size', 'active_count', 'random_count')},
    }


def finalize_gnina_output(output_dir: Path, intended_csv: Path) -> dict[str, Any]:
    '''Finalize GNINA full-docking output with the BayesBind source metric contract.'''
    result = finalize_vina_output(output_dir, intended_csv)
    metrics_path = Path(str(result['metrics_path']))
    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    payload['engine'] = 'gnina'
    payload['score_col'] = 'score'
    payload['score_provenance'] = 'GNINA minimizedAffinity; lower is better'
    payload['cnn_model'] = GNINA_CNN_MODEL
    _write_json(metrics_path, payload)
    return {**result, **payload, 'metrics_path': str(metrics_path)}



def finalize_vina_output(output_dir: Path, intended_csv: Path) -> dict[str, Any]:
    """Finalize one Vina shard with failed intended compounds at worst rank."""
    from ezscreen.benchmark.metrics import compute_metrics

    output_dir = output_dir.expanduser().resolve()
    intended_csv = intended_csv.expanduser().resolve()
    with intended_csv.open(newline="", encoding="utf-8") as handle:
        intended_rows = list(csv.DictReader(handle))
    with (output_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
        scored_rows = list(csv.DictReader(handle))
    index_path = output_dir / "index.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        prepared_rows = list(csv.DictReader(handle))
    filtered_path = output_dir / "filtered_gpu_size.csv"
    if filtered_path.exists():
        with filtered_path.open(newline="", encoding="utf-8") as handle:
            filtered_rows = list(csv.DictReader(handle))
    else:
        filtered_rows = []
    intended_by_id = {row["source_compound_id"]: row for row in intended_rows}
    scored_by_id: dict[str, dict[str, str]] = {}
    unmatched = 0
    for row in scored_rows:
        source_id = str(row.get("name") or "").strip()
        if source_id not in intended_by_id:
            unmatched += 1
            continue
        current = scored_by_id.get(source_id)
        if current is None or float(row["score"]) < float(current["score"]):
            scored_by_id[source_id] = row
    finite_scores = [float(row["score"]) for row in scored_by_id.values()]
    worst_score = math.nextafter(max(finite_scores), math.inf) if finite_scores else 0.0
    ranking: list[dict[str, Any]] = []
    for source_id, intended in intended_by_id.items():
        score_row = scored_by_id.get(source_id)
        ranking.append(
            {
                "source_compound_id": source_id,
                "target_id": intended["target_id"],
                "label": int(intended["label"]),
                "score": float(score_row["score"]) if score_row else worst_score,
                "scored": int(score_row is not None),
                "failed": int(score_row is None),
                "pose_retained": int(
                    bool(score_row) and str(score_row.get("pose_retained", "0")) == "1"
                ),
            }
        )
    ranking.sort(key=lambda row: (float(row["score"]), row["source_compound_id"]))
    metrics = compute_metrics(
        [float(row["score"]) for row in ranking],
        [int(row["label"]) for row in ranking],
        scored=[bool(row["scored"]) for row in ranking],
    )
    ranking_path = output_dir / "source_ranking.csv"
    _write_csv(
        ranking_path,
        ranking,
        [
            "source_compound_id",
            "target_id",
            "label",
            "score",
            "scored",
            "failed",
            "pose_retained",
        ],
    )
    payload = {
        "schema_version": 1,
        "target_id": intended_rows[0]["target_id"] if intended_rows else "",
        "ef1": metrics.ef1,
        "ef5": metrics.ef5,
        "ef20": metrics.ef20,
        "roc_auc": metrics.auc_roc,
        "bedroc": metrics.bedroc,
        "coverage": metrics.coverage,
        "first_active_rank": metrics.first_active_rank,
        "pose_retention": (
            round(
                sum(int(row["pose_retained"]) for row in ranking)
                / sum(int(row["scored"]) for row in ranking),
                6,
            )
            if any(int(row["scored"]) for row in ranking)
            else 0.0
        ),
        "counts": {
            "before_prep": len(intended_rows),
            "after_prep": len(prepared_rows),
            "after_aggregation": len(ranking),
            "scored": len(scored_by_id),
            "failed": len(intended_rows) - len(scored_by_id),
            "filtered": len(filtered_rows),
            "unmatched": unmatched,
        },
        "failed_intended_policy": "worst score with shared boundary tie",
        "tie_policy": "tie-aware AUC and expected-actives EF boundary",
    }
    metrics_path = output_dir / "target_metrics.json"
    _write_json(metrics_path, payload)
    return {**payload, "metrics_path": str(metrics_path), "ranking_path": str(ranking_path)}



def _safe_run_id_token(value: str) -> str:
    token = ''.join(ch.lower() if ch.isalnum() else '-' for ch in value)
    token = '-'.join(part for part in token.split('-') if part)
    return token or 'target'


def _load_bayesbind_targets(campaign_dir: Path) -> list[dict[str, Any]]:
    targets_path = campaign_dir / "targets.json"
    if not targets_path.is_file():
        raise FileNotFoundError(f"BayesBind targets.json missing: {targets_path}")
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    if not isinstance(targets, list):
        raise ValueError("BayesBind targets.json must contain a list")
    return targets


def freeze_bayesbind_full_run_plan(
    campaign_dir: Path,
    calibration_output_dir: Path,
    calibration_stage_manifest: Path,
    *,
    method: str = "vina",
    target_ids: list[str] | None = None,
    output_path: Path | None = None,
    account_count: int | None = None,
    max_notebooks_per_account: int | None = None,
    safety_factor: float = 1.25,
    max_shard_wall_hours: float = 11.0,
    minimum_calibration_compounds: int = 100,
    maximum_calibration_compounds: int = 250,
) -> dict[str, Any]:
    """Freeze calibration-derived BayesBind full-target shard and quota plan."""
    from ezscreen.benchmark.kaggle_planning import (
        distribute_notebooks_across_accounts,
        plan_kaggle_shards,
    )

    campaign_dir = campaign_dir.expanduser().resolve()
    calibration_output_dir = calibration_output_dir.expanduser().resolve()
    calibration_stage_manifest = calibration_stage_manifest.expanduser().resolve()
    method = method.strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("method must be vina or gnina")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be at least 1.0")
    if max_shard_wall_hours <= 0.0:
        raise ValueError("max_shard_wall_hours must be positive")
    if minimum_calibration_compounds < 1 or maximum_calibration_compounds < minimum_calibration_compounds:
        raise ValueError("invalid calibration compound bounds")

    stage_manifest = json.loads(calibration_stage_manifest.read_text(encoding="utf-8"))
    metrics_path = calibration_output_dir / "target_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Calibration metrics missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary_name = "gnina_baseline_summary.json" if method == "gnina" else "vina_baseline_summary.json"
    summary_path = calibration_output_dir / summary_name
    if not summary_path.is_file():
        raise FileNotFoundError(f"Calibration runtime summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    counts = metrics.get("counts") or {}
    calibration_count = int(counts.get("before_prep") or stage_manifest.get("sample_size") or 0)
    if not (minimum_calibration_compounds <= calibration_count <= maximum_calibration_compounds):
        raise ValueError(
            "Calibration must contain "
            f"{minimum_calibration_compounds}-{maximum_calibration_compounds} intended compounds; "
            f"observed {calibration_count}"
        )
    runtime_seconds = float(summary.get("runtime_seconds") or summary.get("wall_time_seconds") or 0.0)
    if runtime_seconds <= 0.0:
        raise ValueError("Calibration runtime_seconds must be positive")
    calibration_target_id = str(stage_manifest.get("target_id") or metrics.get("target_id") or "")
    if not calibration_target_id:
        raise ValueError("Calibration target_id missing")

    targets = _load_bayesbind_targets(campaign_dir)
    target_by_id = {str(row["target_id"]): row for row in targets}
    calibration_target = target_by_id.get(calibration_target_id)
    if calibration_target is None:
        raise ValueError(f"Calibration target not present in campaign targets: {calibration_target_id}")
    if str(calibration_target.get("split")) != "development":
        raise ValueError("Calibration target must be from the development split, never held-out")
    selected_ids = target_ids or [str(row["target_id"]) for row in targets]
    missing_targets = sorted(set(selected_ids) - set(target_by_id))
    if missing_targets:
        raise ValueError("Unknown BayesBind target id(s): " + ", ".join(missing_targets))

    quota_path = campaign_dir / "quota_budget.json"
    quota = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.is_file() else {}
    account_count = int(account_count or quota.get("account_count") or 1)
    compute_kind = "cpu" if method == "vina" else "gpu"
    max_notebooks_per_account = int(
        max_notebooks_per_account
        or quota.get(f"maximum_concurrent_{compute_kind}_notebooks_per_account")
        or quota.get("maximum_concurrent_notebooks_per_account")
        or (10 if compute_kind == "cpu" else 2)
    )
    maximum_planned_gpu_hours = float(quota.get("maximum_planned_gpu_hours", 100.0))
    reserve_gpu_hours = float(quota.get("reserve_gpu_hours", 20.0))
    existing_planned_gpu_hours = float(quota.get("planned_gpu_hours", 0.0))
    available_gpu_hours = max(0.0, maximum_planned_gpu_hours - reserve_gpu_hours - existing_planned_gpu_hours)
    stage_protocol = stage_manifest.get("paper_protocol") or {}
    execution_defaults = {
        "ligand_workers": int(stage_protocol.get("ligand_workers") or 1),
        "gnina_workers": int(stage_protocol.get("gnina_workers") or 1),
    }
    seconds_per_compound = runtime_seconds / calibration_count
    adjusted_seconds_per_compound = seconds_per_compound * safety_factor
    max_shard_wall_seconds = max_shard_wall_hours * 3600.0

    target_plans: list[dict[str, Any]] = []
    total_planned_gpu_hours = 0.0
    total_cpu_notebook_hours = 0.0
    runtime_cap_safe = True
    for target_id in selected_ids:
        target = target_by_id[target_id]
        source_count = int(target.get("source_compounds") or target.get("paper_intended_rows") or 0)
        if source_count <= 0:
            raise ValueError(f"Target has no source compound count: {target_id}")
        count_based_shard_plan = plan_kaggle_shards(
            source_count,
            account_count,
            max_notebooks_per_account=max_notebooks_per_account,
        )
        runtime_cap_notebooks = max(
            1,
            math.ceil(source_count * adjusted_seconds_per_compound / max_shard_wall_seconds),
        )
        capacity = account_count * max_notebooks_per_account
        total_notebooks = max(count_based_shard_plan.total_notebooks, runtime_cap_notebooks)
        if total_notebooks > capacity:
            runtime_cap_safe = False
            total_notebooks = capacity
        notebooks_per_account = distribute_notebooks_across_accounts(
            total_notebooks,
            account_count,
            max_notebooks_per_account=max_notebooks_per_account,
        )
        shard_size = math.ceil(source_count / total_notebooks)
        shards: list[dict[str, Any]] = []
        target_wall_hours = 0.0
        for index in range(total_notebooks):
            start = index * shard_size
            end = min(source_count, start + shard_size)
            shard_count = max(0, end - start)
            wall_seconds = shard_count * adjusted_seconds_per_compound
            gpu_hours = wall_seconds / 3600.0 if compute_kind == "gpu" else 0.0
            cpu_hours = wall_seconds / 3600.0 if compute_kind == "cpu" else 0.0
            target_wall_hours = max(target_wall_hours, wall_seconds / 3600.0)
            total_planned_gpu_hours += gpu_hours
            total_cpu_notebook_hours += cpu_hours
            shards.append(
                {
                    "shard_index": index + 1,
                    "shard_count": shard_count,
                    "source_row_start": start,
                    "source_row_end_exclusive": end,
                    "run_id": f"bayesbind-{method}-exact-sdf-full-{_safe_run_id_token(target_id)}-{index + 1:03d}-of-{total_notebooks:03d}",
                    "estimated_wall_seconds": round(wall_seconds, 3),
                    "planned_gpu_hours": round(gpu_hours, 6),
                }
            )
        target_plans.append(
            {
                "target_id": target_id,
                "position": int(target.get("position", 0)),
                "split": target.get("split"),
                "source_compounds": source_count,
                "total_notebooks": total_notebooks,
                "notebooks_per_account": list(notebooks_per_account),
                "count_based_notebooks": count_based_shard_plan.total_notebooks,
                "runtime_cap_notebooks": runtime_cap_notebooks,
                "runtime_cap_satisfied": runtime_cap_notebooks <= capacity,
                "estimated_target_wall_hours": round(target_wall_hours, 6),
                "planned_gpu_hours": round(sum(float(shard["planned_gpu_hours"]) for shard in shards), 6),
                "shards": shards,
            }
        )

    split_counts: dict[str, int] = {}
    for row in target_plans:
        split = str(row.get("split") or "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    total_planned_gpu_hours = round(total_planned_gpu_hours, 6)
    total_cpu_notebook_hours = round(total_cpu_notebook_hours, 6)
    quota_safe = total_planned_gpu_hours <= available_gpu_hours
    plan_ready = quota_safe and runtime_cap_safe
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else campaign_dir / "full_run_plans" / f"bayesbind_{method}_full_run_plan.json"
    )
    payload = {
        "schema_version": 1,
        "status": "ready" if plan_ready else "blocked",
        "method": method,
        "compute_kind": compute_kind,
        "campaign_dir": str(campaign_dir),
        "calibration": {
            "target_id": calibration_target_id,
            "split": calibration_target.get("split"),
            "stage_manifest": str(calibration_stage_manifest),
            "output_dir": str(calibration_output_dir),
            "summary_path": str(summary_path),
            "metrics_path": str(metrics_path),
            "intended_compounds": calibration_count,
            "runtime_seconds": runtime_seconds,
            "seconds_per_compound": round(seconds_per_compound, 6),
            "safety_factor": safety_factor,
            "adjusted_seconds_per_compound": round(adjusted_seconds_per_compound, 6),
            "max_shard_wall_hours": max_shard_wall_hours,
            "coverage": metrics.get("coverage"),
            "scored": counts.get("scored"),
            "failed": counts.get("failed"),
        },
        "quota_gate": {
            "maximum_planned_gpu_hours": maximum_planned_gpu_hours,
            "reserve_gpu_hours": reserve_gpu_hours,
            "existing_planned_gpu_hours": existing_planned_gpu_hours,
            "available_gpu_hours_after_reserve": round(available_gpu_hours, 6),
            "planned_gpu_hours": total_planned_gpu_hours,
            "quota_safe": quota_safe,
        },
        "runtime_gate": {
            "max_shard_wall_hours": max_shard_wall_hours,
            "max_shard_wall_seconds": round(max_shard_wall_seconds, 3),
            "runtime_cap_safe": runtime_cap_safe,
        },
        "execution_defaults": execution_defaults,
        "planning": {
            "account_count": account_count,
            "maximum_concurrent_notebooks_per_account": max_notebooks_per_account,
            "target_count": len(target_plans),
            "split_counts": split_counts,
            "total_notebooks": sum(int(row["total_notebooks"]) for row in target_plans),
            "total_source_compounds": sum(int(row["source_compounds"]) for row in target_plans),
            "estimated_cpu_notebook_hours": total_cpu_notebook_hours,
            "uses_held_out_for_tuning": False,
            "held_out_policy": "Shard sizes and quota are frozen from development calibration; held-out targets are not tuned.",
        },
        "targets": target_plans,
    }
    _write_json(output_path, payload)
    return {**payload, "plan_path": str(output_path)}


def stage_bayesbind_plan_shard(
    campaign_dir: Path,
    plan_path: Path,
    target_id: str,
    shard_index: int,
    *,
    output_notebook: Path | None = None,
    ligand_workers: int | None = None,
    gnina_workers: int | None = None,
) -> dict[str, Any]:
    """Render one exact-SDF full-run notebook shard from a frozen plan."""
    campaign_dir = campaign_dir.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if str(plan.get("status")) != "ready":
        raise ValueError(f"Full-run plan is not ready: {plan.get('status')}")
    method = str(plan.get("method") or "").strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("Full-run plan method must be vina or gnina")
    targets = plan.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Full-run plan targets must be a list")
    target_plan = next((row for row in targets if str(row.get("target_id")) == target_id), None)
    if target_plan is None:
        raise ValueError(f"Target not present in full-run plan: {target_id}")
    shards = target_plan.get("shards")
    if not isinstance(shards, list):
        raise ValueError("Full-run plan target shards must be a list")
    shard = next((row for row in shards if int(row.get("shard_index") or 0) == int(shard_index)), None)
    if shard is None:
        raise ValueError(f"Shard {shard_index} not present for target {target_id}")
    run_id = str(shard.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Full-run plan shard is missing run_id")
    source_row_start = int(shard["source_row_start"])
    source_row_end = int(shard["source_row_end_exclusive"])
    source_row_count = source_row_end - source_row_start
    if source_row_count <= 0:
        raise ValueError("Full-run plan shard row count must be positive")
    if source_row_count != int(shard.get("shard_count") or source_row_count):
        raise ValueError("Full-run plan shard row count is inconsistent")
    output_notebook = (
        output_notebook.expanduser().resolve()
        if output_notebook is not None
        else campaign_dir / "notebooks" / f"{run_id}.ipynb"
    )
    plan_sha = _sha256(plan_path)
    provenance = {
        "source": "bayesbind_full_run_plan",
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "plan_status": plan.get("status"),
        "shard_index": int(shard_index),
        "planned_gpu_hours": shard.get("planned_gpu_hours"),
        "estimated_wall_seconds": shard.get("estimated_wall_seconds"),
    }
    defaults = plan.get("execution_defaults") or {}
    sample_seed = int((plan.get("calibration") or {}).get("sample_seed") or 42)
    if method == "vina":
        result = stage_vina_notebook(
            campaign_dir,
            target_id,
            output_notebook,
            run_id=run_id,
            sample_size=None,
            sample_seed=sample_seed,
            use_exact_sdf=True,
            ligand_workers=int(ligand_workers or defaults.get("ligand_workers") or 1),
            source_row_start=source_row_start,
            source_row_count=source_row_count,
            selection_provenance=provenance,
        )
    else:
        result = stage_gnina_notebook(
            campaign_dir,
            target_id,
            output_notebook,
            run_id=run_id,
            sample_size=None,
            sample_seed=sample_seed,
            gnina_workers=int(gnina_workers or defaults.get("gnina_workers") or 1),
            source_row_start=source_row_start,
            source_row_count=source_row_count,
            selection_provenance=provenance,
        )
    manifest_path = Path(str(result["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_run_plan"] = {
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "method": method,
        "target_id": target_id,
        "shard_index": int(shard_index),
        "shard_count": source_row_count,
        "planned_gpu_hours": shard.get("planned_gpu_hours"),
        "estimated_wall_seconds": shard.get("estimated_wall_seconds"),
    }
    manifest.setdefault("files_sha256", {})[plan_path.name] = plan_sha
    _write_json(manifest_path, manifest)
    return {
        **result,
        "method": method,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "shard_index": int(shard_index),
        "source_row_start": source_row_start,
        "source_row_end_exclusive": source_row_end,
        "planned_gpu_hours": shard.get("planned_gpu_hours"),
    }

def finalize_bayesbind_plan_target(
    campaign_dir: Path,
    plan_path: Path,
    target_id: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge completed planned shards into one target-level BayesBind result."""
    from ezscreen.benchmark.metrics import compute_metrics

    campaign_dir = campaign_dir.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    method = str(plan.get("method") or "").strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("Full-run plan method must be vina or gnina")
    selected = _planned_shards_from_plan(plan, target_ids=[target_id])
    target_plan = next(
        row for row in plan["targets"] if str(row.get("target_id")) == target_id
    )
    source_csv = campaign_dir / "paper_inputs" / target_id / "source_compounds.csv"
    with source_csv.open(newline="", encoding="utf-8") as handle:
        intended_rows = list(csv.DictReader(handle))
    intended_by_id = {row["source_compound_id"]: row for row in intended_rows}
    if len(intended_by_id) != len(intended_rows):
        raise ValueError(f"Duplicate intended source_compound_id in {source_csv}")
    expected_count = int(target_plan.get("source_compounds") or len(intended_rows))
    if expected_count != len(intended_rows):
        raise ValueError(
            f"Plan target source count {expected_count} does not match intended CSV {len(intended_rows)}"
        )

    state_path = campaign_dir / "campaign_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Campaign state missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("campaign_state jobs must be a list")
    complete_jobs = {
        str(job.get("run_id") or job.get("job_id")): job
        for job in jobs
        if str(job.get("status")) == "complete"
    }

    ranking_by_id: dict[str, dict[str, Any]] = {}
    shard_rows: list[dict[str, Any]] = []
    shard_counts = {
        "after_prep": 0,
        "filtered": 0,
        "unmatched": 0,
    }
    for _target_id, shard_index, shard in selected:
        run_id = str(shard.get("run_id") or "").strip()
        job = complete_jobs.get(run_id)
        if job is None:
            raise ValueError(f"Planned shard is not complete in campaign_state: {run_id}")
        output_dir_raw = str(job.get("output_dir") or "").strip()
        if not output_dir_raw:
            raise ValueError(f"Completed planned shard has no output_dir in campaign_state: {run_id}")
        output_path = Path(output_dir_raw).expanduser()
        if not output_path.is_absolute():
            output_path = campaign_dir / output_path
        if not output_path.is_dir():
            raise FileNotFoundError(f"Planned shard output missing: {output_path}")
        ranking_path = output_path / "source_ranking.csv"
        metrics_path = output_path / "target_metrics.json"
        if not ranking_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Planned shard is not finalized: {run_id}")
        shard_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        counts = shard_metrics.get("counts") or {}
        shard_counts["after_prep"] += int(counts.get("after_prep") or 0)
        shard_counts["filtered"] += int(counts.get("filtered") or 0)
        shard_counts["unmatched"] += int(counts.get("unmatched") or 0)
        with ranking_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_shard_count = int(shard.get("shard_count") or 0)
        if len(rows) != expected_shard_count:
            raise ValueError(
                f"Shard {run_id} ranking rows {len(rows)} != planned {expected_shard_count}"
            )
        for row in rows:
            source_id = str(row.get("source_compound_id") or "").strip()
            if source_id in ranking_by_id:
                raise ValueError(f"Duplicate source_compound_id across planned shards: {source_id}")
            if source_id not in intended_by_id:
                raise ValueError(f"Shard {run_id} produced source id outside target denominator: {source_id}")
            ranking_by_id[source_id] = row
        shard_rows.append(
            {
                "target_id": target_id,
                "method": method,
                "shard_index": shard_index,
                "run_id": run_id,
                "job_status": job.get("status"),
                "output_dir": str(output_path),
                "ranking_path": str(ranking_path),
                "metrics_path": str(metrics_path),
                "planned_gpu_hours": shard.get("planned_gpu_hours"),
                "planned_count": expected_shard_count,
                "ranking_rows": len(rows),
            }
        )
    missing_ids = sorted(set(intended_by_id) - set(ranking_by_id))
    extra_ids = sorted(set(ranking_by_id) - set(intended_by_id))
    if missing_ids or extra_ids:
        raise ValueError(
            "Planned shard coverage mismatch: "
            f"missing={len(missing_ids)} extra={len(extra_ids)}"
        )

    scored_values = [
        float(row["score"])
        for row in ranking_by_id.values()
        if str(row.get("scored")) == "1"
    ]
    worst_score = math.nextafter(max(scored_values), math.inf) if scored_values else 0.0
    ranking: list[dict[str, Any]] = []
    for source_id, intended in intended_by_id.items():
        row = ranking_by_id[source_id]
        scored = str(row.get("scored")) == "1"
        ranking.append(
            {
                "source_compound_id": source_id,
                "target_id": target_id,
                "label": int(intended["label"]),
                "score": float(row["score"]) if scored else worst_score,
                "scored": int(scored),
                "failed": int(not scored),
                "pose_retained": int(str(row.get("pose_retained")) == "1" and scored),
            }
        )
    ranking.sort(key=lambda row: (float(row["score"]), row["source_compound_id"]))
    metrics = compute_metrics(
        [float(row["score"]) for row in ranking],
        [int(row["label"]) for row in ranking],
        scored=[bool(row["scored"]) for row in ranking],
    )
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else campaign_dir / "final_results" / method / target_id
    )
    ranking_path = output_dir / "source_ranking.csv"
    metrics_path = output_dir / "target_metrics.json"
    shard_manifest_path = output_dir / "shard_manifest.csv"
    _write_csv(
        ranking_path,
        ranking,
        ["source_compound_id", "target_id", "label", "score", "scored", "failed", "pose_retained"],
    )
    _write_csv(
        shard_manifest_path,
        shard_rows,
        [
            "target_id",
            "method",
            "shard_index",
            "run_id",
            "job_status",
            "output_dir",
            "ranking_path",
            "metrics_path",
            "planned_gpu_hours",
            "planned_count",
            "ranking_rows",
        ],
    )
    scored_count = sum(int(row["scored"]) for row in ranking)
    payload = {
        "schema_version": 1,
        "target_id": target_id,
        "method": method,
        "engine": "autodock_vina_1.2.5_full_redock" if method == "vina" else "gnina",
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "ef1": metrics.ef1,
        "ef5": metrics.ef5,
        "ef20": metrics.ef20,
        "roc_auc": metrics.auc_roc,
        "bedroc": metrics.bedroc,
        "coverage": metrics.coverage,
        "first_active_rank": metrics.first_active_rank,
        "pose_retention": (
            round(sum(int(row["pose_retained"]) for row in ranking) / scored_count, 6)
            if scored_count
            else 0.0
        ),
        "counts": {
            "before_prep": len(intended_rows),
            "after_prep": shard_counts["after_prep"],
            "after_aggregation": len(ranking),
            "scored": scored_count,
            "failed": len(ranking) - scored_count,
            "filtered": shard_counts["filtered"],
            "unmatched": shard_counts["unmatched"],
            "planned_shards": len(selected),
            "completed_shards": len(shard_rows),
        },
        "failed_intended_policy": "worst score with shared boundary tie across full target",
        "tie_policy": "tie-aware AUC and expected-actives EF boundary",
        "outputs": {
            "source_ranking_csv": str(ranking_path),
            "shard_manifest_csv": str(shard_manifest_path),
        },
    }
    if method == "gnina":
        payload["score_provenance"] = "GNINA minimizedAffinity; lower is better"
        payload["cnn_model"] = GNINA_CNN_MODEL
    else:
        payload["score_provenance"] = "AutoDock Vina affinity; lower is better"
    _write_json(metrics_path, payload)
    finalized_targets = state.setdefault("finalized_targets", {})
    method_targets = finalized_targets.setdefault(method, {})
    method_targets[target_id] = {
        "schema_version": 1,
        "target_id": target_id,
        "method": method,
        "metrics_path": str(metrics_path),
        "ranking_path": str(ranking_path),
        "shard_manifest_csv": str(shard_manifest_path),
        "plan_path": str(plan_path),
        "plan_sha256": payload["plan_sha256"],
        "counts": payload["counts"],
        "coverage": payload["coverage"],
        "roc_auc": payload["roc_auc"],
        "ef1": payload["ef1"],
        "ef5": payload["ef5"],
        "ef20": payload["ef20"],
        "bedroc": payload["bedroc"],
        "pose_retention": payload["pose_retention"],
    }
    _write_json(state_path, state)
    return {**payload, "metrics_path": str(metrics_path), "ranking_path": str(ranking_path)}



def _planned_shards_from_plan(
    plan: dict[str, Any],
    *,
    target_ids: list[str] | None = None,
    shard_indices: list[int] | None = None,
) -> list[tuple[str, int, dict[str, Any]]]:
    if str(plan.get("status")) != "ready":
        raise ValueError(f"Full-run plan is not ready: {plan.get('status')}")
    targets = plan.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Full-run plan targets must be a list")
    selected_targets = set(target_ids or [str(row.get("target_id")) for row in targets])
    selected_shards = {int(index) for index in shard_indices or []}
    unknown_targets = selected_targets - {str(row.get("target_id")) for row in targets}
    if unknown_targets:
        raise ValueError("Target not present in full-run plan: " + ", ".join(sorted(unknown_targets)))
    planned: list[tuple[str, int, dict[str, Any]]] = []
    for target in targets:
        target_id = str(target.get("target_id"))
        if target_id not in selected_targets:
            continue
        shards = target.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"Full-run plan target shards must be a list: {target_id}")
        for shard in sorted(shards, key=lambda row: int(row.get("shard_index") or 0)):
            shard_index = int(shard.get("shard_index") or 0)
            if selected_shards and shard_index not in selected_shards:
                continue
            planned.append((target_id, shard_index, shard))
    if not planned:
        raise ValueError("No plan shards selected")
    return planned


def submit_bayesbind_plan_shards(
    campaign_dir: Path,
    plan_path: Path,
    *,
    target_ids: list[str] | None = None,
    shard_indices: list[int] | None = None,
    backend: str | None = None,
    account_name: str | None = None,
    stage: str = "baseline",
    max_submissions: int | None = None,
    resume_blocked: bool = False,
) -> dict[str, Any]:
    """Stage and submit exact-SDF BayesBind shards from a ready full-run plan."""
    from ezscreen.benchmark.kaggle_notebook import submit_notebook

    campaign_dir = campaign_dir.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    method = str(plan.get("method") or "").strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("Full-run plan method must be vina or gnina")
    selected = _planned_shards_from_plan(
        plan,
        target_ids=target_ids,
        shard_indices=shard_indices,
    )
    if max_submissions is not None:
        if max_submissions < 1:
            raise ValueError("max_submissions must be positive")
        selected = selected[:max_submissions]
    submit_backend = backend or ("cpu" if method == "vina" else "t4x2")
    finalizer = "bayesbind-vina" if method == "vina" else "bayesbind-gnina"
    submissions: list[dict[str, Any]] = []
    first = True
    for target_id, shard_index, shard in selected:
        staged = stage_bayesbind_plan_shard(
            campaign_dir,
            plan_path,
            target_id,
            shard_index,
        )
        stage_dir = Path(str(staged["stage_dir"])).expanduser().resolve()
        run_id = str(staged["run_id"])
        intended_csv = stage_dir / f"{run_id}_intended_compounds.csv"
        result = submit_notebook(
            campaign_dir,
            Path(str(staged["notebook"])),
            [Path(path) for path in staged["inputs"]],
            run_id=run_id,
            job_id=run_id,
            stage=stage,
            backend=submit_backend,
            account_name=account_name,
            planned_gpu_hours=float(shard.get("planned_gpu_hours") or 0.0),
            resume_blocked=bool(resume_blocked and first),
            finalizer=finalizer,
            intended_csv=intended_csv,
        )
        first = False
        submissions.append(
            {
                **result,
                "target_id": target_id,
                "shard_index": shard_index,
                "stage_manifest": staged["manifest"],
                "intended_csv": str(intended_csv),
                "planned_gpu_hours": float(shard.get("planned_gpu_hours") or 0.0),
            }
        )
    summary = {
        "schema_version": 1,
        "status": "submitted",
        "method": method,
        "backend": submit_backend,
        "campaign_dir": str(campaign_dir),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "submitted_count": len(submissions),
        "planned_gpu_hours": round(sum(float(row.get("planned_gpu_hours") or 0.0) for row in submissions), 6),
        "submissions": submissions,
    }
    return summary




def preflight_bayesbind_execution(
    campaign_dir: Path,
    *,
    method: str = "vina",
    expected_target_count: int = 25,
    minimum_calibration_compounds: int = 100,
    maximum_calibration_compounds: int = 250,
    minimum_free_gb_warning: float = 50.0,
    plan_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Report exact BayesBind execution readiness without using credentials."""
    campaign_dir = campaign_dir.expanduser().resolve()
    method = method.strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("method must be vina or gnina")
    gates: list[dict[str, Any]] = []
    issues: list[str] = []
    warnings: list[str] = []

    def add_gate(name: str, status: str, detail: str = "", **extra: Any) -> None:
        gates.append({"name": name, "status": status, "detail": detail, **extra})
        if status == "fail":
            issues.append(f"{name}: {detail}")
        elif status == "warn":
            warnings.append(f"{name}: {detail}")

    if not campaign_dir.is_dir():
        raise FileNotFoundError(f"Campaign directory missing: {campaign_dir}")
    usage = shutil.disk_usage(campaign_dir)
    free_gb = usage.free / (1024 ** 3)
    add_gate(
        "disk_free",
        "pass" if free_gb >= minimum_free_gb_warning else "warn",
        f"{free_gb:.2f} GiB free",
        free_bytes=usage.free,
    )

    targets_path = campaign_dir / "targets.json"
    targets: list[dict[str, Any]] = []
    target_by_id: dict[str, dict[str, Any]] = {}
    if not targets_path.is_file():
        add_gate("targets", "fail", f"missing {targets_path}")
    else:
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        if not isinstance(targets, list):
            add_gate("targets", "fail", "targets.json must contain a list")
            targets = []
        else:
            target_by_id = {str(row.get("target_id")): row for row in targets}
            duplicate_count = len(targets) - len(target_by_id)
            if len(targets) != expected_target_count or duplicate_count:
                add_gate(
                    "targets",
                    "fail",
                    f"target_count={len(targets)} expected={expected_target_count} duplicates={duplicate_count}",
                )
            else:
                add_gate("targets", "pass", f"{len(targets)} frozen targets")

    source_total = 0
    source_missing = 0
    for target in targets:
        target_id = str(target.get("target_id"))
        source_csv = campaign_dir / "paper_inputs" / target_id / "source_compounds.csv"
        if not source_csv.is_file():
            source_missing += 1
            continue
        with source_csv.open(newline="", encoding="utf-8") as handle:
            source_total += sum(1 for _row in csv.DictReader(handle))
    if source_missing:
        add_gate("source_compounds", "fail", f"missing source_compounds.csv for {source_missing} target(s)")
    elif targets:
        add_gate("source_compounds", "pass", f"{source_total} source compounds across targets")

    inputs_dir = campaign_dir / "inputs"
    bigbind_archive = inputs_dir / "BigBindV1.5.tar.gz"
    baselines_archive = inputs_dir / "BayesBindBaselinesV1.5.tar.gz"
    archive_checks = [
        ("BigBind archive", bigbind_archive, BIGBIND_ARCHIVE_BYTES, BIGBIND_ARCHIVE_SHA256),
        ("BayesBind baselines archive", baselines_archive, BASELINES_ARCHIVE_BYTES, BASELINES_ARCHIVE_SHA256),
    ]
    for name, archive_path, expected_size, expected_sha in archive_checks:
        if not archive_path.is_file():
            add_gate(name, "fail", f"missing {archive_path}")
            continue
        size = archive_path.stat().st_size
        if size != expected_size:
            add_gate(name, "fail", f"size {size} != expected {expected_size}")
        else:
            add_gate(name, "pass", f"size {size} bytes", expected_sha256=expected_sha)

    archive_manifest = inputs_dir / "archive_manifests" / "official_archives_manifest.json"
    if not archive_manifest.is_file():
        add_gate("archive inventory", "fail", f"missing {archive_manifest}")
    else:
        manifest = json.loads(archive_manifest.read_text(encoding="utf-8"))
        archives = manifest.get("archives") if isinstance(manifest, dict) else None
        if not isinstance(archives, list) or len(archives) < 2:
            add_gate("archive inventory", "fail", "official archive manifest incomplete")
        else:
            add_gate("archive inventory", "pass", f"{len(archives)} archive records")

    extraction_manifest_path = inputs_dir / "exact_ligands" / "exact_ligand_extraction_manifest.json"
    resolution_manifest_path = inputs_dir / "exact_ligands" / "lig_file_member_resolution_manifest.json"
    exact_ligand_hashes = inputs_dir / "exact_ligands" / "exact_ligand_hashes.csv"
    if not extraction_manifest_path.is_file() or not resolution_manifest_path.is_file() or not exact_ligand_hashes.is_file():
        add_gate("exact_ligands", "fail", "exact ligand extraction manifests or hashes missing")
    else:
        extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
        resolution = json.loads(resolution_manifest_path.read_text(encoding="utf-8"))
        extracted = int(extraction.get("extracted_ligand_count") or 0)
        unique = int(extraction.get("unique_lig_file_count") or 0)
        missing = int(resolution.get("missing_count") or 0)
        duplicate = int(resolution.get("duplicate_count") or 0)
        if extracted != unique or missing or duplicate:
            add_gate(
                "exact_ligands",
                "fail",
                f"extracted={extracted} unique={unique} missing={missing} duplicate={duplicate}",
            )
        else:
            add_gate("exact_ligands", "pass", f"{extracted} exact SDFs extracted")

    official_manifest_path = inputs_dir / "official_baselines" / method / "official_baseline_manifest.json"
    if not official_manifest_path.is_file():
        add_gate("official_reference", "warn", f"missing {official_manifest_path}")
    else:
        official = json.loads(official_manifest_path.read_text(encoding="utf-8"))
        official_status = str(official.get("status") or "")
        status = "pass" if official_status == "ready" else "warn"
        add_gate("official_reference", status, f"status={official_status}")

    quota_path = campaign_dir / "quota_budget.json"
    if not quota_path.is_file():
        add_gate("quota", "warn", f"missing {quota_path}")
        quota: dict[str, Any] = {}
    else:
        quota = json.loads(quota_path.read_text(encoding="utf-8"))
        max_gpu = float(quota.get("maximum_planned_gpu_hours") or 100.0)
        reserve = float(quota.get("reserve_gpu_hours") or 20.0)
        planned = float(quota.get("planned_gpu_hours") or 0.0)
        observed = float(quota.get("observed_gpu_hours") or 0.0)
        if planned > max_gpu - reserve or observed > max_gpu - reserve:
            add_gate("quota", "fail", f"planned={planned} observed={observed} reserve={reserve} max={max_gpu}")
        else:
            add_gate("quota", "pass", f"planned_gpu={planned} observed_gpu={observed} reserve={reserve}")

    state_path = campaign_dir / "campaign_state.json"
    jobs: list[dict[str, Any]] = []
    if not state_path.is_file():
        add_gate("campaign_state", "warn", f"missing {state_path}")
        state: dict[str, Any] = {}
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        raw_jobs = state.get("jobs") or []
        if not isinstance(raw_jobs, list):
            add_gate("campaign_state", "fail", "campaign_state jobs must be a list")
        else:
            jobs = raw_jobs
            add_gate("campaign_state", "pass", f"{len(jobs)} registered job(s)")

    work_dir = campaign_dir / "work"
    calibration_manifests = sorted(work_dir.glob(f"bayesbind-{method}-exact-sdf-calib-*/input/stage_manifest.json"))
    valid_calibration_manifests: list[dict[str, Any]] = []
    for manifest_path in calibration_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_size = int(manifest.get("sample_size") or 0)
        target_id = str(manifest.get("target_id") or "")
        target_split = str((target_by_id.get(target_id) or {}).get("split") or "")
        exact_members = int(manifest.get("exact_sdf_archive_member_count") or 0)
        valid = (
            minimum_calibration_compounds <= sample_size <= maximum_calibration_compounds
            and target_split == "development"
            and exact_members == sample_size
        )
        valid_calibration_manifests.append(
            {
                "path": str(manifest_path),
                "run_id": str(manifest.get("run_id") or manifest_path.parents[1].name),
                "target_id": target_id,
                "target_split": target_split,
                "sample_size": sample_size,
                "exact_sdf_archive_member_count": exact_members,
                "valid": valid,
            }
        )
    if not valid_calibration_manifests:
        add_gate("calibration_staged", "fail", f"no {method} calibration stage_manifest found")
    elif not any(row["valid"] for row in valid_calibration_manifests):
        add_gate("calibration_staged", "fail", "calibration staged but no manifest satisfies sample/split/exact-SDF gates")
    else:
        best_calibration = next(row for row in valid_calibration_manifests if row["valid"])
        add_gate("calibration_staged", "pass", f"{best_calibration['run_id']} sample={best_calibration['sample_size']}")

    complete_jobs = {str(job.get("run_id") or job.get("job_id")): job for job in jobs if str(job.get("status")) == "complete"}
    complete_calibrations: list[dict[str, Any]] = []
    for row in valid_calibration_manifests:
        run_id = str(row["run_id"])
        job = complete_jobs.get(run_id)
        output_dir = Path(str((job or {}).get("output_dir") or work_dir / run_id / "output")).expanduser()
        if not output_dir.is_absolute():
            output_dir = campaign_dir / output_dir
        summary_name = "gnina_baseline_summary.json" if method == "gnina" else "vina_baseline_summary.json"
        if job and (output_dir / "target_metrics.json").is_file() and (output_dir / summary_name).is_file():
            complete_calibrations.append({**row, "output_dir": str(output_dir), "summary": str(output_dir / summary_name)})
    if complete_calibrations:
        add_gate("calibration_complete", "pass", complete_calibrations[0]["run_id"])
    else:
        add_gate("calibration_complete", "wait", "no completed finalized calibration job found")

    if plan_path is None:
        default_plan = campaign_dir / "full_run_plans" / f"bayesbind_{method}_full_run_plan.json"
        candidates = sorted(
            (campaign_dir / "full_run_plans").glob(f"bayesbind_{method}_full_run_plan*.json")
        )
        ready_candidates: list[Path] = []
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                str(payload.get("method") or "").strip().lower() == method
                and str(payload.get("status") or "") == "ready"
            ):
                ready_candidates.append(candidate)
        plan_path = default_plan if default_plan.is_file() else None
        if ready_candidates:
            plan_path = max(ready_candidates, key=lambda path: path.stat().st_mtime)
    else:
        plan_path = plan_path.expanduser().resolve()

    plan: dict[str, Any] = {}
    if plan_path is not None and plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_status = str(plan.get("status") or "")
        add_gate("full_run_plan", "pass" if plan_status == "ready" else "fail", f"status={plan_status}")
    else:
        missing_plan = plan_path or (
            campaign_dir / "full_run_plans" / f"bayesbind_{method}_full_run_plan.json"
        )
        add_gate("full_run_plan", "wait", f"missing {missing_plan}")

    planned_full_run_ids: set[str] = set()
    if plan and isinstance(plan.get("targets"), list):
        for target in plan["targets"]:
            shards = target.get("shards") if isinstance(target, dict) else None
            if isinstance(shards, list):
                planned_full_run_ids.update(str(shard.get("run_id") or "") for shard in shards)
    submitted_full = [job for job in jobs if str(job.get("run_id") or job.get("job_id")) in planned_full_run_ids]
    completed_full = [job for job in submitted_full if str(job.get("status")) == "complete"]
    if planned_full_run_ids:
        if not submitted_full:
            add_gate("full_shards_submitted", "wait", "no planned full-run shard jobs registered")
        elif len(completed_full) != len(planned_full_run_ids):
            add_gate("full_shards_complete", "wait", f"{len(completed_full)}/{len(planned_full_run_ids)} complete")
        else:
            add_gate("full_shards_complete", "pass", f"{len(completed_full)} complete")

    registry = state.get("finalized_targets") if isinstance(state, dict) else None
    finalized_for_method = registry.get(method) if isinstance(registry, dict) else None
    finalized_count = len(finalized_for_method) if isinstance(finalized_for_method, dict) else 0
    if planned_full_run_ids:
        planned_targets = len(plan.get("targets") or [])
        add_gate(
            "target_finalization",
            "pass" if finalized_count == planned_targets and planned_targets else "wait",
            f"{finalized_count}/{planned_targets} finalized targets",
        )

    audit_path = campaign_dir / "final_results" / method / "bayesbind_campaign_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_status = str(audit.get("status") or "")
        add_gate("campaign_audit", "pass" if audit_status == "ready" else "wait", f"status={audit_status}")
    elif planned_full_run_ids:
        add_gate("campaign_audit", "wait", f"missing {audit_path}")

    blocking_failures = [gate for gate in gates if gate["status"] == "fail"]
    plan_exists = bool(plan_path is not None and plan_path.is_file())
    if blocking_failures:
        status = "not_ready"
        next_action = f"repair gate `{blocking_failures[0]['name']}`"
    elif plan_exists and planned_full_run_ids and submitted_full and len(completed_full) != len(planned_full_run_ids):
        status = "full_shards_running_or_incomplete"
        next_action = "continue supervisor polling/recovery"
    elif plan_exists and planned_full_run_ids and len(completed_full) == len(planned_full_run_ids) and finalized_count != len(plan.get("targets") or []):
        status = "ready_to_finalize_targets"
        next_action = "run plan-target finalization for completed targets"
    elif plan_exists and (not audit_path.is_file() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "ready"):
        status = "ready_to_run_campaign_audit"
        next_action = "run audit-bayesbind-campaign --strict"
    elif plan_exists:
        status = "ready"
        next_action = "publish defensible target-level summary"
    elif not complete_calibrations:
        status = "ready_to_submit_calibration"
        next_action = f"submit and supervise {method} calibration"
    else:
        status = "ready_to_freeze_full_plan"
        next_action = f"freeze {method} full-run plan from calibration output"

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else campaign_dir / "preflight" / f"bayesbind_{method}_execution_preflight.json"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "method": method,
        "campaign_dir": str(campaign_dir),
        "next_action": next_action,
        "issues": issues,
        "warnings": warnings,
        "gates": gates,
        "calibration_manifests": valid_calibration_manifests,
        "full_run_plan_path": str(plan_path) if plan_path is not None else "",
        "full_run_plan_exists": bool(plan_path is not None and plan_path.is_file()),
        "finalized_target_count": finalized_count,
        "disk_free_bytes": usage.free,
    }
    _write_json(output_path, payload)
    return {**payload, "preflight_path": str(output_path)}

def audit_bayesbind_campaign_results(
    campaign_dir: Path,
    plan_path: Path,
    *,
    method: str | None = None,
    output_path: Path | None = None,
    require_official_comparison: bool = False,
    expected_target_count: int | None = None,
) -> dict[str, Any]:
    """Audit finalized BayesBind target results against a frozen full-run plan."""
    campaign_dir = campaign_dir.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_method = str(plan.get("method") or "").strip().lower()
    if plan_method not in {"vina", "gnina"}:
        raise ValueError("Full-run plan method must be vina or gnina")
    if method is not None and method.strip().lower() != plan_method:
        raise ValueError("Audit method does not match full-run plan method")
    method = plan_method
    plan_sha = _sha256(plan_path)
    issues: list[str] = []
    warnings: list[str] = []
    if str(plan.get("status")) != "ready":
        issues.append(f"full-run plan is not ready: {plan.get('status')}")
    targets = plan.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Full-run plan targets must be a list")
    if expected_target_count is not None and len(targets) != expected_target_count:
        issues.append(
            f"target count {len(targets)} != expected {expected_target_count}"
        )

    state_path = campaign_dir / "campaign_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Campaign state missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    finalized_targets = state.get("finalized_targets") or {}
    method_registry = finalized_targets.get(method) if isinstance(finalized_targets, dict) else None
    if not isinstance(method_registry, dict):
        method_registry = {}
        issues.append(f"campaign_state missing finalized_targets for method {method}")
    jobs = state.get("jobs") or []
    if not isinstance(jobs, list):
        raise ValueError("campaign_state jobs must be a list")
    complete_run_ids = {
        str(job.get("run_id") or job.get("job_id"))
        for job in jobs
        if str(job.get("status")) == "complete"
    }

    def resolve_record_path(raw: Any, *, label: str, target_id: str, target_issues: list[str]) -> Path | None:
        value = str(raw or "").strip()
        if not value:
            target_issues.append(f"missing registry path `{label}`")
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = campaign_dir / path
        return path

    target_rows: list[dict[str, Any]] = []
    finalized_count = 0
    total_source_compounds = 0
    total_scored = 0
    total_failed = 0
    total_planned_shards = 0
    total_completed_shards = 0
    for target_plan in targets:
        target_id = str(target_plan.get("target_id") or "").strip()
        target_issues: list[str] = []
        target_warnings: list[str] = []
        planned_shards = target_plan.get("shards")
        if not target_id:
            target_issues.append("plan target missing target_id")
            planned_shards = []
        if not isinstance(planned_shards, list):
            target_issues.append("plan target shards must be a list")
            planned_shards = []
        total_planned_shards += len(planned_shards)
        expected_source_count = int(target_plan.get("source_compounds") or 0)
        source_csv = campaign_dir / "paper_inputs" / target_id / "source_compounds.csv"
        intended_rows: list[dict[str, str]] = []
        intended_by_id: dict[str, dict[str, str]] = {}
        if not source_csv.is_file():
            target_issues.append(f"source_compounds.csv missing: {source_csv}")
        else:
            with source_csv.open(newline="", encoding="utf-8") as handle:
                intended_rows = list(csv.DictReader(handle))
            intended_by_id = {row["source_compound_id"]: row for row in intended_rows}
            if len(intended_by_id) != len(intended_rows):
                target_issues.append("duplicate source_compound_id in intended CSV")
            if expected_source_count != len(intended_rows):
                target_issues.append(
                    f"plan source_compounds {expected_source_count} != intended CSV {len(intended_rows)}"
                )
        total_source_compounds += len(intended_rows)

        registry = method_registry.get(target_id) if isinstance(method_registry, dict) else None
        if not isinstance(registry, dict):
            target_issues.append("missing finalized target registry entry")
            registry = {}
        else:
            finalized_count += 1
        metrics_path = resolve_record_path(
            registry.get("metrics_path"), label="metrics_path", target_id=target_id, target_issues=target_issues
        )
        ranking_path = resolve_record_path(
            registry.get("ranking_path"), label="ranking_path", target_id=target_id, target_issues=target_issues
        )
        shard_manifest_path = resolve_record_path(
            registry.get("shard_manifest_csv"), label="shard_manifest_csv", target_id=target_id, target_issues=target_issues
        )

        metrics: dict[str, Any] = {}
        ranking_rows: list[dict[str, str]] = []
        shard_rows: list[dict[str, str]] = []
        if metrics_path is not None:
            if not metrics_path.is_file():
                target_issues.append(f"metrics file missing: {metrics_path}")
            else:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if ranking_path is not None:
            if not ranking_path.is_file():
                target_issues.append(f"ranking file missing: {ranking_path}")
            else:
                with ranking_path.open(newline="", encoding="utf-8") as handle:
                    ranking_rows = list(csv.DictReader(handle))
        if shard_manifest_path is not None:
            if not shard_manifest_path.is_file():
                target_issues.append(f"shard manifest missing: {shard_manifest_path}")
            else:
                with shard_manifest_path.open(newline="", encoding="utf-8") as handle:
                    shard_rows = list(csv.DictReader(handle))

        for key in ("ef1", "ef5", "ef20", "roc_auc", "bedroc", "coverage", "first_active_rank", "pose_retention"):
            if metrics.get(key) in (None, ""):
                target_issues.append(f"metrics missing `{key}`")
        if metrics:
            if str(metrics.get("target_id")) != target_id:
                target_issues.append("metrics target_id does not match plan")
            if str(metrics.get("method")) != method:
                target_issues.append("metrics method does not match plan")
            if str(metrics.get("plan_sha256")) != plan_sha:
                target_issues.append("metrics plan_sha256 does not match current plan")
            if str(registry.get("plan_sha256") or "") != plan_sha:
                target_issues.append("registry plan_sha256 does not match current plan")

        ranking_by_id = {str(row.get("source_compound_id") or "").strip(): row for row in ranking_rows}
        duplicate_ranking_ids = len(ranking_rows) - len(ranking_by_id)
        if "" in ranking_by_id:
            target_issues.append("ranking contains empty source_compound_id")
        if duplicate_ranking_ids:
            target_issues.append(f"ranking duplicate source_compound_id count {duplicate_ranking_ids}")
        missing_ids = sorted(set(intended_by_id) - set(ranking_by_id))
        extra_ids = sorted(set(ranking_by_id) - set(intended_by_id))
        if missing_ids:
            target_issues.append(f"ranking missing intended compounds: {len(missing_ids)}")
        if extra_ids:
            target_issues.append(f"ranking has extra source ids: {len(extra_ids)}")
        if len(ranking_rows) != len(intended_rows):
            target_issues.append(f"ranking rows {len(ranking_rows)} != intended rows {len(intended_rows)}")
        label_mismatches = 0
        for source_id, intended in intended_by_id.items():
            row = ranking_by_id.get(source_id)
            if row is not None and str(row.get("label")) != str(intended.get("label")):
                label_mismatches += 1
        if label_mismatches:
            target_issues.append(f"ranking label mismatches: {label_mismatches}")
        scored_count = sum(str(row.get("scored")) == "1" for row in ranking_rows)
        failed_count = sum(str(row.get("failed")) == "1" for row in ranking_rows)
        total_scored += scored_count
        total_failed += failed_count
        if scored_count + failed_count != len(ranking_rows):
            target_issues.append("ranking scored+failed count does not equal row count")
        scored_scores = [float(row["score"]) for row in ranking_rows if str(row.get("scored")) == "1"]
        failed_scores = [float(row["score"]) for row in ranking_rows if str(row.get("scored")) != "1"]
        if scored_scores and failed_scores and any(score <= max(scored_scores) for score in failed_scores):
            target_issues.append("failed rows are not tied worse than every scored row")
        counts = metrics.get("counts") if isinstance(metrics.get("counts"), dict) else {}
        if counts:
            expected_counts = {
                "before_prep": len(intended_rows),
                "after_aggregation": len(ranking_rows),
                "scored": scored_count,
                "failed": len(ranking_rows) - scored_count,
                "planned_shards": len(planned_shards),
                "completed_shards": len(shard_rows),
            }
            for key, expected_value in expected_counts.items():
                if int(counts.get(key) or 0) != expected_value:
                    target_issues.append(f"counts.{key} {counts.get(key)} != {expected_value}")
            if intended_rows:
                expected_coverage = round(scored_count / len(intended_rows), 6)
                if abs(float(metrics.get("coverage") or 0.0) - expected_coverage) > 0.000001:
                    target_issues.append("coverage does not equal scored/intended denominator")

        plan_run_ids = {str(shard.get("run_id") or "").strip() for shard in planned_shards}
        manifest_run_ids = {str(row.get("run_id") or "").strip() for row in shard_rows}
        if len(shard_rows) != len(planned_shards):
            target_issues.append(f"shard manifest rows {len(shard_rows)} != planned shards {len(planned_shards)}")
        missing_complete_jobs = sorted(run_id for run_id in plan_run_ids if run_id and run_id not in complete_run_ids)
        if missing_complete_jobs:
            target_issues.append(f"planned complete jobs missing from campaign_state: {len(missing_complete_jobs)}")
        missing_manifest_shards = sorted(plan_run_ids - manifest_run_ids)
        extra_manifest_shards = sorted(manifest_run_ids - plan_run_ids)
        if missing_manifest_shards:
            target_issues.append(f"shard manifest missing planned run ids: {len(missing_manifest_shards)}")
        if extra_manifest_shards:
            target_issues.append(f"shard manifest has extra run ids: {len(extra_manifest_shards)}")
        total_completed_shards += len(shard_rows)

        official_status = "not_required"
        if require_official_comparison:
            comparison_path = (metrics_path.parent if metrics_path is not None else campaign_dir) / f"official_{method}_comparison.json"
            if not comparison_path.is_file():
                target_issues.append(f"official comparison missing: {comparison_path}")
                official_status = "missing"
            else:
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                official_status = str(comparison.get("status") or "")
                if official_status != "ready":
                    target_issues.append(f"official comparison status is {official_status}")
                comparison_counts = comparison.get("counts") or {}
                for key in ("rerun_missing_intended", "rerun_extra_ids", "rerun_duplicate_ids", "official_missing_intended"):
                    if int(comparison_counts.get(key) or 0) != 0:
                        target_issues.append(f"official comparison {key}={comparison_counts.get(key)}")

        if target_warnings:
            warnings.extend(f"{target_id}: {warning}" for warning in target_warnings)
        if target_issues:
            issues.extend(f"{target_id}: {issue}" for issue in target_issues)
        target_rows.append(
            {
                "target_id": target_id,
                "method": method,
                "split": target_plan.get("split", ""),
                "status": "ready" if not target_issues else "not_ready",
                "source_compounds": len(intended_rows),
                "scored": scored_count,
                "failed": len(ranking_rows) - scored_count,
                "coverage": metrics.get("coverage", ""),
                "ef1": metrics.get("ef1", ""),
                "ef5": metrics.get("ef5", ""),
                "ef20": metrics.get("ef20", ""),
                "roc_auc": metrics.get("roc_auc", ""),
                "bedroc": metrics.get("bedroc", ""),
                "planned_shards": len(planned_shards),
                "completed_shards": len(shard_rows),
                "official_comparison_status": official_status,
                "metrics_path": str(metrics_path) if metrics_path is not None else "",
                "ranking_path": str(ranking_path) if ranking_path is not None else "",
                "shard_manifest_csv": str(shard_manifest_path) if shard_manifest_path is not None else "",
                "issues": "; ".join(target_issues),
            }
        )

    status = "ready" if not issues else "not_ready"
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else campaign_dir / "final_results" / method / "bayesbind_campaign_audit.json"
    )
    target_audit_path = output_path.with_suffix(".targets.csv")
    _write_csv(
        target_audit_path,
        target_rows,
        [
            "target_id",
            "method",
            "split",
            "status",
            "source_compounds",
            "scored",
            "failed",
            "coverage",
            "ef1",
            "ef5",
            "ef20",
            "roc_auc",
            "bedroc",
            "planned_shards",
            "completed_shards",
            "official_comparison_status",
            "metrics_path",
            "ranking_path",
            "shard_manifest_csv",
            "issues",
        ],
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "method": method,
        "campaign_dir": str(campaign_dir),
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "require_official_comparison": require_official_comparison,
        "expected_target_count": expected_target_count,
        "target_count": len(targets),
        "finalized_target_count": finalized_count,
        "ready_target_count": sum(row["status"] == "ready" for row in target_rows),
        "total_source_compounds": total_source_compounds,
        "total_scored": total_scored,
        "total_failed": total_failed,
        "total_planned_shards": total_planned_shards,
        "total_completed_shards": total_completed_shards,
        "issues": issues,
        "warnings": warnings,
        "targets": target_rows,
        "target_audit_csv": str(target_audit_path),
    }
    _write_json(output_path, payload)
    return {**payload, "audit_path": str(output_path)}



BASELINES_ARCHIVE_SHA256 = (
    "513f6e6a9a0e9141ed2aa28185c6a675c79a4c3c6a9a0d83d78e33a9a633e199"
)


def _baseline_class_from_dir(result_dir: str) -> str:
    if result_dir == "actives_results":
        return "paper_active_set"
    if result_dir == "random_results":
        return "paper_random_set"
    raise ValueError(f"Unsupported BayesBind result directory: {result_dir}")


def _baseline_label_from_class(label_source: str) -> int:
    if label_source == "paper_active_set":
        return 1
    if label_source == "paper_random_set":
        return 0
    raise ValueError(f"Unsupported BayesBind label source: {label_source}")


def _baseline_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _baseline_median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return round(sorted_values[middle], 6)
    return round((sorted_values[middle - 1] + sorted_values[middle]) / 2, 6)


def recover_official_bayesbind_baseline(
    campaign_dir: Path,
    *,
    method: str = "vina",
    archive_path: Path | None = None,
    inventory_path: Path | None = None,
    output_dir: Path | None = None,
    target_ids: list[str] | None = None,
    expected_archive_sha256: str | None = BASELINES_ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Recover official baseline score evidence without whole-archive extraction."""
    from ezscreen.benchmark.metrics import compute_metrics

    campaign_dir = campaign_dir.expanduser().resolve()
    method = method.strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("method must be vina or gnina")
    archive_path = (
        archive_path.expanduser().resolve()
        if archive_path is not None
        else campaign_dir / "inputs" / "BayesBindBaselinesV1.5.tar.gz"
    )
    inventory_path = (
        inventory_path.expanduser().resolve()
        if inventory_path is not None
        else campaign_dir / "inputs" / "archive_manifests" / "BayesBindBaselinesV1.5.members.jsonl"
    )
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else campaign_dir / "inputs" / "official_baselines" / method
    )
    if not archive_path.is_file():
        raise FileNotFoundError(f"Official baseline archive not found: {archive_path}")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Official baseline inventory not found: {inventory_path}")
    archive_sha = _sha256(archive_path)
    if expected_archive_sha256 is not None and archive_sha != expected_archive_sha256:
        raise ValueError(f"BayesBind baseline archive SHA256 mismatch: {archive_sha}")

    selected_targets = set(target_ids or [])
    if not selected_targets:
        selected_targets = {
            path.name for path in (campaign_dir / "paper_inputs").iterdir() if path.is_dir()
        }
    source_by_target_class: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    source_duplicates: list[dict[str, str]] = []
    intended_counts: dict[str, int] = {}
    source_rows_by_target: dict[str, list[dict[str, str]]] = {}
    for target_id in sorted(selected_targets):
        source_csv = campaign_dir / "paper_inputs" / target_id / "source_compounds.csv"
        if not source_csv.is_file():
            raise FileNotFoundError(f"Frozen source compounds not found: {source_csv}")
        with source_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        intended_counts[target_id] = len(rows)
        source_rows_by_target[target_id] = rows
        for row in rows:
            label_source = row["label_source"]
            key = (target_id, str(row["label"]))
            canonical = row["canonical_smiles"]
            bucket = source_by_target_class.setdefault(key, {})
            if canonical in bucket:
                source_duplicates.append(
                    {
                        "target_id": target_id,
                        "label_source": label_source,
                        "canonical_smiles": canonical,
                    }
                )
            else:
                bucket[canonical] = row

    result_members: dict[tuple[str, str], str] = {}
    pose_members: set[str] = set()
    member_records: list[dict[str, Any]] = []
    with inventory_path.open(encoding="utf-8") as handle:
        for line in handle:
            member = json.loads(line)
            if member.get("kind") != "file":
                continue
            name = str(member.get("name") or "")
            parts = name.split("/")
            if len(parts) < 6 or parts[0] != "BayesBindBaselinesV1.5":
                continue
            if parts[1] != method:
                continue
            split, target_id, result_dir, filename = parts[2], parts[3], parts[4], parts[-1]
            if target_id not in selected_targets:
                continue
            if result_dir not in {"actives_results", "random_results"}:
                continue
            if filename == "results.csv":
                key = (target_id, result_dir)
                if key in result_members:
                    raise ValueError(f"Duplicate official result member for {key}")
                result_members[key] = name
            elif filename.endswith(".sdf"):
                pose_members.add(name)
            member_records.append(
                {
                    "method": method,
                    "split": split,
                    "target_id": target_id,
                    "result_dir": result_dir,
                    "filename": filename,
                    "member_name": name,
                    "member_size": member.get("size", 0),
                    "member_mtime": member.get("mtime", ""),
                }
            )
    expected_result_keys = {
        (target_id, result_dir)
        for target_id in selected_targets
        for result_dir in ("actives_results", "random_results")
    }
    missing_result_keys = sorted(expected_result_keys - set(result_members))
    if missing_result_keys:
        raise ValueError(f"Missing official result members: {missing_result_keys[:10]}")

    rows_by_target: dict[str, list[dict[str, Any]]] = {target: [] for target in selected_targets}
    result_hash_rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:*") as tar:
        for (target_id, result_dir), member_name in sorted(result_members.items()):
            extracted = tar.extractfile(member_name)
            if extracted is None:
                raise ValueError(f"Could not read official result member: {member_name}")
            data = extracted.read()
            result_hash_rows.append(
                {
                    "target_id": target_id,
                    "result_dir": result_dir,
                    "member_name": member_name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            label_source = _baseline_class_from_dir(result_dir)
            label = _baseline_label_from_class(label_source)
            reader = csv.DictReader(data.decode("utf-8").splitlines())
            for ordinal, row in enumerate(reader):
                official_smiles = str(row.get("smiles") or "").strip()
                filename = str(row.get("filename") or "").strip()
                score_text = str(row.get("score") or "").strip()
                try:
                    canonical = _canonical_smiles(official_smiles)
                    canonical_ok = True
                except ValueError:
                    canonical = ""
                    canonical_ok = False
                source = source_by_target_class.get((target_id, str(label)), {}).get(canonical)
                official_score = float(score_text) if score_text else math.nan
                scored = bool(filename) and math.isfinite(official_score)
                pose_member = (
                    member_name.rsplit("/", 1)[0] + "/" + filename if filename else ""
                )
                pose_found = bool(pose_member) and pose_member in pose_members
                rows_by_target[target_id].append(
                    {
                        "target_id": target_id,
                        "label": label,
                        "label_source": label_source,
                        "official_result_dir": result_dir,
                        "official_row_index": ordinal,
                        "source_compound_id": source["source_compound_id"] if source else "",
                        "source_matched": int(source is not None),
                        "canonical_ok": int(canonical_ok),
                        "canonical_smiles": canonical,
                        "official_smiles": official_smiles,
                        "official_score": official_score if math.isfinite(official_score) else "",
                        "official_filename": filename,
                        "official_scored": int(scored),
                        "official_failed": int(not scored),
                        "pose_member": pose_member,
                        "pose_member_found": int(pose_found),
                    }
                )

    reference_rows: list[dict[str, Any]] = []
    source_ranking_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for target_id in sorted(rows_by_target):
        rows = rows_by_target[target_id]
        matched_source_ids = [
            row["source_compound_id"] for row in rows if row["source_compound_id"]
        ]
        duplicate_official_matches = len(matched_source_ids) - len(set(matched_source_ids))
        official_by_source: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["official_scored"]:
                row["metric_score_lower_is_better"] = -float(row["official_score"])
            else:
                row["metric_score_lower_is_better"] = ""
            reference_rows.append(row)
            source_id = str(row["source_compound_id"])
            if not source_id:
                continue
            current = official_by_source.get(source_id)
            if current is None:
                official_by_source[source_id] = row
            elif row["official_scored"] and (
                not current["official_scored"]
                or float(row["official_score"]) > float(current["official_score"])
            ):
                official_by_source[source_id] = row

        matched_scored_metric_scores = [
            -float(row["official_score"])
            for row in official_by_source.values()
            if row["official_scored"]
        ]
        worst_metric_score = (
            math.nextafter(max(matched_scored_metric_scores), math.inf)
            if matched_scored_metric_scores
            else 0.0
        )
        target_source_ranking: list[dict[str, Any]] = []
        for source in source_rows_by_target[target_id]:
            official = official_by_source.get(source["source_compound_id"])
            scored = bool(official and official["official_scored"])
            metric_score = (
                float(official["metric_score_lower_is_better"])
                if scored
                else worst_metric_score
            )
            pose_found = bool(official and official["pose_member_found"])
            ranking_row = {
                "target_id": target_id,
                "source_compound_id": source["source_compound_id"],
                "label": int(source["label"]),
                "label_source": source["label_source"],
                "canonical_smiles": source["canonical_smiles"],
                "official_score": official["official_score"] if official else "",
                "metric_score_lower_is_better": metric_score,
                "official_scored": int(scored),
                "official_failed": int(not scored),
                "pose_member_found": int(pose_found),
            }
            target_source_ranking.append(ranking_row)
            source_ranking_rows.append(ranking_row)
        metrics = compute_metrics(
            [float(row["metric_score_lower_is_better"]) for row in target_source_ranking],
            [int(row["label"]) for row in target_source_ranking],
            scored=[bool(row["official_scored"]) for row in target_source_ranking],
        )
        all_score_values = [
            float(row["official_score"]) for row in rows if row["official_scored"]
        ]
        intended_scored = sum(int(row["official_scored"]) for row in target_source_ranking)
        intended_pose_found = sum(
            1
            for row in target_source_ranking
            if row["official_scored"] and row["pose_member_found"]
        )
        official_scored = sum(int(row["official_scored"]) for row in rows)
        official_pose_found = sum(
            1 for row in rows if row["official_scored"] and row["pose_member_found"]
        )
        metrics_rows.append(
            {
                "target_id": target_id,
                "method": method,
                "official_score_provenance": "official archive score; higher is better; metric_score=-score",
                "intended_count": intended_counts[target_id],
                "metric_ranked_intended": len(target_source_ranking),
                "official_result_rows": len(rows),
                "source_matched": sum(int(row["source_matched"]) for row in rows),
                "source_unmatched": sum(1 for row in rows if not row["source_matched"]),
                "duplicate_source_matches": duplicate_official_matches,
                "canonical_failures": sum(1 for row in rows if not row["canonical_ok"]),
                "intended_scored": intended_scored,
                "intended_failed": len(target_source_ranking) - intended_scored,
                "official_scored": official_scored,
                "official_failed": len(rows) - official_scored,
                "pose_members_found": official_pose_found,
                "pose_members_missing": official_scored - official_pose_found,
                "intended_pose_retention": round(intended_pose_found / intended_scored, 6) if intended_scored else 0.0,
                "coverage": metrics.coverage,
                "ef1": metrics.ef1,
                "ef5": metrics.ef5,
                "ef20": metrics.ef20,
                "roc_auc": metrics.auc_roc,
                "bedroc20": metrics.bedroc,
                "first_active_rank": metrics.first_active_rank or "",
                "score_min": min(all_score_values) if all_score_values else "",
                "score_max": max(all_score_values) if all_score_values else "",
                "score_mean": _baseline_mean(all_score_values) or "",
                "score_median": _baseline_median(all_score_values) or "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "official_reference_rows.csv"
    metrics_path = output_dir / "official_target_metrics.csv"
    source_ranking_path = output_dir / "official_source_ranking.csv"
    member_manifest_path = output_dir / "official_member_manifest.csv"
    result_hash_path = output_dir / "official_result_hashes.csv"
    _write_csv(
        rows_path,
        reference_rows,
        [
            "target_id",
            "label",
            "label_source",
            "official_result_dir",
            "official_row_index",
            "source_compound_id",
            "source_matched",
            "canonical_ok",
            "canonical_smiles",
            "official_smiles",
            "official_score",
            "metric_score_lower_is_better",
            "official_filename",
            "official_scored",
            "official_failed",
            "pose_member",
            "pose_member_found",
        ],
    )
    _write_csv(
        metrics_path,
        metrics_rows,
        [
            "target_id",
            "method",
            "official_score_provenance",
            "intended_count",
            "metric_ranked_intended",
            "official_result_rows",
            "source_matched",
            "source_unmatched",
            "duplicate_source_matches",
            "canonical_failures",
            "intended_scored",
            "intended_failed",
            "official_scored",
            "official_failed",
            "pose_members_found",
            "pose_members_missing",
            "intended_pose_retention",
            "coverage",
            "ef1",
            "ef5",
            "ef20",
            "roc_auc",
            "bedroc20",
            "first_active_rank",
            "score_min",
            "score_max",
            "score_mean",
            "score_median",
        ],
    )
    _write_csv(
        source_ranking_path,
        source_ranking_rows,
        [
            "target_id",
            "source_compound_id",
            "label",
            "label_source",
            "canonical_smiles",
            "official_score",
            "metric_score_lower_is_better",
            "official_scored",
            "official_failed",
            "pose_member_found",
        ],
    )
    _write_csv(
        member_manifest_path,
        member_records,
        [
            "method",
            "split",
            "target_id",
            "result_dir",
            "filename",
            "member_name",
            "member_size",
            "member_mtime",
        ],
    )
    _write_csv(result_hash_path, result_hash_rows, ["target_id", "result_dir", "member_name", "bytes", "sha256"])
    totals = {
        "target_count": len(metrics_rows),
        "intended_count": sum(int(row["intended_count"]) for row in metrics_rows),
        "official_result_rows": sum(int(row["official_result_rows"]) for row in metrics_rows),
        "metric_ranked_intended": sum(int(row["metric_ranked_intended"]) for row in metrics_rows),
        "source_unmatched": sum(int(row["source_unmatched"]) for row in metrics_rows),
        "duplicate_source_matches": sum(int(row["duplicate_source_matches"]) for row in metrics_rows),
        "canonical_failures": sum(int(row["canonical_failures"]) for row in metrics_rows),
        "official_scored": sum(int(row["official_scored"]) for row in metrics_rows),
        "official_failed": sum(int(row["official_failed"]) for row in metrics_rows),
        "pose_members_missing": sum(int(row["pose_members_missing"]) for row in metrics_rows),
    }
    status = "ready" if all(
        totals[key] == 0
        for key in (
            "source_unmatched",
            "duplicate_source_matches",
            "canonical_failures",
            "pose_members_missing",
        )
    ) and totals["intended_count"] == totals["official_result_rows"] else "needs_review"
    manifest = {
        "schema_version": 1,
        "method": method,
        "status": status,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "inventory_path": str(inventory_path),
        "target_ids": sorted(selected_targets),
        "score_direction": "higher_is_better",
        "metric_transform": "metric_score_lower_is_better = -official_score; failed rows tied at worst metric score",
        "totals": totals,
        "outputs": {
            "official_reference_rows_csv": str(rows_path),
            "official_target_metrics_csv": str(metrics_path),
            "official_source_ranking_csv": str(source_ranking_path),
            "official_member_manifest_csv": str(member_manifest_path),
            "official_result_hashes_csv": str(result_hash_path),
        },
        "reference_only": True,
    }
    manifest_path = output_dir / "official_baseline_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}



def _summary_stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"n": 0, "min": "", "max": "", "mean": "", "median": ""}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _baseline_mean(values) or "",
        "median": _baseline_median(values) or "",
    }


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b):
        raise ValueError("correlation inputs must have the same length")
    if len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    den_a = math.sqrt(sum((a - mean_a) ** 2 for a in values_a))
    den_b = math.sqrt(sum((b - mean_b) ** 2 for b in values_b))
    if den_a == 0.0 or den_b == 0.0:
        return None
    return round(num / (den_a * den_b), 6)


def _spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2:
        return None
    return _pearson(_average_ranks(values_a), _average_ranks(values_b))


def compare_bayesbind_rerun_to_official(
    output_dir: Path,
    intended_csv: Path,
    official_dir: Path,
    *,
    method: str = "vina",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare finalized rerun outputs against official reference evidence."""
    from ezscreen.benchmark.metrics import compute_metrics

    output_dir = output_dir.expanduser().resolve()
    intended_csv = intended_csv.expanduser().resolve()
    official_dir = official_dir.expanduser().resolve()
    method = method.strip().lower()
    if method not in {"vina", "gnina"}:
        raise ValueError("method must be vina or gnina")
    rerun_metrics_path = output_dir / "target_metrics.json"
    rerun_ranking_path = output_dir / "source_ranking.csv"
    official_ranking_path = official_dir / "official_source_ranking.csv"
    official_metrics_path = official_dir / "official_target_metrics.csv"
    official_manifest_path = official_dir / "official_baseline_manifest.json"
    for path in (
        intended_csv,
        rerun_metrics_path,
        rerun_ranking_path,
        official_ranking_path,
        official_metrics_path,
        official_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison input missing: {path}")

    rerun_metrics = json.loads(rerun_metrics_path.read_text(encoding="utf-8"))
    official_manifest = json.loads(official_manifest_path.read_text(encoding="utf-8"))
    with intended_csv.open(newline="", encoding="utf-8") as handle:
        intended_rows = list(csv.DictReader(handle))
    if not intended_rows:
        raise ValueError("intended_csv has no rows")
    target_id = str(intended_rows[0]["target_id"])
    if any(str(row["target_id"]) != target_id for row in intended_rows):
        raise ValueError("intended_csv must contain exactly one target")
    intended_ids = {row["source_compound_id"] for row in intended_rows}

    with rerun_ranking_path.open(newline="", encoding="utf-8") as handle:
        rerun_rows = list(csv.DictReader(handle))
    with official_ranking_path.open(newline="", encoding="utf-8") as handle:
        official_all = list(csv.DictReader(handle))
    with official_metrics_path.open(newline="", encoding="utf-8") as handle:
        official_target_metrics = list(csv.DictReader(handle))
    official_full = [row for row in official_target_metrics if row["target_id"] == target_id]
    if len(official_full) != 1:
        raise ValueError(f"Expected one official target metric row for {target_id}")

    rerun_by_id = {row["source_compound_id"]: row for row in rerun_rows}
    official_by_id = {
        row["source_compound_id"]: row
        for row in official_all
        if row["target_id"] == target_id
    }
    duplicate_rerun_ids = len(rerun_rows) - len(rerun_by_id)
    missing_rerun_ids = sorted(intended_ids - set(rerun_by_id))
    extra_rerun_ids = sorted(set(rerun_by_id) - intended_ids)
    missing_official_ids = sorted(intended_ids - set(official_by_id))

    official_subset_rows: list[dict[str, Any]] = []
    for intended in intended_rows:
        source_id = intended["source_compound_id"]
        official = official_by_id.get(source_id)
        if official is None:
            official_subset_rows.append(
                {
                    "source_compound_id": source_id,
                    "target_id": target_id,
                    "label": int(intended["label"]),
                    "metric_score_lower_is_better": 0.0,
                    "official_scored": 0,
                    "official_failed": 1,
                    "pose_member_found": 0,
                }
            )
        else:
            official_subset_rows.append(official)
    official_scored_scores = [
        float(row["metric_score_lower_is_better"])
        for row in official_subset_rows
        if str(row.get("official_scored")) == "1"
    ]
    official_worst = (
        math.nextafter(max(official_scored_scores), math.inf)
        if official_scored_scores
        else 0.0
    )
    for row in official_subset_rows:
        if str(row.get("official_scored")) != "1":
            row["metric_score_lower_is_better"] = official_worst
    official_subset_metrics = compute_metrics(
        [float(row["metric_score_lower_is_better"]) for row in official_subset_rows],
        [int(row["label"]) for row in official_subset_rows],
        scored=[str(row.get("official_scored")) == "1" for row in official_subset_rows],
    )

    paired_rows: list[dict[str, Any]] = []
    rerun_pair_scores: list[float] = []
    official_pair_scores: list[float] = []
    for source_id in sorted(intended_ids):
        rerun = rerun_by_id.get(source_id)
        official = official_by_id.get(source_id)
        if not rerun or not official:
            continue
        rerun_scored = str(rerun.get("scored")) == "1"
        official_scored = str(official.get("official_scored")) == "1"
        score_delta = ""
        if rerun_scored and official_scored:
            rerun_score = float(rerun["score"])
            official_score = float(official["metric_score_lower_is_better"])
            score_delta = rerun_score - official_score
            rerun_pair_scores.append(rerun_score)
            official_pair_scores.append(official_score)
        paired_rows.append(
            {
                "source_compound_id": source_id,
                "target_id": target_id,
                "label": rerun.get("label", official.get("label", "")),
                "rerun_score": rerun.get("score", ""),
                "official_metric_score": official.get(
                    "metric_score_lower_is_better", ""
                ),
                "metric_score_delta": score_delta,
                "rerun_scored": int(rerun_scored),
                "official_scored": int(official_scored),
                "rerun_pose_retained": rerun.get("pose_retained", ""),
                "official_pose_member_found": official.get("pose_member_found", ""),
            }
        )

    metric_keys = [
        ("ef1", "ef1"),
        ("ef5", "ef5"),
        ("ef20", "ef20"),
        ("roc_auc", "auc_roc"),
        ("bedroc", "bedroc"),
        ("coverage", "coverage"),
    ]
    metric_comparison: dict[str, dict[str, float]] = {}
    for rerun_key, official_attr in metric_keys:
        rerun_value = float(rerun_metrics[rerun_key])
        official_value = float(getattr(official_subset_metrics, official_attr))
        metric_comparison[rerun_key] = {
            "rerun": rerun_value,
            "official_subset": official_value,
            "delta": round(rerun_value - official_value, 6),
        }
    rerun_scores = [float(row["score"]) for row in rerun_rows if row.get("score") not in ("", None)]
    official_subset_scores = [
        float(row["metric_score_lower_is_better"])
        for row in official_subset_rows
        if str(row.get("official_scored")) == "1"
    ]
    counts = {
        "intended": len(intended_rows),
        "rerun_ranking_rows": len(rerun_rows),
        "rerun_missing_intended": len(missing_rerun_ids),
        "rerun_extra_ids": len(extra_rerun_ids),
        "rerun_duplicate_ids": duplicate_rerun_ids,
        "official_missing_intended": len(missing_official_ids),
        "paired_scored": len(rerun_pair_scores),
        "rerun_scored": sum(str(row.get("scored")) == "1" for row in rerun_rows),
        "official_subset_scored": sum(
            str(row.get("official_scored")) == "1" for row in official_subset_rows
        ),
        "rerun_pose_retained": sum(
            str(row.get("pose_retained")) == "1" for row in rerun_rows
        ),
        "official_subset_pose_found": sum(
            str(row.get("pose_member_found")) == "1" for row in official_subset_rows
        ),
    }
    status = "ready" if all(
        counts[key] == 0
        for key in (
            "rerun_missing_intended",
            "rerun_extra_ids",
            "rerun_duplicate_ids",
            "official_missing_intended",
        )
    ) else "needs_review"
    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else output_dir / f"official_{method}_comparison.json"
    )
    paired_path = output_path.with_suffix(".paired_scores.csv")
    _write_csv(
        paired_path,
        paired_rows,
        [
            "source_compound_id",
            "target_id",
            "label",
            "rerun_score",
            "official_metric_score",
            "metric_score_delta",
            "rerun_scored",
            "official_scored",
            "rerun_pose_retained",
            "official_pose_member_found",
        ],
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "target_id": target_id,
        "method": method,
        "rerun_output_dir": str(output_dir),
        "intended_csv": str(intended_csv),
        "official_dir": str(official_dir),
        "official_manifest_status": official_manifest.get("status"),
        "reference_only": True,
        "counts": counts,
        "rerun_metrics_path": str(rerun_metrics_path),
        "official_full_target_metrics": official_full[0],
        "official_subset_metrics": {
            "ef1": official_subset_metrics.ef1,
            "ef5": official_subset_metrics.ef5,
            "ef20": official_subset_metrics.ef20,
            "roc_auc": official_subset_metrics.auc_roc,
            "bedroc": official_subset_metrics.bedroc,
            "coverage": official_subset_metrics.coverage,
            "first_active_rank": official_subset_metrics.first_active_rank,
        },
        "metric_comparison": metric_comparison,
        "score_summary": {
            "rerun_lower_is_better": _summary_stats(rerun_scores),
            "official_subset_lower_is_better": _summary_stats(official_subset_scores),
            "paired_lower_is_better": {
                "n": len(rerun_pair_scores),
                "pearson": _pearson(rerun_pair_scores, official_pair_scores),
                "spearman": _spearman(rerun_pair_scores, official_pair_scores),
            },
        },
        "identity_examples": {
            "missing_rerun_ids": missing_rerun_ids[:20],
            "extra_rerun_ids": extra_rerun_ids[:20],
            "missing_official_ids": missing_official_ids[:20],
        },
        "paired_scores_csv": str(paired_path),
    }
    _write_json(output_path, payload)
    return {**payload, "comparison_path": str(output_path)}
