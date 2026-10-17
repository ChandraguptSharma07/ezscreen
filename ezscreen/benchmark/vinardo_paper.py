from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MOL2_MARKER = b"@<TRIPOS>MOLECULE"


@dataclass(frozen=True)
class PaperTarget:
    position: int
    target_id: str
    pdb_id: str
    split: str
    actives: int
    decoys: int
    library_stem: str
    special_prep: str | None = None

    @property
    def intended_compounds(self) -> int:
        return self.actives + self.decoys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_targets(path: Path) -> list[PaperTarget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected non-empty target list: {path}")
    targets = [PaperTarget(**entry) for entry in payload]
    ids = [target.target_id for target in targets]
    if ids != sorted(ids):
        raise ValueError("Paper targets must be alphabetically sorted")
    if len(ids) != len(set(ids)):
        raise ValueError("Paper target IDs must be unique")
    expected_positions = list(range(1, len(targets) + 1))
    if [target.position for target in targets] != expected_positions:
        raise ValueError("Paper target positions must be contiguous and one-based")
    for target in targets:
        expected_split = "development" if target.position % 2 else "validation"
        if target.split != expected_split:
            raise ValueError(
                f"Target {target.target_id} has split {target.split}; "
                f"expected {expected_split}"
            )
        if target.actives <= 0 or target.decoys <= 0:
            raise ValueError(f"Target {target.target_id} has invalid class counts")
    return targets


def count_mol2_records(payload: bytes) -> int:
    return payload.count(MOL2_MARKER)


def _read_tar_member(archive: Path, member_name: str) -> bytes:
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        handle = tar.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"Unreadable archive member: {member_name}")
        return handle.read()


def read_nested_gzip_member(archive: Path, member_name: str) -> bytes:
    compressed = _read_tar_member(archive, member_name)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as handle:
        return handle.read()


def mol2_box(payload: bytes | str, padding: float = 15.0) -> dict[str, list[float]]:
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    coordinates: list[tuple[float, float, float]] = []
    in_atoms = False
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            coordinates.append((float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    if not coordinates:
        raise ValueError("MOL2 payload contains no readable atom coordinates")
    mins = [min(point[axis] for point in coordinates) for axis in range(3)]
    maxs = [max(point[axis] for point in coordinates) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]
    size = [(maxs[axis] - mins[axis]) + 2.0 * padding for axis in range(3)]
    return {
        "center": [round(value, 6) for value in center],
        "size": [round(value, 6) for value in size],
    }


def strip_pdb_residue(pdb_text: str, residue_name: str) -> tuple[str, int]:
    wanted = residue_name.strip().upper()
    output: list[str] = []
    removed = 0
    for line in pdb_text.splitlines(keepends=True):
        record = line[:6].strip().upper()
        residue = line[17:20].strip().upper() if len(line) >= 20 else ""
        if record in {"ATOM", "HETATM"} and residue == wanted:
            removed += 1
            continue
        output.append(line)
    return "".join(output), removed


def _member_names(archive: Path) -> set[str]:
    with tarfile.open(archive, "r:gz") as tar:
        return {member.name for member in tar.getmembers() if member.isfile()}


def verify_dud_archives(
    targets: Iterable[PaperTarget],
    *,
    ligand_archive: Path,
    decoy_archive: Path,
    receptor_archive: Path,
    crystal_ligand_archive: Path,
) -> dict[str, object]:
    target_list = list(targets)
    receptor_members = _member_names(receptor_archive)
    crystal_members = _member_names(crystal_ligand_archive)
    rows: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    for target in target_list:
        ligand_member = (
            f"dud_ligands2006/{target.library_stem}_ligands.mol2.gz"
        )
        decoy_member = f"dud_decoys2006/{target.library_stem}_decoys.mol2.gz"
        receptor_member = f"targets/{target.target_id}/rec.pdb"
        crystal_member = f"targets/{target.target_id}/xtal-lig.mol2"
        active_count = count_mol2_records(
            read_nested_gzip_member(ligand_archive, ligand_member)
        )
        decoy_count = count_mol2_records(
            read_nested_gzip_member(decoy_archive, decoy_member)
        )
        receptor_present = receptor_member in receptor_members
        crystal_present = crystal_member in crystal_members
        row = {
            "target_id": target.target_id,
            "expected_actives": target.actives,
            "actual_actives": active_count,
            "expected_decoys": target.decoys,
            "actual_decoys": decoy_count,
            "receptor_present": receptor_present,
            "crystal_ligand_present": crystal_present,
        }
        rows.append(row)
        if (
            active_count != target.actives
            or decoy_count != target.decoys
            or not receptor_present
            or not crystal_present
        ):
            mismatches.append(row)
    return {
        "target_count": len(target_list),
        "expected_actives": sum(target.actives for target in target_list),
        "actual_actives": sum(int(row["actual_actives"]) for row in rows),
        "expected_decoys": sum(target.decoys for target in target_list),
        "actual_decoys": sum(int(row["actual_decoys"]) for row in rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "targets": rows,
    }


def target_box_from_archive(
    target: PaperTarget,
    crystal_ligand_archive: Path,
    *,
    padding: float = 15.0,
) -> dict[str, list[float]]:
    member = f"targets/{target.target_id}/xtal-lig.mol2"
    return mol2_box(_read_tar_member(crystal_ligand_archive, member), padding=padding)
