from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AD4ZN_PARAMETER_FILE = "AD4Zn.dat"
AD4ZN_GRID_SPACING_ANGSTROM = 0.375


@dataclass(frozen=True)
class Ad4ZnMapSpec:
    center: tuple[float, float, float]
    requested_size: tuple[float, float, float]
    npts: tuple[int, int, int]
    spacing_angstrom: float = AD4ZN_GRID_SPACING_ANGSTROM

    @property
    def covered_size(self) -> tuple[float, float, float]:
        return tuple(value * self.spacing_angstrom for value in self.npts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "requested_size": list(self.requested_size),
            "npts": list(self.npts),
            "spacing_angstrom": self.spacing_angstrom,
            "covered_size": list(self.covered_size),
            "parameter_file": AD4ZN_PARAMETER_FILE,
            "requires_tz_pseudoatoms": True,
            "scoring": "ad4",
        }


def build_ad4zn_map_spec(
    center: list[float],
    size: list[float],
    *,
    spacing_angstrom: float = AD4ZN_GRID_SPACING_ANGSTROM,
) -> Ad4ZnMapSpec:
    if len(center) != 3 or len(size) != 3:
        raise ValueError("AD4Zn map center and size must contain three values")
    if not math.isfinite(spacing_angstrom) or spacing_angstrom <= 0:
        raise ValueError("AD4Zn grid spacing must be finite and positive")
    values = tuple(float(value) for value in [*center, *size])
    if not all(math.isfinite(value) for value in values) or any(
        value <= 0 for value in size
    ):
        raise ValueError("AD4Zn map center must be finite and size must be positive")
    npts = tuple(
        max(2, int(math.ceil(float(length) / spacing_angstrom / 2.0) * 2))
        for length in size
    )
    return Ad4ZnMapSpec(
        center=tuple(float(value) for value in center),
        requested_size=tuple(float(value) for value in size),
        npts=npts,
        spacing_angstrom=spacing_angstrom,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ad4zn_map_manifest(manifest: dict[str, Any], spec: Ad4ZnMapSpec) -> None:
    if manifest.get("scoring") != "ad4":
        raise ValueError("AD4Zn map manifest must declare ad4 scoring")
    if manifest.get("parameter_file") != AD4ZN_PARAMETER_FILE:
        raise ValueError("AD4Zn map manifest must use AD4Zn.dat")
    if manifest.get("tz_pseudoatoms") is not True:
        raise ValueError("AD4Zn map manifest must confirm TZ pseudoatoms")
    expected = spec.as_dict()
    for key in ("center", "npts", "parameter_file", "scoring"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"AD4Zn map manifest mismatch: {key}")
    covered = manifest.get("covered_size")
    if not isinstance(covered, list) or len(covered) != 3:
        raise ValueError("AD4Zn map manifest missing covered_size")
    if any(
        float(actual) + 1e-9 < required
        for actual, required in zip(covered, spec.requested_size)
    ):
        raise ValueError("AD4Zn maps do not cover requested docking box")
    files = manifest.get("map_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("AD4Zn map manifest missing map file hashes")
    if any(
        not isinstance(digest, str) or len(digest) != 64 for digest in files.values()
    ):
        raise ValueError("AD4Zn map manifest has invalid map hash")
