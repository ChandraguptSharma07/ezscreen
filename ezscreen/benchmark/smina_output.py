from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MODE_PATTERN = re.compile(
    r"(?m)^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)
MODEL_PATTERN = re.compile(r"(?m)^MODEL\s+\d+\s*$")
STEM_PATTERN = re.compile(
    r"^(?P<sample_index>\d+)_(?P<scoring>dkoes_scoring|vina|vinardo)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_smina_attempt(log_path: Path, pose_path: Path) -> dict[str, Any]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    pose_text = pose_path.read_text(encoding="utf-8", errors="replace")
    mode_rows = MODE_PATTERN.findall(log_text)
    scores = [float(row[1]) for row in mode_rows]
    pose_count = len(MODEL_PATTERN.findall(pose_text))
    return {
        "score": min(scores) if scores else None,
        "log_mode_count": len(scores),
        "pose_count": pose_count,
        "status": "scored" if scores and pose_count else "failed",
        "log_sha256": _sha256(log_path),
        "pose_sha256": _sha256(pose_path),
    }


def recover_smina_directory(
    output_dir: Path,
    *,
    expected_attempts: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    logs_dir = output_dir / "logs"
    poses_dir = output_dir / "poses"
    if not logs_dir.is_dir() or not poses_dir.is_dir():
        raise FileNotFoundError("Expected output_dir/logs and output_dir/poses")

    rows: list[dict[str, Any]] = []
    for pose_path in sorted(poses_dir.glob("*.pdbqt")):
        match = STEM_PATTERN.fullmatch(pose_path.stem)
        if not match:
            continue
        log_path = logs_dir / f"dock_{pose_path.stem}.log"
        parsed = {
            "sample_index": int(match.group("sample_index")),
            "scoring": match.group("scoring"),
            "pose_file": str(pose_path),
            "log_file": str(log_path),
        }
        if log_path.is_file():
            parsed.update(parse_smina_attempt(log_path, pose_path))
        else:
            parsed.update(
                {
                    "score": None,
                    "log_mode_count": 0,
                    "pose_count": 0,
                    "status": "failed",
                    "log_sha256": "",
                    "pose_sha256": _sha256(pose_path),
                }
            )
        rows.append(parsed)

    expected = expected_attempts if expected_attempts is not None else len(rows)
    scored = sum(row["status"] == "scored" for row in rows)
    audit = {
        "schema_version": 1,
        "parser": "smina mode-table affinity + PDBQT MODEL count",
        "output_dir": str(output_dir),
        "expected_attempts": expected,
        "observed_attempts": len(rows),
        "scored": scored,
        "failed": len(rows) - scored,
        "coverage": scored / expected if expected else 0.0,
        "pose_retention": (
            sum(int(row["pose_count"]) > 0 for row in rows) / expected
            if expected
            else 0.0
        ),
        "gate_pass": len(rows) == expected and scored == expected,
    }

    csv_path = output_dir / "scores_recovered.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    audit_path = output_dir / "smina_output_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {**audit, "scores_csv": str(csv_path), "audit_json": str(audit_path)}
