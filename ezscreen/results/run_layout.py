from __future__ import annotations

import json
from pathlib import Path


def resolve_output_dir(run_dir: Path) -> Path:
    """Return the directory holding final result artifacts for a run."""
    root_scores = run_dir / "scores.csv"
    nested_output = run_dir / "output"
    nested_scores = nested_output / "scores.csv"
    if root_scores.exists():
        return run_dir
    if nested_scores.exists():
        return nested_output
    return nested_output


def resolve_run_dir(path: Path) -> Path:
    """Normalize either a run root or a legacy output dir back to the run root."""
    return path.parent if path.name == "output" else path


def resolve_receptor_pdb(run_dir: Path) -> Path | None:
    """Locate the prepared receptor PDB across legacy and team-runner layouts."""
    resume_json = run_dir / "resume.json"
    if resume_json.exists():
        try:
            info = json.loads(resume_json.read_text())
            receptor_pdb = info.get("receptor_pdb")
            if receptor_pdb:
                candidate = Path(receptor_pdb)
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    output_dir = resolve_output_dir(run_dir)
    for candidate in (
        run_dir / "receptor" / "receptor_prep.pdb",
        output_dir / "receptor_prep.pdb",
    ):
        if candidate.exists():
            return candidate
    return None
