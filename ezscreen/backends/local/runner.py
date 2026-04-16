from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ezscreen.backends.local.vina_binary import get_vina_binary

console = Console()
_SCORE_FLOOR = -15.0


def _sdf_to_pdbqt(sdf_path: Path, out_dir: Path) -> list[tuple[str, Path]]:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    prep     = MoleculePreparation()
    out      = []

    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"lig_{i:05d}"
        try:
            setup = prep.prepare(mol)
            pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(setup)
        except Exception:
            continue

        pdbqt_path = out_dir / f"{name}.pdbqt"
        pdbqt_path.write_text(pdbqt_str)
        out.append((name, pdbqt_path))

    return out


def _parse_vina_score(pdbqt_text: str) -> float | None:
    m = re.search(r"REMARK VINA RESULT:\s+([-\d.]+)", pdbqt_text)
    return float(m.group(1)) if m else None


def _run_vina(
    vina: Path,
    receptor: Path,
    ligand: Path,
    out_path: Path,
    center: list[float],
    size: list[float],
    exhaustiveness: int,
    num_modes: int,
) -> str | None:
    cmd = [
        str(vina),
        "--receptor", str(receptor),
        "--ligand",   str(ligand),
        "--out",      str(out_path),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x",   str(size[0]),
        "--size_y",   str(size[1]),
        "--size_z",   str(size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes",      str(num_modes),
        "--cpu", "1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            return out_path.read_text()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def run_local_screening(
    run_id: str,
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    box_center: list[float],
    box_size: list[float],
    work_dir: Path,
    exhaustiveness: int = 8,
    num_modes: int = 3,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        vina = get_vina_binary()
    except Exception as exc:
        return {"status": "failed", "output_dir": None, "error_type": f"vina_download_failed: {exc}"}

    rows: list[dict]  = []
    poses_sdf_parts:  list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for shard in shard_paths:
            ligands = _sdf_to_pdbqt(shard, tmp)
            console.print(f"  [dim]Docking {len(ligands)} ligands from {shard.name}...[/dim]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(shard.name, total=len(ligands))

                for name, lig_pdbqt in ligands:
                    out_pdbqt = tmp / f"{name}_out.pdbqt"
                    pdbqt_text = _run_vina(
                        vina, receptor_pdbqt, lig_pdbqt, out_pdbqt,
                        box_center, box_size, exhaustiveness, num_modes,
                    )
                    progress.advance(task)

                    if pdbqt_text is None:
                        continue
                    score = _parse_vina_score(pdbqt_text)
                    if score is None or score < _SCORE_FLOOR:
                        continue

                    rows.append({"ligand": name, "docking_score": score})

                    # pull SMILES from original sdf for enrichment later
                    from rdkit import Chem
                    sup = Chem.SDMolSupplier(str(shard), removeHs=False)
                    for mol in sup:
                        if mol and (mol.GetProp("_Name").strip() if mol.HasProp("_Name") else "") == name:
                            rows[-1]["smiles"] = Chem.MolToSmiles(mol)
                            break

                    poses_sdf_parts.append(pdbqt_text)  # store raw pdbqt; proper SDF needs obabel

    rows.sort(key=lambda r: r["docking_score"])

    scores_csv = output_dir / "scores.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with scores_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # poses are stored as pdbqt text in poses.pdbqt (SDF conversion requires obabel)
    poses_out = output_dir / "poses.pdbqt"
    poses_out.write_text("\n".join(poses_sdf_parts))

    console.print(f"  [green]Local docking done — {len(rows)} scored poses[/green]")
    return {"status": "complete", "output_dir": output_dir, "error_type": None}
