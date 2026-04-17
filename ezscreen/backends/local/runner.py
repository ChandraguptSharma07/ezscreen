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


def _split_pdbqt_shard(shard_path: Path, out_dir: Path) -> list[tuple[str, Path]]:
    """Split a multi-molecule PDBQT shard into individual (lig_id, path) tuples.

    Meeko PDBQT terminates each ligand block with a TORSDOF line (e.g. 'TORSDOF 3').
    ligand_prep injects 'REMARK lig_id lig_00042' so we can correlate back to index.csv.
    """
    text    = shard_path.read_text(errors="replace")
    results = []
    current: list[str] = []
    current_lig_id: str | None = None
    fallback_idx = 0

    def _flush(lines: list[str], lig_id: str | None, fb_idx: int) -> None:
        if not any(ln.strip() for ln in lines):
            return
        block = "\n".join(lines) + "\n"
        name  = lig_id if lig_id else f"lig_{fb_idx:05d}"
        safe  = re.sub(r"[^\w\-]", "_", name)
        p     = out_dir / f"{safe}.pdbqt"
        p.write_text(block)
        results.append((name, p))

    for line in text.splitlines():
        current.append(line)
        if line.startswith("REMARK lig_id "):
            parts = line.split()
            if len(parts) >= 3:
                current_lig_id = parts[2]
        if line.startswith("TORSDOF"):
            _flush(current, current_lig_id, fallback_idx)
            current = []
            current_lig_id = None
            fallback_idx += 1

    _flush(current, current_lig_id, fallback_idx)
    return results


def _load_smiles_index(shard_dir: Path) -> dict[str, dict]:
    """Load lig_id → {name, smiles} from index.csv written by ligand_prep."""
    index_csv = shard_dir / "index.csv"
    if not index_csv.exists():
        return {}
    mapping: dict[str, dict] = {}
    try:
        with index_csv.open() as f:
            for row in csv.DictReader(f):
                lig_id = row.get("ligand", "")
                if lig_id:
                    mapping[lig_id] = {
                        "name":   row.get("name", ""),
                        "smiles": row.get("smiles", ""),
                    }
    except Exception:
        pass
    return mapping


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
    cpu_cores: int = 0,
    ligand_name: str = "",
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
        "--cpu", str(cpu_cores),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            return out_path.read_text()
        err = (result.stderr or result.stdout or "").strip()
        tag = f"[{ligand_name}] " if ligand_name else ""
        console.print(f"  [dim]{tag}Vina failed (code {result.returncode}): {err[:120]}[/dim]")
    except subprocess.TimeoutExpired:
        console.print(f"  [dim][{ligand_name}] Vina timed out (120s)[/dim]")
    except FileNotFoundError:
        console.print("  [red]Vina binary not found[/red]")
    return None


def run_local_screening(
    run_id: str,  # kept for interface parity with kaggle runner; used in console output
    receptor_pdbqt: Path,
    shard_paths: list[Path],
    box_center: list[float],
    box_size: list[float],
    work_dir: Path,
    exhaustiveness: int | None = None,
    num_modes: int = 3,
) -> dict[str, Any]:
    from ezscreen import config as _cfg
    _lc           = _cfg.load().get("local", {})
    if exhaustiveness is None:
        exhaustiveness = int(_lc.get("exhaustiveness", 4))
    _enable_floor = bool(_lc.get("enable_score_floor", True))
    _score_floor  = float(_lc.get("score_floor", -15.0))
    _cpu_cores    = int(_lc.get("cpu_cores", 0))

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        vina = get_vina_binary()
    except Exception as exc:
        return {"status": "failed", "output_dir": None, "error_type": f"vina_download_failed: {exc}"}

    # Load SMILES from index.csv written by ligand_prep (best-effort)
    shard_dir    = shard_paths[0].parent if shard_paths else work_dir
    smiles_index = _load_smiles_index(shard_dir)

    rows: list[dict]     = []
    poses_parts: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for shard in shard_paths:
            # Shards are already PDBQT — split into individual molecules
            ligands = _split_pdbqt_shard(shard, tmp)
            console.print(
                f"  [dim]{len(ligands)} ligand(s) parsed from {shard.name} — docking...[/dim]"
            )

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
                    safe_out   = re.sub(r"[^\w\-]", "_", name)
                    out_pdbqt  = tmp / f"{safe_out}_out.pdbqt"
                    pdbqt_text = _run_vina(
                        vina, receptor_pdbqt, lig_pdbqt, out_pdbqt,
                        box_center, box_size, exhaustiveness, num_modes,
                        cpu_cores=_cpu_cores, ligand_name=name,
                    )
                    progress.advance(task)

                    if pdbqt_text is None:
                        continue
                    score = _parse_vina_score(pdbqt_text)
                    if score is None or (_enable_floor and score < _score_floor):
                        continue

                    info = smiles_index.get(name, {})
                    row: dict = {
                        "ligand":        name,
                        "name":          info.get("name") or name,
                        "docking_score": score,
                    }
                    if info.get("smiles"):
                        row["smiles"] = info["smiles"]
                    rows.append(row)
                    poses_parts.append(pdbqt_text)

    rows.sort(key=lambda r: r["docking_score"])

    scores_csv = output_dir / "scores.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with scores_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    (output_dir / "poses.pdbqt").write_text("\n".join(poses_parts))

    console.print(f"  [green]Local docking done [{run_id}] — {len(rows)} scored poses[/green]")
    return {"status": "complete", "output_dir": output_dir, "error_type": None}
