from __future__ import annotations

import json
from pathlib import Path

import requests
from rich.console import Console

from ezscreen import auth
from ezscreen.errors import NetworkTimeoutError, NIMAuthError

console = Console()
_NIM_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"


def invoke(receptor_path: Path, hits_path: Path, output_dir: Path) -> None:
    """Run Stage 2 hit validation — DiffDock-L via NVIDIA NIM."""
    nim_key = auth.get_nim_key()
    if not nim_key:
        console.print("[red]✗ NIM API key required for validation.[/red]")
        console.print("  Run [bold]ezscreen auth[/bold] and add your NIM key.")
        console.print("  Free key at [link=https://build.nvidia.com]build.nvidia.com[/link]")
        return

    for p in (receptor_path, hits_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"  Submitting [bold]{hits_path.name}[/bold] → DiffDock-L (NIM)..."
    )

    try:
        r = requests.post(
            _NIM_URL,
            headers={"Authorization": f"Bearer {nim_key}"},
            json={
                "ligand":           hits_path.read_text(),
                "ligand_file_type": hits_path.suffix.lstrip("."),
                "protein":          receptor_path.read_text(),
                "num_poses":        1,
                "time_divisions":   20,
                "steps":            18,
            },
            timeout=300,
        )
    except requests.Timeout as exc:
        raise NetworkTimeoutError("NIM request timed out (>5 min)") from exc
    except requests.ConnectionError as exc:
        raise NetworkTimeoutError(f"Could not reach NIM API: {exc}") from exc

    if r.status_code == 401:
        raise NIMAuthError("NIM key rejected — get a new key at build.nvidia.com")

    if not r.ok:
        console.print(f"[red]NIM returned HTTP {r.status_code}: {r.text[:200]}[/red]")
        return

    result = r.json()
    out = output_dir / "validated_poses.json"
    out.write_text(json.dumps(result, indent=2))

    n_poses = len(result) if isinstance(result, list) else 1
    console.print(f"  [green]Validation complete[/green]  {n_poses} pose(s) → [bold]{out}[/bold]")
