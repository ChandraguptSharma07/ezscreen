from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import requests

from ezscreen import auth
from ezscreen.errors import NetworkTimeoutError, NIMAuthError

_NIM_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"


def run_diffdock_l(
    receptor_path: str,
    ligand_path: str,
    output_dir: str,
    nim_key: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Submit a DiffDock-L job to NVIDIA NIM and write results to output_dir.

    Returns a dict with at least ``poses_written`` (int).
    Raises NIMAuthError, NetworkTimeoutError, or RuntimeError on failure.
    """
    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if nim_key is None:
        creds = auth.load_credentials()
        nim_key = auth.get_nim_key(creds)
    if not nim_key:
        raise NIMAuthError(
            "NIM API key required.  Add it via Auth Setup or set NVIDIA_NIM_API_KEY."
        )

    receptor = Path(receptor_path)
    ligand   = Path(ligand_path)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[#6e7681]Submitting {ligand.name} → DiffDock-L (NIM)...[/#6e7681]")

    try:
        r = requests.post(
            _NIM_URL,
            headers={"Authorization": f"Bearer {nim_key}"},
            json={
                "ligand":           ligand.read_text(),
                "ligand_file_type": ligand.suffix.lstrip("."),
                "protein":          receptor.read_text(),
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
        raise RuntimeError(f"NIM returned HTTP {r.status_code}: {r.text[:200]}")

    result = r.json()
    out    = out_dir / "validated_poses.json"
    out.write_text(json.dumps(result, indent=2))

    n_poses = len(result) if isinstance(result, list) else 1
    _log(f"[#3fb950]Done — {n_poses} pose(s) written to {out}[/#3fb950]")

    return {"poses_written": n_poses, "output_path": str(out)}
