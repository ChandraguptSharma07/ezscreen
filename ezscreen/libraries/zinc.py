from __future__ import annotations

from pathlib import Path

import requests

# ZINC15 REST API — still operational, simpler than ZINC22 tranche system
_ZINC15_API = "https://zinc15.docking.org/substances.txt"
_PAGE_SIZE = 1000

SIZE_OPTIONS: dict[str, int] = {
    "1k":   1_000,
    "10k":  10_000,
    "100k": 100_000,
}

# MW and logP bounds for each preset; HBD/HBA follow Lipinski / Veber rules
PRESETS: dict[str, dict[str, float | int]] = {
    "drug-like": {
        "mwt-gt": 250,
        "mwt-lt": 500,
        "logp-gt": -1,
        "logp-lt": 5,
    },
    "lead-like": {
        "mwt-gt": 150,
        "mwt-lt": 350,
        "logp-gt": -3,
        "logp-lt": 3,
    },
    "fragment-like": {
        "mwt-gt": 100,
        "mwt-lt": 250,
        "logp-gt": -3,
        "logp-lt": 3,
    },
}


def _parse_lines(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("smiles"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
        elif parts:
            out.append((parts[0], ""))
    return out


def _fetch_page(
    session: requests.Session,
    base_params: dict,
    page: int,
) -> list[tuple[str, str]]:
    params = {**base_params, "count": _PAGE_SIZE, "page": page}
    try:
        resp = session.get(_ZINC15_API, params=params, timeout=30)
        resp.raise_for_status()
        return _parse_lines(resp.text)
    except Exception:
        return []


def download_zinc_library(
    output_path: Path,
    size: str = "10k",
    count: int | None = None,
    preset: str = "drug-like",
    purchasable: bool = True,
    mw_min: float | None = None,
    mw_max: float | None = None,
    logp_min: float | None = None,
    logp_max: float | None = None,
) -> int:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    # count= takes priority; size= is the named-preset fallback
    n_wanted = count if count is not None else SIZE_OPTIONS.get(size, 10_000)
    filters = dict(PRESETS.get(preset, PRESETS["drug-like"]))

    if mw_min is not None:
        filters["mwt-gt"] = mw_min
    if mw_max is not None:
        filters["mwt-lt"] = mw_max
    if logp_min is not None:
        filters["logp-gt"] = logp_min
    if logp_max is not None:
        filters["logp-lt"] = logp_max

    if purchasable:
        filters["purchasable"] = "1"

    console = Console()
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []

    with requests.Session() as session:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            label = str(count) if count is not None else size
            task = progress.add_task(
                f"Downloading ZINC {label} {preset}...", total=n_wanted
            )
            page = 1
            while len(rows) < n_wanted:
                batch = _fetch_page(session, filters, page)
                if not batch:
                    break
                for smi, zid in batch:
                    if smi not in seen and len(rows) < n_wanted:
                        seen.add(smi)
                        rows.append((smi, zid))
                progress.update(task, completed=len(rows))
                page += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{smi} {zid}".strip() for smi, zid in rows]
    output_path.write_text("\n".join(lines) + "\n")
    console.print(f"  Wrote {len(rows)} compounds to {output_path}")
    return len(rows)
