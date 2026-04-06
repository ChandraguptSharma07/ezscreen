from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ezscreen import __version__

console = Console()


def write_report(
    run_id: str,
    receptor_data: dict[str, Any],
    binding_site_data: dict[str, Any],
    ligand_data: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Build and write the prep report in .txt and .json formats.
    Returns {"json": Path, "txt": Path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_warnings = (
        receptor_data.get("warnings", [])
        + binding_site_data.get("warnings", [])
        + ligand_data.get("warnings", [])
    )

    report: dict[str, Any] = {
        "run_id":            run_id,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "ezscreen_version":  __version__,
        "receptor": {
            "source":               receptor_data.get("source", "unknown"),
            "pdb_id":               receptor_data.get("pdb_id"),
            "chain":                receptor_data.get("chains_selected", []),
            "chain_selection":      receptor_data.get("chain_selection_method", "user"),
            "residues":             receptor_data.get("residue_count", 0),
            "missing_residues":     receptor_data.get("missing_residues", 0),
            "alternates_resolved":  receptor_data.get("alternates_resolved", 0),
            "waters_removed":       receptor_data.get("waters_removed", 0),
            "is_alphafold":         receptor_data.get("is_alphafold", False),
            "alphafold_version":    receptor_data.get("af_version"),
            "tools":                receptor_data.get("tools", {}),
        },
        "binding_site": {
            "method":               binding_site_data.get("method", "unknown"),
            "reference_ligand":     binding_site_data.get("reference_ligand"),
            "reference_chain":      binding_site_data.get("reference_chain"),
            "box_center":           binding_site_data.get("center"),
            "box_size":             binding_site_data.get("size"),
            "box_volume_angstrom3": binding_site_data.get("volume_angstrom3"),
        },
        "ligands": {
            "input_source":           ligand_data.get("input_source"),
            "input_files":            ligand_data.get("input_files", 0),
            "total_input":            ligand_data.get("total_input", 0),
            "admet_removed":          ligand_data.get("admet_removed", 0),
            "admet_breakdown":        ligand_data.get("admet_breakdown", {}),
            "prep_passed":            ligand_data.get("prep_passed", 0),
            "prep_failed":            ligand_data.get("prep_failed", 0),
            "prep_failures":          ligand_data.get("prep_failures", {}),
            "failed_prep_file":       ligand_data.get("failed_prep_file"),
            "tautomers_enumerated":   ligand_data.get("tautomers_enumerated", False),
            "protonation_ph":         ligand_data.get("protonation_ph", 7.4),
            "tools":                  ligand_data.get("tools", {}),
        },
        "warnings": all_warnings,
    }

    json_path = output_dir / f"{run_id}_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    txt_path = output_dir / f"{run_id}_report.txt"
    txt_path.write_text(_render_txt(report))

    return {"json": json_path, "txt": txt_path, "report": report}


def print_summary(paths: dict[str, Path], report: dict[str, Any]) -> None:
    """Print a condensed prep report panel to the terminal."""
    rec = report.get("receptor", {})
    bs  = report.get("binding_site", {})
    lig = report.get("ligands", {})

    text = Text()
    text.append("Receptor     ", style="bold dim")
    text.append(
        f"{rec.get('pdb_id') or 'local'}  "
        f"chains {rec.get('chain')}  "
        f"{rec.get('residues')} residues\n"
    )
    text.append("Binding site ", style="bold dim")
    text.append(
        f"{bs.get('method')}  "
        f"center {bs.get('box_center')}  "
        f"volume {bs.get('box_volume_angstrom3')} Å³\n"
    )
    text.append("Ligands      ", style="bold dim")
    text.append(f"{lig.get('total_input', 0):,} input ? {lig.get('prep_passed', 0):,} prepared")
    if lig.get("admet_removed"):
        text.append(f"  ({lig['admet_removed']:,} ADMET removed)", style="dim")
    if lig.get("prep_failed"):
        text.append(f"  ({lig['prep_failed']:,} prep failures)", style="yellow")
    text.append("\n")

    for w in report.get("warnings", []):
        sev   = w.get("severity", "medium")
        style = "bold red" if sev == "high" else "yellow"
        text.append(f"?  {w.get('message', '')}\n", style=style)

    text.append(f"\nFull report ? {paths['txt']}", style="dim")
    console.print(Panel(text, title="[bold]Prep Report[/bold]"))


def _render_txt(r: dict[str, Any]) -> str:
    rec = r["receptor"]
    bs  = r["binding_site"]
    lig = r["ligands"]

    lines = [
        "ezscreen Prep Report",
        "====================",
        f"Run ID   : {r['run_id']}",
        f"Created  : {r['timestamp']}",
        f"Version  : {r['ezscreen_version']}",
        "",
        "RECEPTOR",
        "--------",
        f"Source            : {rec['source']}",
        f"PDB ID            : {rec.get('pdb_id') or 'local'}",
        f"Chains            : {rec['chain']}",
        f"Residues          : {rec['residues']}",
        f"Missing residues  : {rec['missing_residues']}",
        f"Alternates fixed  : {rec['alternates_resolved']}",
        f"Waters removed    : {rec['waters_removed']}",
        f"AlphaFold         : {rec['is_alphafold']} ({rec.get('alphafold_version') or 'N/A'})",
        f"Tools             : {rec['tools']}",
        "",
        "BINDING SITE",
        "------------",
        f"Method            : {bs['method']}",
        f"Reference ligand  : {bs.get('reference_ligand') or 'N/A'}",
        f"Box center (A)    : {bs['box_center']}",
        f"Box size   (A)    : {bs['box_size']}",
        f"Box volume (A3)   : {bs['box_volume_angstrom3']}",
        "",
        "LIGANDS",
        "-------",
        f"Input source      : {lig['input_source']}",
        f"Total input       : {lig['total_input']:,}",
        f"ADMET removed     : {lig['admet_removed']:,}",
        f"Prep passed       : {lig['prep_passed']:,}",
        f"Prep failed       : {lig['prep_failed']:,}",
        f"Tautomers         : {lig['tautomers_enumerated']}",
        f"Protonation pH    : {lig['protonation_ph']}",
        f"Tools             : {lig['tools']}",
        "",
        "WARNINGS",
        "--------",
    ]

    warnings = r.get("warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"[{w.get('severity','med').upper()}] {w.get('category','')}: {w.get('message','')}")
            if w.get("action"):
                lines.append(f"         Action: {w['action']}")
    else:
        lines.append("None.")

    return "\n".join(lines) + "\n"
