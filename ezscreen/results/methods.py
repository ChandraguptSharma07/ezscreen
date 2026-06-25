from __future__ import annotations

from pathlib import Path

_CITATIONS = {
    "unidock":  "UniDock (Yu et al., 2023)",
    "vina":     "AutoDock Vina (Eberhardt et al., 2021)",
    "rdkit":    "RDKit (https://www.rdkit.org)",
    "meeko":    "Meeko (Forli Lab)",
    "pdbfixer": "PDBFixer (Eastman et al., 2017)",
    "p2rank":   "P2Rank (Krivak & Hoksza, 2018)",
    "plip":     "PLIP (Salentin et al., 2015)",
}


def _fmt_center(v) -> str:
    if not v or len(v) < 3:
        return "n/a"
    return f"({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) A"


def _fmt_size(v) -> str:
    if not v or len(v) < 3:
        return "n/a"
    return f"{v[0]:.1f} x {v[1]:.1f} x {v[2]:.1f} A"


def build_methods_text(run_meta: dict) -> str:
    """Compose a publication-ready Methods paragraph from normalised run metadata.

    run_meta keys (all optional): version, receptor{pdb_id, is_alphafold, af_accession,
    af_version, chains, source}, binding_site{method, center, size, reference_ligand},
    ligands{total_input, admet_applied, admet_removed, protonation_ph, force_field},
    docking{engine, exhaustiveness, search_mode, num_poses, backend}.
    """
    rec  = run_meta.get("receptor", {})
    bs   = run_meta.get("binding_site", {})
    lig  = run_meta.get("ligands", {})
    dock = run_meta.get("docking", {})

    sentences: list[str] = []

    # Receptor
    chains = rec.get("chains") or []
    chain_str = ""
    if chains:
        chain_str = f", chain{'s' if len(chains) > 1 else ''} {', '.join(str(c) for c in chains)}"
    if rec.get("is_alphafold"):
        acc = rec.get("af_accession") or rec.get("pdb_id") or "the target"
        ver = rec.get("af_version")
        receptor_src = f"The AlphaFold model for {acc}" + (f" (v{ver})" if ver else "") + chain_str
    elif rec.get("pdb_id"):
        receptor_src = f"The crystal structure {rec['pdb_id']}{chain_str}"
    else:
        receptor_src = f"The provided receptor structure{chain_str}"
    sentences.append(
        f"{receptor_src} was prepared with PDBFixer and Meeko "
        "(hydrogen addition, protonation, and AutoDock atom typing)."
    )

    # Binding site
    method = (bs.get("method") or "").lower()
    extra_cite: str | None = None
    if "cocrystal" in method or "co-crystal" in method:
        ref = bs.get("reference_ligand")
        site_desc = "centred on the co-crystallised ligand" + (f" {ref}" if ref else "")
    elif "p2rank" in method:
        site_desc = "predicted with P2Rank"
        extra_cite = "p2rank"
    elif "residue" in method:
        site_desc = "defined from the selected binding-site residues"
    elif "blind" in method or "whole" in method:
        site_desc = "set to the whole-protein bounding box (blind docking)"
    else:
        site_desc = "defined"
    sentences.append(
        f"The docking box was {site_desc}, centred at {_fmt_center(bs.get('center'))} "
        f"with dimensions {_fmt_size(bs.get('size'))}."
    )

    # Ligand prep
    n  = lig.get("total_input")
    ff = lig.get("force_field", "MMFF94")
    ph = lig.get("protonation_ph", 7.4)
    n_str = f"{n:,} input compounds were" if isinstance(n, int) and n else "Input compounds were"
    sentences.append(
        f"{n_str} protonated at pH {ph}, embedded in 3D with RDKit ETKDG and "
        f"energy-minimised with the {ff} force field, then converted to PDBQT with Meeko."
    )
    if lig.get("admet_removed"):
        sentences.append(
            f"ADMET pre-filtering removed {lig['admet_removed']:,} compounds before docking."
        )
    elif lig.get("admet_applied"):
        sentences.append(
            "ADMET pre-filtering (Lipinski Ro5, PAINS, Brenk, and Veber rules) "
            "was applied before docking."
        )

    # Docking
    engine = dock.get("engine", "UniDock")
    engine_cite = "vina" if "vina" in engine.lower() else "unidock"
    depth_bits: list[str] = []
    if dock.get("exhaustiveness"):
        depth_bits.append(f"exhaustiveness {dock['exhaustiveness']}")
    if dock.get("search_mode"):
        depth_bits.append(f"{dock['search_mode']} search mode")
    if dock.get("num_poses"):
        depth_bits.append(f"up to {dock['num_poses']} poses per ligand")
    depth = (" at " + ", ".join(depth_bits)) if depth_bits else ""
    backend = dock.get("backend")
    backend_str = f" on {backend}" if backend else ""
    sentences.append(f"Molecular docking was performed with {engine}{depth}{backend_str}.")

    # Citations
    cite_keys = [engine_cite, "rdkit", "meeko", "pdbfixer"]
    if extra_cite:
        cite_keys.append(extra_cite)
    cite_keys.append("plip")
    seen: set[str] = set()
    cites: list[str] = []
    for key in cite_keys:
        if key not in seen and key in _CITATIONS:
            cites.append(_CITATIONS[key])
            seen.add(key)
    citation_line = "Software and citations: " + "; ".join(cites) + "."

    version = run_meta.get("version")
    footer = f"Generated with ezscreen v{version}." if version else "Generated with ezscreen."

    return " ".join(sentences) + "\n\n" + citation_line + "\n\n" + footer + "\n"


def write_methods(run_meta: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "methods.txt"
    out.write_text(build_methods_text(run_meta), encoding="utf-8")
    return out


def _reference_ligand(ctx: dict) -> str | None:
    site = ctx.get("site_details") or {}
    if site.get("type") in ("cocrystal", "co-crystal"):
        ligs = site.get("ligands")
        if ligs:
            first = ligs[0]
            return first.get("resn") if isinstance(first, dict) else str(first)
    return None


def run_meta_from_checkpoint(run_id: str) -> dict | None:
    """Rebuild a normalised run_meta from the persisted run config + global config."""
    import json

    from ezscreen import __version__, checkpoint
    from ezscreen import config as _cfg

    row = checkpoint.get_run(run_id)
    if not row:
        return None
    try:
        ctx = json.loads(row.get("config_json") or "{}")
    except Exception:
        ctx = {}

    try:
        cfg = _cfg.load()
    except Exception:
        cfg = {}
    ph = cfg.get("run", {}).get("default_ph", 7.4)
    ff = cfg.get("prep", {}).get("force_field", "MMFF94")

    run_locally = bool(ctx.get("run_locally"))
    box = ctx.get("box") or {}

    return {
        "version": __version__,
        "receptor": {
            "pdb_id":       ctx.get("pdb_id"),
            "is_alphafold": ctx.get("is_alphafold", False),
            "af_accession": ctx.get("pdb_id") if ctx.get("is_alphafold") else None,
            "af_version":   ctx.get("af_version"),
            "chains":       ctx.get("selected_chains") or ctx.get("chains") or [],
            "source":       ctx.get("pdb_source"),
        },
        "binding_site": {
            "method":           ctx.get("site_method") or (ctx.get("site_details") or {}).get("type"),
            "center":           box.get("center"),
            "size":             box.get("size"),
            "reference_ligand": _reference_ligand(ctx),
        },
        "ligands": {
            "total_input":    row.get("total_compounds"),
            "admet_applied":  bool(ctx.get("admet_pre_filter")),
            "protonation_ph": ph,
            "force_field":    ff,
        },
        "docking": {
            "engine":         "AutoDock Vina" if run_locally else "UniDock",
            "exhaustiveness": ctx.get("exhaustiveness"),
            "search_mode":    (ctx.get("search_params") or {}).get("search_mode"),
            "backend":        "local CPU" if run_locally else "Kaggle GPU",
        },
    }
