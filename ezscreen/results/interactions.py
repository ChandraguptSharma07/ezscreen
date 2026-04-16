from __future__ import annotations

from pathlib import Path


def compute_interactions(
    poses_sdf: Path,
    receptor_pdb: Path,
    top_n: int = 20,
) -> dict[str, dict[str, dict[str, int]]] | None:
    """Return per-compound, per-residue interaction counts using ProLIF.

    Returns {compound_name: {residue_label: {interaction_type: count}}}
    or None if ProLIF / MDAnalysis are not installed.
    """
    try:
        import MDAnalysis as mda
        import prolif
    except ImportError:
        return None

    if not poses_sdf.exists() or not receptor_pdb.exists():
        return None

    try:
        protein = mda.Universe(str(receptor_pdb))
        ligands = mda.Universe(str(poses_sdf))

        prot_sel = protein.select_atoms("protein")
        fp = prolif.Fingerprint()
        fp.run(ligands.trajectory[:top_n], ligands.atoms, prot_sel)
        df = fp.to_dataframe()
    except Exception:
        return None

    # df columns are MultiIndex: (residue_label, interaction_type)
    # rows are one per pose/frame
    result: dict[str, dict[str, dict[str, int]]] = {}
    for frame_idx, row in enumerate(df.itertuples(index=False)):
        name = f"pose_{frame_idx + 1}"
        contacts: dict[str, dict[str, int]] = {}
        for col, val in zip(df.columns, row):
            if not val:
                continue
            residue, itype = col[0], col[1]
            contacts.setdefault(residue, {})[itype] = contacts.get(residue, {}).get(itype, 0) + 1
        result[name] = contacts

    return result or None


def interactions_summary(
    data: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    """Collapse per-compound data to residue → interaction_type → count across all compounds."""
    summary: dict[str, dict[str, int]] = {}
    for compound_contacts in data.values():
        for residue, itypes in compound_contacts.items():
            for itype, count in itypes.items():
                summary.setdefault(residue, {})[itype] = (
                    summary.get(residue, {}).get(itype, 0) + count
                )
    return summary
