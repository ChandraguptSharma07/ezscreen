# PPARD UniDock Docking Audit - Interim Results

PPARD redocking remains reliable under ligand uncharge and receptor-prep variants, but raw UniDock early enrichment is still weak.

| Ablation | RMSD A | EF1 | EF5 | AUC | Active survival | Top20 actives | Clash-like top20 |
|---|---:|---:|---:|---:|---|---:|---:|
| current_box_balance | 1.521 | 0.000 | 0.000 | 0.3314 | 15/15 | 3 | 0 |
| search_fast_current_box | 1.535 | 0.000 | 0.000 | 0.2686 | 15/15 | 2 | 0 |
| search_detail_current_box | 1.615 | 0.000 | 0.000 | 0.2762 | 15/15 | 1 | 0 |
| box_center_crystal_centroid | 1.581 | 0.000 | 0.000 | 0.3219 | 15/15 | 1 | 0 |
| box_padding_4a_crystal_centroid | 1.780 | 0.000 | 0.000 | 0.2990 | 15/15 | 4 | 0 |
| box_padding_8a_crystal_centroid | 1.753 | 0.000 | 0.000 | 0.2724 | 15/15 | 1 | 0 |
| seed_repeat_current_box_001 | 1.298 | 0.000 | 0.000 | 0.3181 | 15/15 | 3 | 0 |
| ligand_uncharge_current_receptor | 1.141 | 0.000 | 0.000 | 0.3162 | 15/15 | 1 | 0 |
| recprep_obabel | 1.362 | 0.000 | 0.000 | 0.3238 | 15/15 | 2 | 0 |

Interpretation:
- Ligand uncharge changed 18/50 audit-panel ligands and the cocrystal ligand formal charge from -1 to 0; redocking improved to 1.141 A but panel AUC stayed weak at 0.3162.
- Open Babel receptor_prep.pdb PDBQT redocked at 1.362 A and produced panel AUC 0.3238, so this receptor-prep/protonation variant does not rescue top ranking.
- The next prep audit should compare remaining receptor PDBQT variants and ligand conformer/prep settings before any full PPARD rerun.
