# PPARD UniDock Docking Audit - Interim Results

PPARD receptor-prep and ligand-uncharge variants redock reliably but do not rescue raw UniDock early enrichment.

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
| recfixed_obabel | 1.617 | 0.000 | 0.000 | 0.2857 | 15/15 | 3 | 0 |
| recraw_obabel | 1.157 | 0.000 | 0.000 | 0.2895 | 15/15 | 2 | 0 |

Interpretation:
- All receptor PDBQT variants tested so far redock the 2ZNP cocrystal ligand under 2 A.
- Open Babel receptor variants from receptor_prep.pdb, receptor_fixed_clean.pdb, and raw receptor.pdb produce weak 50-ligand panel AUC values from 0.2857 to 0.3238.
- Ligand uncharge improved redocking RMSD to 1.141 A but still left EF1/EF5 at 0.0 and only 1 active in the top 20.
- Remaining high-yield branch is ligand conformer/prep handling, especially pregenerated conformers versus Kaggle-side SMILES prep.
