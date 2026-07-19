# PPARD UniDock Docking Audit - Interim Results

PPARD ligand protonation/enumeration variants redock reliably but still do not rescue raw UniDock early enrichment.

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
| ligprep_local | 1.181 | 0.000 | 0.000 | 0.3295 | 15/15 | 3 | 0 |
| ligprep_uncharge | 1.135 | 0.000 | 0.000 | 0.3067 | 15/15 | 2 | 0 |
| ligprep_enum2 | 1.053 | 0.000 | 0.000 | 0.2781 | 15/15 | 3 | 0 |

Interpretation:
- All prep/search/box variants tested so far redock the 2ZNP cocrystal ligand under 2 A.
- Kaggle Gypsum protonation/tautomer enumeration generated 95 scored variants from 50 parent ligands; metrics collapsed to best score per parent and still gave EF1/EF5 0.0, AUC 0.2781, and 3 actives in top 20.
- Local ETKDG seed42 + MMFF94 200-iteration PDBQT panel produced RMSD 1.181 A, AUC 0.3295, EF1/EF5 0.0, and 3 actives in top 20.
- Local uncharged ETKDG/MMFF PDBQT panel produced RMSD 1.135 A, AUC 0.3067, EF1/EF5 0.0, and 2 actives in top 20; neutralization did not improve ranking.
- Current evidence says pose generation for the cocrystal is credible, but raw UniDock ranking remains dominated by hard decoys on the audit panel.
