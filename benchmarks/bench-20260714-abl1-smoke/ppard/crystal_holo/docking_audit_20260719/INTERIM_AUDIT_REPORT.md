# PPARD UniDock Docking Audit - Interim Results

All tested settings redock the cocrystal under 2 A, but none fix raw UniDock early enrichment on the hard 50-ligand PPARD panel.

| Ablation | RMSD A | EF1 | EF5 | AUC | Active survival | Top20 actives | Clash-like top20 |
|---|---:|---:|---:|---:|---|---:|---:|
| current_box_balance | 1.521 | 0.000 | 0.000 | 0.3314 | 15/15 | 3 | 0 |
| search_fast_current_box | 1.535 | 0.000 | 0.000 | 0.2686 | 15/15 | 2 | 0 |
| search_detail_current_box | 1.615 | 0.000 | 0.000 | 0.2762 | 15/15 | 1 | 0 |
| box_center_crystal_centroid | 1.581 | 0.000 | 0.000 | 0.3219 | 15/15 | 1 | 0 |
| box_padding_4a_crystal_centroid | 1.780 | 0.000 | 0.000 | 0.2990 | 15/15 | 4 | 0 |
| box_padding_8a_crystal_centroid | 1.753 | 0.000 | 0.000 | 0.2724 | 15/15 | 1 | 0 |
| seed_repeat_current_box_001 | 1.298 | 0.000 | 0.000 | 0.3181 | 15/15 | 3 | 0 |

Interpretation:
- Redocking is consistently acceptable (<2 A), so the current pocket geometry is not the primary failure mode.
- The 50-ligand hard panel remains decoy-heavy; search mode and simple box padding/center changes are not enough.
- Continue with receptor prep/protonation and ligand charge/protonation handling before any full PPARD rerun.
