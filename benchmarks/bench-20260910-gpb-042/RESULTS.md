# GPB042 completed benchmark

Status: completed and validated on 2026-09-10. This is an as-supplied,
failure-last baseline, not exact reproduction of a historical protocol.

## Frozen protocol

- Original DUD release 2 GPB: 2192 records, 52 actives, 2140 decoys.
- PDB1A8I chain A; optional reviewed LLP A680 linked-residue policy, neutral
  enolimine/phosphate dianion; waters A1014/A1015/A1017/A1018 retained.
- The receptor retains 29 unresolved residues as an explicit limitation; no
  claim that they were rebuilt. The deposited ILE380 discrepancy is retained.
- Validated receptor SHA256:
  `b23d8030b4a86f56475273fde11705d1cbd671a50b619d61ca0e7e43af6420b8`.
- Generalized_v1 ligand preparation, rigid macrocycles, no enumeration;
  RDKit2026.3.5 and Meeko0.7.1.
- UniDock/Vina, exhaustiveness32, seeds1509202611/1509202623;
  three identical label-blind shards731/731/730 per seed.
- No target-specific global default changes. LLP/water retention remains optional.

## Results

| Metric | Seed1509202611 | Seed1509202623 |
|---|---:|---:|
| EF1%, top22 | 3.832168 | 3.832168 |
| EF2%, top44 | 6.706294 | 6.706294 |
| EF5%, top110 | 4.981818 | 4.981818 |
| Active counts at these cutoffs | 2 / 7 / 13 | 2 / 7 / 13 |
| ROC-AUC | 0.7557 | 0.7555 |
| BEDROC20 | 0.2423 | 0.2407 |
| Scored / failed | 2185 / 7 | 2185 / 7 |

Tie-aware ceiling-cutoff EF and deterministic-name cutoff EF agree. The seven
known import failures (two actives, five decoys) remain in the full2192/52
denominator. No additional failures occurred. This is not successful preparation
of every library record. Both seeds are reported, with no best-seed selection.
Equal cutoff metrics do not imply identical poses/scores; top22 identity overlap
is21 and maximum matched score difference is6.008 kcal/mol.

## Validation and source artifacts

All six Kaggle kernels completed. Exact identity coverage, receptor hashes,
paired prepared-ligand hashes, successful docking commands/settings, and150
retained score/identity-matched SDF poses passed operational validation.
Native redocking038 previously passed3/3 with RMSDs0.711/0.685/0.672 A.

Original evidence remains in place; no files were deleted or moved:

- `tmp/gpb-fullscreen-042/experiment_freeze.json`
- `tmp/gpb-fullscreen-042/operational_gate.json`
- `tmp/gpb-fullscreen-042/metrics.json`
- `tmp/gpb-fullscreen-042/seed_0_ranking.csv`, `seed_1_ranking.csv`
- `tmp/gpb-fullscreen-042/recovered/` (all six raw outputs)
- `tmp/gpb-fullscreen-042/evaluate.py`
- Receptor037, redocking038, blocked039, converter040 and SDF audit041 remain
  separate provenance records, not overwritten by042.

## Published comparison and claim boundary

Empereur-Mot et al.2015 Additional file3/TableS3 reports Vina1.1.2 GPB
EF2=2.87 and EF5=4.22. Source: https://doi.org/10.1186/s13321-015-0100-8;
local audited evidence row `EM15_GPB` in
`benchmarks/vina_family_ef_literature_review/evidence_table.csv`.
Current values are numerically higher, but different preparation and engine
implementation prevent an exact-reproduction claim or proof of perfect prep.
No isolated causal benefit of LLP or waters was measured by this experiment.

## Operational issue

All original local runners misclassified status-query SSL failures as kernel
failures. An independent monitor recovered the still-running jobs without
resubmission. Raw local `launch_result.json` failure statuses are not the remote
outcomes. The subsequent polling fix changes no scientific outputs or settings.
