# Benchmark Workspace

This directory stores reproducible benchmark campaigns and their archived artifacts.

See `INPUT_ACQUISITION_CHECKLIST.md` for the exact files to gather before a smoke run or full benchmark campaign.

## Directory model

- `benchmarks/<benchmark_id>/`
- `benchmarks/<benchmark_id>/inputs/`
- `benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/`

## Start a new benchmark campaign

Preferred CLI path (creates `benchmarks/<benchmark_id>/` in the current folder; `--root <dir>` picks another benchmark root):
- `ezscreen benchmark init bench-20260714-core-panel`
- `ezscreen benchmark init bench-20260714-core-panel --target hiv1_protease --regime crystal_holo --run-label run-001`

Manual path:
1. Pick a new `benchmark_id`, for example `bench-20260714-core-panel`.
2. Create `benchmarks/<benchmark_id>/inputs/`.
3. Copy these templates from `ezscreen/benchmark/templates/` into the new campaign:
   - `campaign_manifest.template.json` -> `benchmarks/<benchmark_id>/campaign_manifest.json`
   - `targets.template.json` -> `benchmarks/<benchmark_id>/inputs/targets.json`
4. Create target-specific active and decoy files under `benchmarks/<benchmark_id>/inputs/actives/` and `benchmarks/<benchmark_id>/inputs/decoys/`.
5. For each executed benchmark cell, archive outputs under:
   - `benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/`
6. Inside each cell directory, copy or generate:
   - `cell_manifest.json`
   - `provenance.json`
   - `failure_summary.json`
   - `notes.md`
   - `inputs/`, `raw_outputs/`, `merged_outputs/`, `logs/`, `reports/`

## Templates available

The templates ship with ezscreen in `ezscreen/benchmark/templates/`:

- `campaign_manifest.template.json`
- `targets.template.json`
- `cell_manifest.template.json`
- `provenance.template.json`
- `failure_summary.template.json`
- `notes.template.md`

## Rules

- Do not silently regenerate active or decoy files after a campaign starts.
- Preserve active and decoy identifiers after the SMILES column. Benchmark metrics match by result `name`/ID first and canonical SMILES second, because Kaggle-side protonation or tautomer preparation can change canonical SMILES. `metrics_summary.json` records `label_diagnostics` and `unscored_diagnostics` so ID-vs-SMILES matches, first active rank, and active entries lost to scoring/prep filters are auditable.
- Do not overwrite archived cell outputs in place; create a new `run_label`.
- Keep the benchmark campaign self-describing enough that another person can inspect the folder without guessing how it was produced.



## Submit one benchmark cell to Kaggle

Use this when docking must run remotely instead of local CPU. This command only shards raw ligand input locally; ligand prep and docking run inside the Kaggle notebook.

```powershell
ezscreen benchmark submit-kaggle-cell benchmarks/<benchmark_id>/<target>/<regime>/<run_label> `
  --receptor-pdbqt benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/inputs/receptor_fixed.pdbqt `
  --ligands benchmarks/external_sources/<target>/active_and_decoy.ism `
  --actives benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/inputs/actives.smi `
  --box-center x,y,z `
  --box-size x,y,z `
  --shards 4 `
  --accelerator nvidiaTeslaT4
```

After Kaggle completes, the command archives outputs into `raw_outputs/` and `merged_outputs/`, then regenerates `metrics_summary.json`, `failure_summary.json`, `provenance.json`, and `reports/benchmark_report.html`.

For AD4/AutoDock-GPU ablations, pass `--engine autodock-gpu`. The rendered Kaggle notebook builds AutoDock-GPU, generates AutoGrid4 maps from the prepared receptor, docks against `.maps.fld`, and writes AD4 scores with an `ad4_map` provenance column.

## Run one archived benchmark cell

Default path:
- `ezscreen benchmark run-cell benchmarks/<benchmark_id>/<target>/<regime>/<run_label>`

Optional overrides:
- `--actives <path-to-actives.smi>`
- `--scores <path-to-scores.csv>`
- `--score-col <column>` to rank by an explicit score column

To test ligand-bias-corrected ranking on archived scores:
- `ezscreen benchmark correct-scores benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/merged_outputs/scores.csv`
- `ezscreen benchmark run-cell benchmarks/<benchmark_id>/<target>/<regime>/<run_label> --score-col bias_corrected_score`
- `ezscreen benchmark correct-scores benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/merged_outputs/scores.csv --output-col balanced_corrected_score`
- `ezscreen benchmark run-cell benchmarks/<benchmark_id>/<target>/<regime>/<run_label> --score-col balanced_corrected_score`

`bias_corrected_score` uses `score + 8.0 * abs_formal_charge + 0.1 * heavy_atoms - 0.03 * tpsa`. `balanced_corrected_score` uses `score + 25.0 * abs_formal_charge - 0.8 * heavy_atoms - 0.03 * tpsa - 0.4 * rotatable_bonds`. Treat both as benchmark ablations until broader target validation passes.

Outputs written in the cell directory:
- `metrics_summary.json`
- `failure_summary.json`
- `reports/benchmark_report.html`


## Validate and summarize a campaign

Validate manifest + targets + referenced input files:
- `ezscreen benchmark validate-campaign benchmarks/<benchmark_id>`

Summarize all benchmark cells into campaign-level CSV tables:
- `ezscreen benchmark summarize benchmarks/<benchmark_id>`
- `ezscreen benchmark audit-campaign benchmarks/<benchmark_id>`

Use `--strict` in CI or release checks when the campaign must satisfy publishable-pipeline gates. The audit checks evidence shape and trust warnings: target diversity, real baselines, completed cells, repeat coverage, comparator coverage, ablation coverage, provenance, reports, weak-result warnings, and external benchmark-floor failures.

Summary outputs:
- `summary/metrics_table.csv` with EF/AUC, external baseline columns such as `external_baseline_status`, failed metrics, floors, and source, label diagnostics such as `first_active_rank`, `matched_by_identifier`, `unmatched_active_entries`, `active_unscored_entries`, active unscored reason counts, ranking diagnostics such as `ranking_flags`, top-window charged fractions, top-window decoy counts, charge/size-adjusted exploratory metrics, ligand-bias-corrected and balanced-corrected exploratory metrics, and prep-risk columns such as `prep_protocol_risk`, retained/removed metals/cofactors, and `prep_ablation_suggestions`
- Retained-Zn receptor prep records `receptor_prep.zn_coordination_qc` and `receptor_prep.ad4zn_prep` in provenance and may write `ad4zn_prep_manifest.json` as setup evidence for later AD4Zn/AutoDock-GPU ablations.
- `summary/failure_table.csv`
- `summary/throughput_table.csv`
- `summary/variance_table.csv`
