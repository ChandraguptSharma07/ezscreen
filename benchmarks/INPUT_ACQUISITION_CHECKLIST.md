# Benchmark Input Acquisition Checklist

Use this file before starting a real benchmark campaign or benchmark smoke run.

## 1. Decide the run type

Pick one:

- Archived-run benchmark
  - Use this when `scores.csv` already exists from a finished ezscreen run.
  - This is the fastest path for the first smoke benchmark.
- From-scratch benchmark
  - Use this when you still need to produce the screening outputs yourself.
  - This requires receptor, pocket, and full ligand-input collection.

## 2. Files required for every benchmark campaign

Create or confirm:

- `benchmarks/<benchmark_id>/campaign_manifest.json`
- `benchmarks/<benchmark_id>/inputs/targets.json`
- `benchmarks/<benchmark_id>/inputs/actives/<target>.smi`
- `benchmarks/<benchmark_id>/inputs/decoys/<target>_decoys.smi`

For each target entry in `targets.json`, fill at minimum:

- `target_id`
- `target_class`
- `primary_structure.kind`
- `actives_file`
- `decoys_file`

Current validator rule:

- every `actives_file` path must exist
- every `decoys_file` path must exist

## 3. Archived-run smoke benchmark checklist

Get these files first:

- Known actives file
  - Destination: `benchmarks/<benchmark_id>/inputs/actives/<target>.smi`
  - Source: the benchmark paper, benchmark dataset, curated target set, or your frozen project active list
- Decoys file
  - Destination: `benchmarks/<benchmark_id>/inputs/decoys/<target>_decoys.smi`
  - Source: frozen decoy set for the same target
- Archived scores file
  - Destination: `benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/merged_outputs/scores.csv`
  - Source: completed ezscreen run archive
- Cell-local actives file
  - Destination: `benchmarks/<benchmark_id>/<target>/<regime>/<run_label>/inputs/actives.smi`
  - Source: usually copied from the campaign-level actives file

Minimum rule for `run-cell`:

- `inputs/actives.smi` must exist
- `merged_outputs/scores.csv` must exist

You do not need these just to benchmark an archived run:

- raw protein PDB file
- prepared receptor file
- pocket file
- full ligand library used to create the screen

## 4. From-scratch benchmark checklist

If you are generating a new benchmark cell rather than benchmarking an archived run, also get:

- Original receptor structure
  - Source: PDB or other frozen structure source recorded in `targets.json`
  - Archive the exact identifier used
- Prepared receptor file actually docked
  - Archive the exact file used by the run
- Pocket definition
  - One of:
    - co-crystal ligand metadata
    - residue list / residue box
    - operator-selected pocket output
- Screened ligand input set
  - Full library or the exact subset screened for that cell
- Run configuration snapshot
  - engine path
  - pocket mode
  - ablation flags
  - seeds if controlled

Recommended archive rule:

- keep both source receptor and prepared receptor
- keep enough metadata to reproduce pocket selection without guessing

## 5. How to get each file type

### Active-set file

Get:

- one frozen SMILES file per target

Prefer:

- literature benchmark sets
- curated internal target-specific actives already used in project evaluation
- a manually reviewed exported active list from the chosen source

Do before freezing:

- canonicalize the SMILES
- preserve stable compound identifiers after the SMILES column whenever available
- remove obvious duplicates
- record source note or DOI in neighboring notes or metadata

Do not:

- swap in a different active list under the same `benchmark_id`

### Decoy-set file

Get:

- one frozen decoy SMILES file per target

Prefer:

- decoys generated once with the project decoy tooling and then archived
- an externally curated decoy set tied to the same target

Do before freezing:

- record how the decoys were generated or obtained
- keep the exact file used by the campaign

Do not:

- regenerate in place after a campaign has started

### `scores.csv`

Get:

- the merged screening score table from the finished run being benchmarked

Prefer:

- archived ezscreen output from the exact run you want to evaluate

Check:

- the file is the final merged score table, not an intermediate raw output
- the file belongs to the same target and regime named in the benchmark cell

### Receptor structure

Get:

- the crystal or model structure named in `targets.json`

Prefer:

- a PDB structure with a clearly documented ID
- an AlphaFold model only when that regime is part of the benchmark plan

Record:

- structure identifier
- retrieval date if downloaded
- any caveat such as apo, weak ligand, low resolution, missing residues

### Pocket definition

Get:

- the exact pocket definition used by the screen

Common sources:

- co-crystal ligand
- residue-defined box
- P2Rank or other operator-selected pocket

Record:

- ligand metadata, residue list, or pocket-selection method

## 6. Per-target preparation table

Fill this before running the first real cell.

| Item | Status | Notes |
|---|---|---|
| target chosen |  |  |
| target class recorded |  |  |
| primary structure ID recorded |  |  |
| active set acquired |  |  |
| active set canonicalized |  |  |
| decoy set acquired |  |  |
| decoy provenance noted |  |  |
| `targets.json` entry filled |  |  |
| archived `scores.csv` located |  |  |
| cell scaffold created |  |  |
| `inputs/actives.smi` copied into cell |  |  |
| `merged_outputs/scores.csv` copied into cell |  |  |
| campaign validation passed |  |  |
| benchmark cell run completed |  |  |
| campaign summary generated |  |  |

## 7. First smoke benchmark recommendation

For the first smoke run, use the archived-run path.

Recommended target bundle:

- one target
- one receptor regime
- one existing `scores.csv`
- one frozen `actives.smi`
- one frozen `decoys.smi`

This is enough to prove:

- campaign scaffold works
- validator catches missing benchmark inputs
- `run-cell` generates benchmark outputs
- campaign summary generation works

## 8. Ready-to-run gate

You are ready to run the first benchmark smoke test when all of these are true:

- `campaign_manifest.json` exists
- `inputs/targets.json` exists
- every target in `targets.json` points to real `actives_file` and `decoys_file`
- cell directory exists
- cell `inputs/actives.smi` exists
- cell `merged_outputs/scores.csv` exists

Then run:

```powershell
ezscreen benchmark validate-campaign benchmarks/<benchmark_id>
ezscreen benchmark run-cell benchmarks/<benchmark_id>/<target>/<regime>/<run_label>
ezscreen benchmark summarize benchmarks/<benchmark_id>
```
