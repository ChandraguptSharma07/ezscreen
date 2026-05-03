# Changelog

## v1.8.0 — 2026-05-03

### Added

- **Multi-account Kaggle submission** — `run_multi_account_screening()` in `ezscreen/backends/kaggle/runner.py`; splits shard list across configured accounts using `_split_shards()` (respects per-account `shard_count`; 0 = auto-distribute evenly); submits one dataset+kernel per account sequentially under `_KAGGLE_API_LOCK` to prevent env-var races; polls all kernels concurrently via `ThreadPoolExecutor`; downloads and merges results with `merge_shard_results`
- **Account shard assignment UI in Run Wizard step 4** — when ≥ 2 Kaggle accounts are configured and local mode is off, a dynamic list of account rows appears, each with a numeric input for shard count (blank = auto); the section hides automatically when the "Run locally" toggle is enabled
- **`_KAGGLE_API_LOCK`** global lock in `runner.py` serialises all `os.environ` credential switches and `authenticate()` calls so concurrent threads never clobber each other's credentials
- **Prep-on-Kaggle** — ligand 3D embedding and PDBQT conversion now runs inside the Kaggle notebook instead of locally; `shard_raw()` in `ligands.py` shards raw SMILES for upload, then the notebook runs ETKDGv3 + MMFF + Meeko on the GPU instance; cuts local CPU time to near-zero for large libraries
- **GPU type selection** — Run Wizard step 4 now shows a radio button to pick between P100 (16 GB, default) and T4 × 2 (32 GB, uses both GPUs via `--gpu_ids 0,1`); selection propagates to `kernel-metadata.json` via the new `accelerator` parameter on `push_kernel()`
- **Docking failure log** — each Kaggle notebook now writes `failed_docking.csv` alongside `scores.csv`; records every ligand that had a valid output PDBQT but was rejected, with `reason` (`score_ceiling`, `score_floor`, `no_remark`, `unparseable_score`, `non_finite_score`) and the raw score string; included in `output.zip`
- **`unscored_reasons.csv`** — merger collects `failed_docking.csv` from all shard dirs and combines it with compounds that got no output PDBQT at all (`no_pose`) and GPU-size-filtered entries (`gpu_size_filter`) into a single file in the run output directory
- **Ligand pre-filter settings** — Settings screen now exposes GPU size filter toggle, max heavy atoms, max MW, and max rotatable bonds; values persisted under `[prep]` in `config.toml`
- **Score ceiling setting** — score ceiling (default 0.0 kcal/mol) is now configurable in Settings alongside the existing floor; both applied consistently by the merger
- **Configurable score floor** — Settings screen now exposes an enable/disable toggle and a custom threshold (default −15.0 kcal/mol); read at runtime by both `run_local_screening` and `merge_shard_results`
- **Local docking performance settings** — exhaustiveness and CPU core count are now configurable in Settings and persisted under `[local]` in `config.toml`
- **Per-run exhaustiveness in Run Wizard** — an Exhaustiveness input appears in Step 4 when local mode is on; search depth radio buttons (UniDock-specific) are hidden for local runs since they have no effect on Vina
- **MMFF minimisation mode** — Settings screen now exposes a "Run MMFF to convergence" toggle (default on) and a fixed iteration count input; convergence mode (`maxIters=0`) is 1.62× faster than the previous hardcoded `maxIters=200` on typical drug-like libraries with no change in pass rate; value persisted under `[prep] mmff_max_iters` in `config.toml` and applied in both the Kaggle notebook template and the local prep path

### Fixed

- **Local docking returned 0 hits** — `run_local_screening` was passing PDBQT shard files to `_sdf_to_pdbqt` which called `Chem.SDMolSupplier` on them, yielding no molecules; replaced with `_split_pdbqt_shard` that reads the multi-molecule PDBQT shard directly, splits on `TORSDOF` boundaries, and writes individual ligand files for Vina; SMILES enrichment now reads from `index.csv` written by `ligand_prep` instead of trying to parse the PDBQT as SDF
- **Local Vina used only 1 CPU core** — `--cpu 1` changed to `--cpu 0` so Vina auto-detects all available cores; default exhaustiveness lowered from 8 to 4 for local mode
- **No SMILES or real names in local results** — `_load_smiles_index` was keying on the human name instead of the lig_id; `ligand_prep` now injects `REMARK lig_id <id>` into every PDBQT block so `_split_pdbqt_shard` can correlate blocks back to `index.csv`; rows now carry both `name` and `smiles` columns
- **Clustering crashed with "max() arg is an empty sequence"** — `_show_cluster_result` and `_cluster_section_html` both called `max(result.sizes)` without guarding against an empty list (no SMILES available); fixed with early return and a clear "No SMILES data" message
- **Detail panel showed "—" for all local hits** — `results_viewer` was reading `row["name"]` but local CSV uses `row["ligand"]`; now falls back correctly
- **`merger.py` score floor was hardcoded** — Kaggle merge path ignored the configurable score floor setting; now reads from `config.toml`
- **CPU cores setting was not wired** — `cpu_cores` config key was added but never passed to Vina's `--cpu` flag; fixed by threading it through `_run_vina`
- **Local docking backend never ran** — `run_wizard.py` step-options validation read `opt-admet` and `opt-depth` but skipped `opt-local`, so `ctx["run_locally"]` was never set and `_do_submit` always fell through to Kaggle; fixed by reading the switch in `_validate_step` and branching on `ctx["run_locally"]` in `_do_submit`
- **Score regex silently dropped 89% of poses** — the `REMARK VINA RESULT` pattern `[-\d.]+` didn't match `nan`, `inf`, or scientific-notation overflow scores, so those PDBQTs were silently skipped; pattern updated to `\S+` with explicit float parse and `math.isfinite` check
- **PDBFixer added terminal atoms that crashed Meeko** — `addMissingAtoms` and `addMissingHydrogens` were being called, causing PDBFixer to insert OXT and N-terminal H atoms that produced valence-5 carbons RDKit couldn't sanitize; both calls removed since Meeko assigns its own H and atom types from residue templates
- **Blank chain IDs caused no chains detected** — PDB files with no chain identifier in column 22 returned an empty list from `get_chains()`; now returns `[" "]` so prep can continue
- **SMTP auth was missing** — email notifications didn't call `smtp.login()`, so any server requiring authentication silently failed; added login with configurable `smtp_password`
- **Account assignment rows overlapped** — wizard account rows had no explicit height so Textual collapsed them together; fixed with `acct-row` CSS class setting `height: 3` and proper label width
- **Account rows past "primary" were inaccessible** — account assignment section was a plain `Vertical` with no scroll, so overflow rows were clipped; replaced with `VerticalScroll` capped at 16 rows

### Changed

- `run_wizard.py` submit path now branches: single account → existing `run_screening_job`; multiple accounts → new `run_multi_account_screening`; confirm summary shows assigned shard counts per account in the log before submission
- Confirm summary now shows "Backend: Local CPU (AutoDock Vina) / Kaggle GPU" line

---

## v1.7.0 — 2026-04-16

### Added

- **Team accounts** — `ezscreen/auth.py`: `add_team_account()`, `remove_team_account()`, `list_team_accounts()`, `get_all_kaggle_accounts()`; collaborator credentials stored under `[team.<name>]` in `~/.ezscreen/credentials`; validated against kaggle.json at add time
- **Team Accounts TUI screen** — `ezscreen/tui/screens/team_accounts.py`; lists collaborators with name / email / Kaggle username / path; add form with consent checkbox; remove selected; accessible from home Quick Actions
- **Round-robin account selection on resume** — `resume_failed_shards()` distributes failed shards across all configured accounts in round-robin order; per-account lock prevents concurrent env-var clobbering; effective Kaggle username per shard derived from assigned account
- **Desktop notifications** — `ezscreen/notify.py`; `plyer`-based toast on run complete / failed / timeout; graceful no-op if `plyer` not installed or display unavailable; toggled via Settings → Desktop notifications switch
- **Email notifications** — `ezscreen/notify.py`; SMTP with STARTTLS; configurable host, port, from/to in Settings; no-op when host not set
- **Notification settings** — Settings screen extended with Notifications section: desktop toggle, SMTP host, port, from/to address; all values persisted under `[notify]` in `config.toml`
- **Poller notification hook** — `ezscreen/backends/kaggle/poller.py` calls `notify.send_run_complete()` after every terminal kernel status (complete, failed, timeout)

---

## v1.6.0 — 2026-04-16

### Added

- **Shard resume** — `resume_failed_shards(run_id, work_dir)` in `ezscreen/backends/kaggle/runner.py`; reads `work_dir/resume.json` (written at submission time) to locate receptor, shard files, and notebook; resubmits failed shards with up to 2 kernels running in parallel via `ThreadPoolExecutor`; uses a `threading.Lock` for all checkpoint writes during concurrent execution; merges new partial results back into the main `output/scores.csv` via `merge_shard_results`
- **Resume button in status monitor** — appears next to View/Download/Clean only when the selected run has at least one shard in `status = 'failed'`; runs resume in a background thread and refreshes the run table on completion

---

## v1.5.0 — 2026-04-16

### Added

- **Scaffold clustering** — `ezscreen/results/clustering.py`; Butina algorithm on Morgan fingerprints (radius 2, 2048 bits); configurable Tanimoto cutoff (default 0.4); returns cluster labels, centroid indices, and sizes; `export_centroids()` writes centroid SMILES as `.smi`
- **Cluster Hits button in results viewer** — runs clustering in a background thread; shows cluster count, largest cluster size, and singleton count inline; saves `centroids.smi` to the run output directory
- **Scaffold cluster section in HTML report** — centroid 2D structure grid (top 20 clusters by size) with compound count per cluster; generated automatically when `write_results_report()` is called with `cluster=True` (default)
- **ProLIF interaction fingerprints** — `ezscreen/results/interactions.py`; wraps ProLIF + MDAnalysis to compute per-residue contact fingerprints from `poses.sdf` and the receptor PDB; gracefully returns `None` if ProLIF is not installed
- **Interaction heatmap in HTML report** — compounds × residues table coloured by dominant interaction type (H-bond, hydrophobic, π-stacking, etc.); included in report when interaction data is passed to `write_results_report()`
- **Ligand Efficiency (LE) and BEI columns** — computed by `merge_shard_results()` from docking score and RDKit heavy-atom count / MW; written to `scores.csv` automatically; LE values > 0.5 kcal/mol/atom are highlighted amber in the results viewer as a size-bias warning

---

## v1.4.0 — 2026-04-16

### Added

- **Expanded HTML results report** — `write_results_report()` in `report_html.py` generates a self-contained HTML page from any `scores.csv`; includes a run metadata table, score distribution histogram, score vs MW and score vs LogP scatter plots (matplotlib), and a 2D structure grid of the top 10 hits (RDKit SVG, inline); all images are base64-embedded so the file is portable
- **Open Report button in results viewer** — appears whenever a run has a `scores.csv`; on first click the report is generated in a background thread and opened in the default browser; on subsequent clicks the cached file is opened directly

---

## v1.3.0 — 2026-04-16

### Added

- **AlphaFold structure downloader** — `ezscreen/prep/alphafold.py`; fetches the EBI AlphaFold DB v4 model for any UniProt accession; parses per-residue pLDDT from the B-factor column; prints a summary of low-confidence spans (pLDDT < 50) and flags disordered loops (pLDDT < 40); escalates to a bold warning if any user-specified binding-site residues fall inside a low-confidence region
- **`AF:UniProt` input in Run Wizard** — receptor step now accepts `AF:P00533` alongside PDB IDs and local file paths; the structure is downloaded and cached to `~/.ezscreen/tmp/wizard/` before validation continues; co-crystal ligand detection is skipped for AlphaFold inputs

---

## v1.2.0 — 2026-04-16

### Added

- **ZINC15 library downloader** — `ezscreen/libraries/zinc.py`; download drug-like, lead-like, or fragment-like compound sets in three sizes (1k / 10k / 100k) or any custom count; purchasable-only filter; Rich progress bar; writes standard `.smi`
- **ChEMBL actives fetcher** — `ezscreen/libraries/chembl.py`; resolve a UniProt accession to a ChEMBL target, fetch binding-assay actives below a configurable IC50 threshold (default 1 µM), deduplicate SMILES, write `.smi`
- **Library Browser TUI screen** — source picker (ZINC / ChEMBL), all filter controls, background download thread, "Use this file" button that auto-fills the wizard ligand path on success
- **Library Browser in Quick Actions** — accessible from the home dashboard sidebar
- **Download library button in Run Wizard step 3** — opens Library Browser mid-wizard; selected file path is injected back into the ligand input automatically
- **Local AutoDock Vina CPU backend** — `ezscreen/backends/local/`; downloads the Vina binary on first use; runs docking locally without a Kaggle account; same CSV/SDF output format as the Kaggle runner
- **Run locally toggle in Run Wizard step 4** — switches the submission backend to the local Vina runner

---

## v1.1.0 — 2026-04-16

### Added

- **Benchmarking infrastructure** — retrospective enrichment validation for any completed screening run
- `ezscreen/benchmark/decoys.py` — DUD-E style property-matched decoy generator; fetches candidates from ChEMBL, filters by Tanimoto < 0.35 to ensure structural dissimilarity from actives
- `ezscreen/benchmark/metrics.py` — computes EF1%, EF5%, and AUC-ROC from a ranked active/decoy list using trapezoidal integration; no scipy dependency
- `ezscreen/benchmark/runner.py` — loads a known actives SMILES file, canonicalises SMILES via RDKit, cross-references against docking results, and calls the metrics module
- `ezscreen/results/report_html.py` — generates a self-contained benchmark HTML report with EF badges and an embedded ROC curve plot (matplotlib, base64 PNG)
- Results viewer now has a **Validate Setup** section: enter a known actives file, run the benchmark in a background thread, see EF1%/EF5%/AUC inline, and open the HTML report in the browser

### Changed

- `pyproject.toml` — added `matplotlib>=3.7.0` dependency (required for the ROC curve plot)
- GitHub Actions CI now publishes to PyPI automatically on version tag pushes (`v*`)

---

## v1.0.0 — 2026-04-15

First public release.

### Features

**Core pipeline**
- Full interactive docking pipeline: receptor prep → binding site detection → ligand prep → ADMET filtering → Kaggle GPU submission → results download
- Receptor preparation via PDBFixer and Meeko (pdbqt output); fetches directly from RCSB by PDB ID or accepts a local file
- AlphaFold 4-tier quality handling with per-chain confidence warnings
- Ligand preparation from SDF and SMILES files; Meeko + optional scrubber for tautomer enumeration and pH correction; sharding for large libraries
- Tiered binding site detection: co-crystal ligand, residue Cα box, P2Rank top-3, and blind whole-protein fallback
- ADMET pre-filter: Lipinski Ro5, PAINS, Brenk toxicophores, Veber oral bioavailability, optional Egan BBB; per-filter toggle with breakdown stats
- UniDock-Pro on T4 GPUs via Kaggle kernel; automatic fallback to UniDock if Pro is unavailable
- SHA-256 dataset deduplication to avoid redundant uploads
- SQLite checkpoint database — runs and shards tracked for retry and resume

**Results and reporting**
- CSV + SDF merge of multi-shard results; best-score deduplication; global sort by docking score
- JSON run report (Section 8.3 schema) + human-readable text summary
- Self-contained py3Dmol HTML viewer with per-compound 3D pose

**TUI (Textual full-screen interface)**
- Home dashboard with recent runs table, quick actions, and auth status
- Status monitor with 30s auto-refresh; shard-level detail; View/Download/Clean actions
- Results viewer with sortable hit table and compound detail panel
- Run Wizard — 5-step guided pipeline covering all prep and submission decisions
- Auth Setup screen with live credential validation
- Settings editor for all config.toml values
- Standalone ADMET filter screen
- DiffDock-L validation screen via NVIDIA NIM
- Help overlay (`?`) with all keybindings

**CLI**
- `ezscreen` — launches TUI when no subcommand given
- `ezscreen auth` — credential wizard for Kaggle and NIM
- `ezscreen status` — live run monitor with auto-refresh
- `ezscreen validate` — Stage 2 DiffDock-L validation
- `ezscreen admet` — standalone ADMET filtering on any CSV or SDF
- `ezscreen view` — Rich table + self-contained HTML results viewer
- `ezscreen clean` — remove Kaggle dataset and kernel artifacts

### Known limitations

- ADMET filtering is rule-based only (physicochemical filters). No ML-based toxicity prediction in v1.
- Docking scores are AutoDock Vina scores. They reflect binding pose quality, not binding affinity.
- P2Rank pocket prediction requires a manual install of the P2Rank binary (`~/.ezscreen/tools/p2rank/`).
- `Ctrl+V` paste in the TUI requires `Ctrl+Shift+V` on Windows terminals that intercept the standard binding.
- Conda-forge recipe is not yet available. Install via pip only.

### Upgrade notes

This is the initial release. No migration needed.
