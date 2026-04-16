# Changelog

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
