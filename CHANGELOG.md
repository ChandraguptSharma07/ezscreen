# Changelog

## v1.12.1 — 2026-06-30

> Theme: quality of life. The run wizard's overloaded options step is split into clear, single-purpose screens.

### Changed

- **Run wizard reorganised into concern-based steps** — the crowded "Run Options" step (which mixed prep, search, and compute choices) is now three separate screens: **Ligand Prep** (ADMET pre-filter, force field, enumeration), **Engine & Scoring** (search depth), and **Compute** (local CPU / Kaggle GPU, exhaustiveness, multi-account assignment). Same options, clearer flow — no change to how a run is configured or submitted

### Internal

- Groundwork for upcoming engine selection (a docking-engine profile registry and a GNINA CNN-rescore Kaggle kernel + runner) is included but not yet wired into the UI

## v1.12.0 — 2026-06-27

> Theme: better chemistry in. Make ligand prep correct and explicit — the right force field, the right protonation/tautomer/stereo form, and a check that the 3D conformer isn't broken — instead of silently docking one fixed guess.

### Added

- **Ligand force-field choice** — pick `MMFF94` / `MMFF94s` / `UFF` for conformer minimisation instead of MMFF-only. Set a default in Settings and override it per run in the wizard (shown in the confirm summary). Threaded through local prep and the Kaggle prep cell. GAFF is deliberately out (needs AmberTools, too heavy for the Kaggle image)
- **Protonation / tautomer / stereo / ring enumeration** — optional Gypsum-DL backend fans each input SMILES out to its plausible forms; enumeration runs on Kaggle's GPUs (off by default, per-run toggle in the wizard, variant types + cap in Settings). Variants are tagged `<name>_v{k}` and dock independently so the best-scoring form wins. Fails soft to the single input form when Gypsum-DL is unavailable. New optional `enumerate` extra (`gypsum-dl`)
- **Collapse variants in results** — toggle (press `c`, or the Variants panel button) to fold enumerated forms back to one row per source molecule (best form kept, with a form count); export honours the toggle (xlsx gains a `forms` column). Appears only when a run has variants
- **3D conformer QC** — after embed+optimise (locally and in the Kaggle prep cell), flag `non_finite_coords` / `bad_bond_length` / `steric_clash`; surfaced per-compound in the results viewer detail panel and counted in the prep report. Flags but does not reject — strained conformers still dock, labelled

### Changed

- Results detail panel is now scrollable with slimmer, full-width controls (it previously clipped lower sections off-screen)
- Kaggle dataset-upload errors now include the API response body, so the actual reason for a 4xx is visible instead of a bare status code

## v1.11.0 — 2026-06-26

> Theme: trust the results. Quality and export features that run locally and post-run on a run's existing `scores.csv` / `poses.sdf` — no Kaggle kernel, no extra GPU cost.

### Added

- **Pose validity gate (PoseBusters)** — every docked pose is checked against PoseBusters' "dock" geometry/energy/clash checks; the results viewer shows a green/red **Valid** column (with the failed checks per compound), and the HTML report gets a Pose Validity section (valid/invalid counts + a bar of the most frequent failed checks). Requires the optional `analysis` extra; degrades gracefully when absent
- **Excel + SDF export** — an "Export Hits" button writes `hits.xlsx` (styled: frozen header, numeric score, amber high-LE cells) and `hits.sdf` (poses with score/LE/BEI and other fields as SD properties) into the run output dir
- **Export count selector** — an "Export top N" field; the xlsx honours any N (the full scores list is local and cheap), the sdf is bounded by the poses actually returned
- **Configurable poses-return cap** — the number of 3D poses brought back per shard from Kaggle (previously hardcoded at 25) is now a `[results] poses_returned` setting, exposed in Settings — the lever for trading 3D detail against transfer cost
- **Methods paragraph** — a "Copy Methods" button generates a publication-ready Methods paragraph (receptor/AlphaFold + chains, pocket method + box, prep, ADMET, engine + search depth, with tool citations) from the run's metadata; also written to `methods.txt` on completion
- **Hit flags + notes** — flag any hit green/yellow/red (press `f`) and attach a note; persisted per run in the checkpoint DB and surfaced as a Flagged Hits section in the HTML report
- **Score-type-aware results** — merged runs record their native score type in a `results_meta.json` sidecar; the viewer's score column and the report's score axes are labelled accordingly (Vina kcal/mol today; the seam for GNINA/DiffDock score types later)

### Changed

- `openpyxl` is now a core dependency (Excel export). PoseBusters/ProLIF/MDAnalysis live in a new optional `analysis` extra — install with `pip install ezscreen[analysis]`

## v1.10.1 — 2026-06-24

### Added

- **Export dropdown restored on the Mol\* viewer** — PNG of the current 3D view, SVG of the 2D diagram, and a 360° WebM rotation video, all of which shipped in the old 3Dmol.js viewer (v1.9.3) and were dropped during the Mol\* migration
- **Residue labels in 3D** — a Labels toggle overlays the name and number of every interacting residue, projected from each contact's Cα to screen pixels each frame (the viewer bundle lacks Mol\*'s label query language, so the chips are HTML overlays)
- **Distance labels in 3D** — distance chips at the midpoint of each active interaction, coloured to match the interaction type, driven by the same per-frame projection as the residue labels
- **Hydrophobicity pocket surface** — a Pocket toggle draws a translucent molecular surface over the contact residues coloured on the Kyte-Doolittle scale (blue polar → white → red greasy), looked up per residue from the chain map so no hierarchy read is needed
- **Citation banner** — the footer now names UniDock (Yu et al. 2023) and PLIP (Salentin et al. 2015) for the methods section

### Fixed

- **2D SVG export rendered on white** — the export now frames to the diagram's actual content bounds and paints the current background behind it, so saved diagrams match the dark / light mode on screen instead of coming out transparent (white)
- **Residue and distance label chips missing from PNG exports** — the HTML overlay chips are now composited into the screenshot at their on-screen positions, since they aren't part of the WebGL scene

## v1.10.0 — 2026-06-24

### Changed

- **Interaction viewer engine: 3Dmol.js → Mol\*** — the 3D pose viewer was rebuilt on Mol\* (4.4.0) for native per-atom hover highlighting and cleaner cartoon / surface rendering; the 2D LIGPLOT-style diagram is unchanged
- **Residue panel redesign** — the bottom sequence strip is now a Maestro-style panel with a tick ruler, monospace residue tiles coloured by PLIP contact type, a chain selector (defaults to a single chain on many-chain receptors, with dots marking the chains the ligand contacts), an interacting-residue filter, and Fly / Select modes (click a residue to fly the camera to its Cα, or multi-select with shift-range)
- **Colouring consolidated into one Colour menu** — a two-layer model: a global atom-colour scheme (CPK / secondary structure / chain / residue type / B-factor / rainbow N→C / solid) with an editable per-element palette, plus per-residue colour overrides painted from a residue picker or the strip selection; scheme changes, element Apply, and Reset all reframe the camera to a default fit

### Added

- **Mol\* 3D toolbar** — an FX toggle (edge outlines + ambient occlusion), light / dark canvas, a Bindings toggle that draws each PLIP interaction as a thin coloured cylinder anchored on the contacting side chain, a Measure mode (click two atoms to drop a distance label), and a Views dropdown to snapshot the camera and fly back to named positions
- **Style dropdown** — protein, ligand, and interacting-residue representations in one nested menu, with a switchable ligand representation
- **Per-element colour customisation** — editable C / N / O / S / H / P / halogen swatches with per-element tickboxes and an Apply step, applied only to atom representations (sticks / spacefill / surface / lines), not the cartoon where individual atoms aren't visible
- **Per-residue colouring** — paint the selected residues a chosen colour, with Decolour-selected and Decolour-all actions and a per-residue override list

### Fixed

- **Per-residue and element colours not applying** — the Mol\* viewer bundle does not expose `StructureProperties`, so the colour callbacks silently fell back to carbon; atom / residue / chain identity is now read directly off the model's atomic hierarchy, so painting and CPK colours work
- **Element colours on line representations** — bond locations carry no element index, so every bond read as carbon and the whole structure turned one colour; bonds are now coloured by their first atom
- **Scroll zoom-out clamped too tight** — the trackball auto-clamped the zoom-out distance to roughly the structure radius, so the wheel could not pull back far enough; widened the envelope so the whole structure can always be framed
- **Viewer script silently broken by an unescaped newline** in the synthetic interaction-PDB join

---

## v1.9.4 — 2026-05-16

### Fixed

- **T4×2 GPU allocation** — Kaggle's server honours the deprecated `enable_gpu` bool over `machine_shape`, causing P100 to be allocated regardless of the T4 selection; fixed by forcing `enable_gpu: false` and routing all GPU selection solely through `machine_shape` (`NvidiaTeslaT4` / `NvidiaTeslaP100`)
- **UniDock v1.1.3 compatibility** — removed `--gpu_ids` flag (dropped in v1.1.3, caused exit=1 on every docking run); replaced `--gpu_batch` with `--ligand_index <txt_file>`; expanded stderr capture to 3000 chars so parse errors at the top of output are visible
- **Dataset upload retry** — Kaggle occasionally returns an empty HTTP body on transient 5xx; mapped `JSONDecodeError` to `KaggleServerError` so `upload_run_dataset()` retries up to 5× with exponential backoff instead of failing permanently
- **LE/BEI missing in single-account runs** — efficiency columns were only appended in the multi-account merger path; fixed so single-account results include LE and BEI
- **Empty `poses.sdf` PLIP crash** — PLIP runner now checks that `poses.sdf` is non-empty before submitting the analysis kernel
- **Meeko API break** — updated call sites for Meeko's changed public API
- **Kaggle file-list truncation** — dataset file listing was being silently truncated; switched to paginated fetch
- **`.ism` ADMET filter** — `.ism` files (Daylight SMILES) are now accepted alongside `.smi`
- **`AF:UniProt` in CLI** — `ezscreen run` CLI command now accepts `AF:UniProt` receptor syntax, matching the TUI wizard
- **GPU selector hidden in wizard** — the GPU type radio set was not visible on the options step in certain layout states; display logic corrected

---

## v1.9.3 — 2026-05-06

### Added

- **Publication-quality 2D interaction diagram** — the interaction viewer now includes a full LIGPLOT-style radial SVG diagram alongside the 3D view; toggled via a 3D/2D button pair in the toolbar
- **Eyelash glyphs for hydrophobic contacts** — LIGPLOT convention: an arc with radiating hash lines facing away from the ligand, oriented toward the nearest interacting atom
- **Inline RDKit ligand structure** — the 2D ligand SVG is embedded via `<g transform>` rather than `<image>`; transparent background, no white-box artefact; bonds recoloured for dark mode automatically
- **Site view / full compound toggle** — switches between a cropped viewBox centred on the binding site and the full ligand layout; both views available via sidebar sub-buttons in 2D mode
- **Dark mode in 2D diagram** — Dark BG and Distances toggles now work in 2D mode (were previously disabled on mode switch); toggling redraws the diagram immediately
- **Type symbols on glyphs** — π for π-stack / π-cation, ± for salt bridge, X for halogen; rendered as superscripts at the top-right rim of each residue circle
- **Collision-spread layout** — 10-pass O(n²) force spread prevents overlapping glyphs; angular separation enforced before radial placement

### Fixed

- **SVG painter's model** — connection lines were drawn after the ligand SVG and appeared visually on top of bond paths; reordered so lines are emitted first and render behind the molecule
- **Atom coordinate scaling** — `transformPt` was scaling x and y independently with separate `vw`/`vh` factors, stretching positions when the cropped viewBox is non-square; replaced with a uniform fit-inside scale (`Math.min(LIG_W/vw, LIG_H/vh)`) matching RDKit's own layout model
- **`ix_atom_pts` index bug** — connection lines were indexed with the filtered visible-array counter; `ix_atom_pts` is parallel to the full `compound.interactions` array; fixed by iterating all interactions with `origI` and skipping toggled-off types without advancing the index
- **Glyph labels unified** — hydrophobic residue labels previously used a sine-based direction calculation that produced near-zero offsets for horizontal glyphs; all residue types now show name + number + chain consistently inside the circle body

---

## v1.9.2 — 2026-05-05

### Fixed

- **P2Rank on Windows** — invoked Java directly instead of via `prank.bat` to bypass `JAVA_HOME` and Win32 execution issues; stripped leading/trailing whitespace from padded CSV column headers (P2Rank pads all headers with spaces, causing `KeyError` on every row and silently returning empty results); cleared stale output directory before each run; raised an explicit error on non-zero exit instead of returning empty results

---

## v1.9.1 — 2026-05-04

### Fixed

- **PLIP dataset path** — Kaggle now mounts datasets under `/kaggle/input/datasets/<username>/` instead of `/kaggle/input/<slug>/`; notebook now searches `/kaggle/input` recursively for `receptor_prep.pdb` so it works regardless of mount structure
- **PLIP install on Kaggle CPU** — `pip install plip` fails on the system Python (`/usr/bin/python3`) due to build backend errors; fixed by installing `openbabel-wheel` + `lxml` via pip and copying the plip source directly into site-packages via `shutil.copytree`, bypassing the build system entirely
- **Receptor END record blocking ligand** — `receptor_prep.pdb` ends with an `END` record; appending ligand HETATM lines after it caused PLIP to see an empty complex; now strips `END`/`CONECT`/`MASTER` from the receptor and inserts `TER` before the ligand block
- **PLIP `pistacking` attribute** — extraction code used `bs.pistacks` which does not exist in PLIP 2.x; corrected to `bs.pistacking`
- **Hydrophobic contact distance** — `HydrophobicInteraction` uses `distance` not `dist`; was always showing `0.00 Å`; fixed with `getattr(hc, 'distance', getattr(hc, 'dist', 0.0))`
- **PLIP dataset retry** — re-running PLIP on the same run uploaded a new version to an existing dataset slug; `dataset_create_new` was failing silently; now falls back to `dataset_create_version` when the dataset already exists
- **Interaction viewer protein visibility** — receptor cartoon was near-invisible on dark background; updated to spectrum coloring with surface, binding site residue sticks via `addStyle`, and `center`+`zoom` instead of `zoomTo(ligand)` for proper pocket context

---

## v1.9.0 — 2026-05-04

### Added

- **Pose Interaction Viewer** — after docking completes, an "Analyse Interactions" button in the results viewer triggers a second user-triggered Kaggle CPU kernel that runs PLIP on the top-N poses; the kernel uploads `receptor_prep.pdb` + `plip_poses.sdf` + `scores_top_n.csv`, runs per-compound PLIP analysis, writes `interactions.json`, and downloads it back; clicking "Open Interaction Viewer" generates a self-contained browser HTML with a 3Dmol.js receptor + ligand scene, per-interaction cylinders, six interaction-type toggles (H-bond, hydrophobic, π-stack, π-cation, salt bridge, halogen), a sidebar residue table, compound dropdown, and a "predicted pose — not experimentally validated" disclaimer
- **`receptor_prep.pdb` saved alongside PDBQT** — `prep_receptor()` now copies the pdbfixer-cleaned PDB to `receptor_prep.pdb` in the output directory; path is persisted in `resume.json` so the PLIP runner can locate it for any run
- **Configurable interaction top-N** — `[results] interaction_top_n = 20` in `config.toml`; editable from Settings under a new Results section; controls how many compounds are sent to the PLIP kernel

---

## v1.8.1 — 2026-05-04

### Fixed

- **poses.sdf wrote 0 poses** — cell 8's inner `except Exception: pass` silently swallowed every Meeko conversion failure; added per-compound error logging and a two-path fallback: Meeko ≥ 0.5 path first, obabel fallback if that fails; `obabel` is now installed in cell 2 alongside meeko and gemmi so the fallback is always available
- **Accounts set to 0 received 1 shard each** — blank input and explicit `0` both stored `shard_count=0`; the runner then applied `0 or 1 = 1` giving every excluded account one notebook; blank input now stores `None` (auto), `0` means explicitly excluded; `run_multi_account_screening()` filters out `shard_count=0` accounts before any notebook count calculation; added guard returning a clear error if all accounts are excluded

### Added

- **Configurable 3D prep location** — Settings screen now has a "Run 3D prep on Kaggle GPU" toggle under Ligand Pre-filter (default on); when off, ETKDGv3 + MMFF + Meeko runs locally and PDBQT shards are uploaded to Kaggle; persisted as `[prep] prep_on_kaggle` in `config.toml`; both single-account and multi-account paths respect the setting
- **`lig_id` SDF property** — docked pose SDF entries now carry an explicit `> <lig_id>` property alongside `_Name` and `score`, enabling reliable correlation between `poses.sdf` and `scores.csv`

### Fixed (template)

- **`mmff_max_iters` missing from single-account template render** — single-account Kaggle path in `run_wizard.py` was not passing `mmff_max_iters` to the Jinja2 template render, so the notebook always used the template default rather than the configured value; multi-account path in `runner.py` was already passing it correctly

---

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
