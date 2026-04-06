# ezscreen ??

**GPU-accelerated virtual screening powered by Kaggle T4 GPUs.**

`ezscreen` is a full-featured CLI tool that orchestrates virtual screening campaigns (molecular docking) completely inside the Kaggle environment, leveraging their free T4 GPUs. It handles everything from receptor preparation and ligand filtering to submitting Kaggle kernels, polling for completion, merging shards, and visualising the results.

## Prerequisites
- Python 3.11+
- A [Kaggle account](https://www.kaggle.com/) with an API token
- *(Optional)* NVIDIA NIM API key for Stage 2 validation with DiffDock-L

## Installation

```bash
pip install ezscreen
```

### Optional: Install scrubber for enhanced ligand prep

[forlilab/scrubber](https://github.com/forlilab/scrubber) provides tautomer enumeration and pH-driven protonation. It is not on PyPI, so install it separately:

```bash
pip install git+https://github.com/forlilab/scrubber.git
```

Without it, `ezscreen` falls back to RDKit-only preparation (still fully functional for most use cases).

## Setup

First, configure your credentials:

```bash
ezscreen auth
```
*(This will ask for your Kaggle `kaggle.json` credentials and optionally an NVIDIA NIM API key).*

## Quickstart

To run an interactive virtual screening campaign:

```bash
ezscreen run
```
You will be guided through a series of steps:
1. Receptors (PDB ID or local file)
2. Chain selection
3. Binding site definition
4. Ligand selection (from Enamine/ChemSpace subsets or a custom CSV/SDF/SMI file)
5. ADMET rules filtering
6. Submission to Kaggle.

After the run finishes, view the results interactively:
```bash
ezscreen view results/ezs-<RUN_ID>
```

## Features

- **End-to-End Orchestration**: Automatically chunks massive ligand sets, spawns parallel Kaggle notebooks, runs them on T4 GPUs, and downloads/merges the results.
- **Docking Engines**: Uses UniDock-Pro by default (if available), with automatic fallback to AutoDock Vina/UniDock.
- **ADMET Filtering**: Built-in interactive filters (Lipinski, Veber, PAINS, Brenk) to drop problematic compounds *before* wasting compute time.
- **DiffDock-L Validation**: Native integration with NVIDIA NIM for high-accuracy re-docking of top hits (`ezscreen validate`).
- **3D Visualisation**: Self-contained py3Dmol HTML viewer (`ezscreen view`).
- **Resilience**: Exponential backoff polling, chunking, and state database ensure your runs can survive network drops.

## Other Commands

- `ezscreen status`: See all recent runs and their completion percentage (use `--live` to auto-refresh).
- `ezscreen admet`: Run standalone filtering on any SDF/CSV file.
- `ezscreen clean <run_id>`: Delete artifacts from Kaggle to free up space.

## License
MIT
