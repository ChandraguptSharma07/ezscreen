# ezscreen

**GPU-accelerated virtual screening powered by Kaggle GPUs.**

`ezscreen` is a CLI tool that runs molecular docking campaigns on Kaggle's free GPUs. It handles receptor preparation, ligand prep, ADMET filtering, Kaggle kernel submission, result download, and hit visualisation — all from a single interactive command.

## Prerequisites

- Python 3.11+
- A [Kaggle account](https://www.kaggle.com/) with GPU quota and an API token (`kaggle.json`)
- *(Optional)* NVIDIA NIM API key for Stage 2 validation with DiffDock-L

## Installation

```bash
pip install ezscreen
```

### Optional: scrubber for enhanced ligand prep

[forlilab/scrubber](https://github.com/forlilab/scrubber) provides tautomer enumeration and pH-driven protonation. Not on PyPI — install separately:

```bash
pip install git+https://github.com/forlilab/scrubber.git
```

Without it, `ezscreen` falls back to RDKit-only preparation (still fully functional).

## Setup

```bash
ezscreen auth
```

Prompts for your Kaggle `kaggle.json` path and optionally an NVIDIA NIM API key.

## Quickstart

```bash
ezscreen run
```

Interactive wizard guides you through:

1. Receptor — PDB ID (auto-downloaded) or local `.pdb` file
2. Chain selection
3. Binding site — co-crystal ligand, residue list, P2Rank prediction, or blind
4. Ligand file — `.smi`, `.smiles`, or `.sdf`
5. ADMET filtering (Lipinski, Veber, PAINS, Brenk)
6. Search depth — Fast / Balanced / Thorough
7. Submit to Kaggle

When the run completes, results are downloaded automatically and displayed:

```bash
ezscreen view ezs-<run_id>
```

## Commands

| Command | Description |
|---|---|
| `ezscreen run` | Interactive screening wizard |
| `ezscreen auth` | Configure Kaggle and NIM credentials |
| `ezscreen view <run_id>` | Show results table + open 3D viewer |
| `ezscreen download <run_id>` | Re-download results for a completed run |
| `ezscreen status` | List all runs with status (`--live` to auto-refresh) |
| `ezscreen resume <run_id>` | Resume an interrupted run |
| `ezscreen admet <file>` | Standalone ADMET filtering on any SDF/SMI file |
| `ezscreen validate <receptor> <hits>` | Stage 2 re-docking via NVIDIA NIM DiffDock-L |
| `ezscreen clean <run_id>` | Delete Kaggle dataset and kernel artifacts |

## Features

- **UniDock GPU docking** — builds UniDock from source on Kaggle to match the installed CUDA toolkit, avoiding pre-built binary ABI mismatches
- **Compound identity** — results include the original name and SMILES from your input file alongside docking scores
- **Artifact filtering** — unphysical scores (< −15 kcal/mol) from GPU batching edge cases are automatically removed
- **Resilient download** — retries with exponential backoff; recovers scores locally from docked PDBQTs if the Kaggle download fails mid-transfer
- **ADMET filtering** — Lipinski, Veber, PAINS, Brenk rules applied locally before submitting to reduce wasted compute
- **3D viewer** — self-contained py3Dmol HTML viewer for top poses
- **DiffDock-L validation** — native NVIDIA NIM integration for high-accuracy re-docking of top hits

## License

Apache-2.0
