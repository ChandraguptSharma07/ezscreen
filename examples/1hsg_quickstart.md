# Quickstart — HIV-1 protease (1HSG) against a small drug library

This walkthrough screens 8 known HIV protease inhibitors against the 1HSG crystal structure.
It takes roughly 3-4 minutes from launch to results when using Kaggle's free T4 GPU.

---

## Prerequisites

- `pip install ezscreen` done
- Kaggle account with GPU quota enabled and `kaggle.json` downloaded
- *(Optional)* NVIDIA NIM API key for Stage 2 DiffDock-L validation

---

## Step 1 — Set up credentials

```
ezscreen auth
```

Point it to your `kaggle.json` when prompted. Skip the NIM step if you do not have a key yet.

---

## Step 2 — Launch the TUI

```
ezscreen
```

Press `r` on the home screen to open the Run Wizard, or navigate via the menu.

---

## Step 3 — Run Wizard walkthrough

**Step 1: Receptor**

- Enter `1HSG` as the PDB ID and press Enter
- ezscreen fetches the structure from RCSB and parses the chains
- Select chain `A` (the main protease chain)
- The co-crystal ligand MK1 is detected automatically

**Step 2: Binding site**

- Select `Co-crystal ligand MK1` — this gives the tightest, most accurate search box
- The box coordinates are shown at the bottom of the screen for your records

**Step 3: Ligand library**

- Enter the path to the ligands file:
  ```
  examples/test_ligands.smi
  ```

**Step 4: Options**

- Leave ADMET pre-filter on (default)
- Set search depth to `Balanced`

**Step 5: Summary and submit**

- Review all parameters — receptor, box coordinates, ligand count, depth
- Press `Submit` to upload the dataset and launch the Kaggle kernel

---

## Step 4 — Monitor the run

```
ezscreen status
```

Or press `s` from the TUI home screen. The table auto-refreshes every 30 seconds.
Typical runtime on a free T4: 2-4 minutes for 8 compounds at Balanced depth.

---

## Step 5 — View results

Once the run completes, navigate to Results in the TUI or run:

```
ezscreen view
```

You should see the HIV protease inhibitors ranked by docking score.
Expected top scorer: **Ritonavir** or **Saquinavir** at around -10 to -12 kcal/mol.

---

## What to expect

| Compound | Expected rank |
|---|---|
| Ritonavir | 1–2 |
| Saquinavir | 1–2 |
| Indinavir | 2–4 |
| Darunavir | 2–4 |
| Nelfinavir | 3–5 |
| Imatinib | 6–8 (off-target for HIV-PR) |
| Ibuprofen | 7–8 |
| Acetaminophen | 7–8 |

Ibuprofen and Acetaminophen are included as negative controls — they should score poorly
against HIV protease, confirming the screen is discriminating.

---

## Optional: Stage 2 DiffDock-L validation

If you have a NIM key, press `v` on the results screen to validate the top hits
using DiffDock-L. This runs a physics-informed re-docking and typically takes 30-60
seconds per compound.

---

## Cleaning up

```
ezscreen clean <run-id>
```

This removes the Kaggle dataset and kernel for this run, freeing quota.
The local results in `~/.ezscreen/results/` are preserved.
