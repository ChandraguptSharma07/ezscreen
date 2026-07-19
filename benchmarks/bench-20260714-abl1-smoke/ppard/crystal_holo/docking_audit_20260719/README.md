# PPARD UniDock Docking Audit

Standalone audit workspace for the 2026-07-19 PPARD handoff.

Current first-pass scope:
- prepare the 2ZNP cocrystal ligand from `benchmarks/external_sources/dude_ppard/crystal_ligand.mol2`
- reuse the current PPARD receptor and box
- build a tiny active / hard-decoy / neutral-decoy panel
- run the smallest UniDock redocking job through Kaggle when approved
- record redocking RMSD, pose sanity, and seed metrics under this directory

Primary commands:

```powershell
python scripts\prepare_audit_inputs.py
python scripts\submit_redocking_kaggle.py --ablation-id current_box_balance
```

Key artifacts:
- `redocking/`
- `audit_panel/audit_panel.csv`
- `audit_panel/audit_panel.smi`
- `pose_validity/current_raw_top20_pose_validity.csv`
- `docking_ablation_metrics.csv`
- `docking_ablation_summary.json`
- `ppard_unidock_docking_audit.ipynb`
