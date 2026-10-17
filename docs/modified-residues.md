# Optional modified-residue preservation

Experimental expert option. Software regression tests are not chemical validation of LLP or any other modified residue. No validated LLP preset ships yet.

## Scope

Version 1 supports explicitly selected **internal peptide residues with unchanged N/C backbone linkage**, including ATOM- or HETATM-labeled input. Terminal residues, insertion codes, modified backbones, automatic receptor protonation and wizard ensembles are not supported in this mode. Ordinary preparation remains unchanged when no policy is supplied.

In wizard step 2, enter a reviewed policy JSON path under **Preserve modified residues**. The confirmation shows its path and SHA256; changing the file requires reconfirmation. Policy selections override ordinary heterogen removal for those exact residues, even when their heterogen checkbox is not selected. Other waters/cofactors retain their existing selection behavior.

Python API:

```python
result = prep_receptor(
    pdb_path, chains=["A"], output_dir=output_dir,
    keep_hetero=selected_waters,
    receptor_isomers="off",
    modified_residue_policy=Path("reviewed_policy.json"),
)
```

This API does not choose a compute backend. Project benchmark policy requires invoking preparation inside a Kaggle GPU notebook, not locally. Standalone source bundles must include both `ezscreen/prep/receptor.py` and `ezscreen/prep/modified_residues.py`. Full-screen shards should consume the already verified PDBQT bytes; they need not regenerate templates.

## Policy schema

The JSON object has `version: 1`, a `templates` object and a nonempty `residues` list. Each template is a Meeko definition with exactly `smiles`, `atom_name` and `link_labels`. Use namespaced keys starting with `EZ_`; built-in templates cannot be overridden. Include only templates explicitly assigned in the policy.

Each selected residue has:

- `resname`, `chain`, integer `resseq`: exact original PDB identity; insertion codes unsupported.
- `template`: key in `templates`.
- `formal_charge`: explicit integer charge of the embedded template, not its partial-charge sum.
- `state_rationale`: nonempty explanation/source for the chemical state. This records expert reasoning; it does not validate that reasoning automatically.
- `links`: two objects, each with `atom`, partner `chain`, partner `resseq`, and `partner_atom`. The selected N must link to a partner C, and the selected C to a partner N.

For LLP A680 in 1A8I, the expected neighbors are A679/C at LLP/N and A681/N at LLP/C. This identity information alone is **not** a complete ready-to-run policy: the linked template and protonation rationale still need review.

Template generation can reuse Meeko's `build_linked_CCs` and JSON export on Kaggle. CCD chemistry and Meeko's canonical acidic-group deprotonation are starting assumptions, not protein-environment protonation evidence. Freeze the source CCD, template, tool versions and rationale before docking outcomes. Do not select the state by whichever EF is largest.

## Integrity gates

Protected residues are exempted from PDBFixer's substitution and heterogen deletion, and never appended twice. Meeko receives explicit template assignments without permissive `-a` or Open Babel fallback. Missing/unsupported chemistry fails preparation rather than silently degrading.

The gate checks exact selected heavy-atom names and residue identity before/after repair and in PDBQT, coordinates within 0.002 A serialization tolerance, finite PDBQT charges and nonempty atom types. It checks the materialized Meeko molecular graph against the frozen template and verifies exactly the declared inter-residue links using Meeko's bond-perception function. Unexpected additional links fail. Input hydrogens, if present, must use template names; prefer the same heavy-atom input convention as the benchmark protocol.

Artifacts: `modified_residue_policy.json` (including original source SHA256), `modified_residue_templates.json`, `receptor_modified_polymer.json`, `receptor_modified_retention.json`. Successful prep reports add `modified_residue_preservation`; existing report/variant fields and docking engine/scoring defaults are unchanged. On failure, inspect intermediate artifacts and the exception; do not treat an existing PDBQT as a successful run.

## Verification

```bash
.venv/bin/python -m pytest tests/test_modified_residues.py tests/test_receptor.py tests/test_kaggle_contracts.py tests/test_engines.py tests/test_tui_smoke.py tests/test_redock.py --no-cov -q
```

These tests mock receptor preparation and include synthetic chemistry; they do not perform local docking or validate LLP. Next experimental checkpoint: a separately frozen Kaggle LLP preparation gate, then native-ligand redocking and remaining structural/water checks before a GPB full screen.
