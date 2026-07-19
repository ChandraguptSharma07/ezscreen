from __future__ import annotations

import json
import textwrap
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[1]
NOTEBOOK = AUDIT / "ppard_unidock_docking_audit.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def main() -> None:
    nb = {
        "cells": [
            md(
                """
                # PPARD UniDock Docking Audit

                Objective: test whether the current PPARD UniDock setup is credible as docking and pose generation before relying on downstream rescoring.

                Done for the first pass:
                - prepare the 2ZNP cocrystal ligand
                - check current box geometry against the crystal ligand
                - build a 50-ligand audit panel
                - submit/evaluate the smallest UniDock redocking run when Kaggle access is available
                """
            ),
            md(
                """
                ## Setup

                This notebook is a thin driver around scripts in the same audit directory. It does not modify core SwiftScreen code.
                """
            ),
            code(
                """
                from pathlib import Path
                import json

                AUDIT = Path.cwd()
                if AUDIT.name != "docking_audit_20260719":
                    AUDIT = Path("benchmarks/bench-20260714-abl1-smoke/ppard/crystal_holo/docking_audit_20260719").resolve()
                AUDIT
                """
            ),
            md(
                """
                ## Prepare Inputs

                This extracts the cocrystal ligand, writes SDF/PDBQT files, snapshots box geometry, writes current top-20 pose diagnostics, and creates the small audit panel.
                """
            ),
            code(
                """
                import subprocess, sys

                subprocess.run([sys.executable, str(AUDIT / "scripts" / "prepare_audit_inputs.py")], check=True)
                """
            ),
            code(
                """
                print((AUDIT / "redocking" / "box_sanity.json").read_text())
                print((AUDIT / "docking_ablation_metrics.csv").read_text())
                """
            ),
            md(
                """
                ## Smallest UniDock Redocking Run

                Local UniDock is not available on this workstation. Run this cell only when a Kaggle upload/submission is allowed for the tiny cocrystal-ligand job.
                """
            ),
            code(
                """
                # subprocess.run([
                #     sys.executable,
                #     str(AUDIT / "scripts" / "submit_redocking_kaggle.py"),
                #     "--ablation-id", "current_box_balance",
                #     "--search-mode", "balance",
                # ], check=True)
                """
            ),
            md(
                """
                ## Review Redocking Result

                After the Kaggle redock finishes, this cell shows the latest RMSD and pose sanity diagnostics.
                """
            ),
            code(
                """
                latest = AUDIT / "redocking" / "redocking_rmsd_latest.json"
                if latest.exists():
                    print(latest.read_text())
                else:
                    print((AUDIT / "redocking" / "redocking_status.json").read_text())
                """
            ),
            md(
                """
                ## Next Ablations

                Add one variable at a time: box padding/center, search mode, num modes, seed stability, receptor prep variant, protonation/charge handling, and ligand conformer/prep choices.
                """
            ),
            code(
                """
                import pandas as pd

                pd.read_csv(AUDIT / "audit_panel" / "audit_panel.csv").groupby(["audit_reason", "label"]).size()
                """
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()


