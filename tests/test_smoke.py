"""
Smoke tests for the core, non-Kaggle units.
All HTTP / Kaggle API calls are fully mocked.
Run with: pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# results.merger
# ---------------------------------------------------------------------------

class TestMerger:
    def test_merge_basic(self, tmp_path: Path) -> None:
        from ezscreen.results.merger import merge_shard_results

        shard0 = tmp_path / "shard0"
        shard1 = tmp_path / "shard1"
        shard0.mkdir(); shard1.mkdir()

        fields = ["name", "docking_score", "rmsd"]

        _csv(shard0 / "scores.csv", [
            {"name": "mol_A", "docking_score": "-9.2", "rmsd": "1.1"},
            {"name": "mol_B", "docking_score": "-7.5", "rmsd": "1.4"},
        ], fields)
        _csv(shard1 / "scores.csv", [
            {"name": "mol_B", "docking_score": "-8.1", "rmsd": "1.2"},  # better mol_B
            {"name": "mol_C", "docking_score": "-6.0", "rmsd": "2.0"},
        ], fields)

        out = tmp_path / "merged"
        result = merge_shard_results([shard0, shard1], out)

        assert result["total_hits"] == 3          # A, B, C
        assert result["scores_csv"].exists()

        rows = list(csv.DictReader(result["scores_csv"].open()))
        names    = [r["name"] for r in rows]
        # mol_A should be best (-9.2), then merged mol_B (-8.1), then mol_C (-6.0)
        assert names == ["mol_A", "mol_B", "mol_C"]
        assert rows[1]["docking_score"] == "-8.1"  # dedup kept better score

    def test_merge_empty_shards(self, tmp_path: Path) -> None:
        from ezscreen.results.merger import merge_shard_results

        out = tmp_path / "merged"
        result = merge_shard_results([], out)
        assert result["total_hits"] == 0


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------

class TestReport:
    def _dummy_receptor(self) -> dict:
        return {
            "source": "rcsb",
            "pdb_id": "7ZTJ",
            "chains_selected": ["A"],
            "chain_selection_method": "user",
            "residue_count": 320,
            "missing_residues": 4,
            "alternates_resolved": 2,
            "waters_removed": 80,
            "is_alphafold": False,
            "tools": {"pdbfixer": "1.9"},
            "warnings": [{"severity": "medium", "category": "prep", "message": "4 missing residues modelled"}],
        }

    def _dummy_bs(self) -> dict:
        return {
            "method": "reference_ligand",
            "reference_ligand": "LIG",
            "center": [12.0, 5.5, -3.1],
            "size":   [22.0, 22.0, 22.0],
            "volume_angstrom3": 10648.0,
            "warnings": [],
        }

    def _dummy_lig(self) -> dict:
        return {
            "input_source": "enamine_real_50k",
            "input_files": 1,
            "total_input": 50_000,
            "admet_removed": 3_200,
            "admet_breakdown": {"lipinski": 3200},
            "prep_passed": 46_750,
            "prep_failed": 50,
            "prep_failures": {"sanitise": 50},
            "tautomers_enumerated": True,
            "protonation_ph": 7.4,
            "tools": {"meeko": "0.5", "scrubber": "1.0"},
            "warnings": [],
        }

    def test_write_report_files(self, tmp_path: Path) -> None:
        from ezscreen.report import write_report

        paths = write_report(
            run_id="ezs-test01",
            receptor_data=self._dummy_receptor(),
            binding_site_data=self._dummy_bs(),
            ligand_data=self._dummy_lig(),
            output_dir=tmp_path,
        )

        assert paths["json"].exists()
        assert paths["txt"].exists()

        data = json.loads(paths["json"].read_text())
        assert data["run_id"] == "ezs-test01"
        assert data["receptor"]["pdb_id"] == "7ZTJ"
        assert data["ligands"]["total_input"] == 50_000
        assert len(data["warnings"]) == 1          # medium from receptor

        txt = paths["txt"].read_text()
        assert "RECEPTOR" in txt
        assert "BINDING SITE" in txt
        assert "LIGANDS" in txt
        assert "WARNINGS" in txt


# ---------------------------------------------------------------------------
# version_check
# ---------------------------------------------------------------------------

class TestVersionCheck:
    def test_banner_no_update(self) -> None:
        """banner() returns None when version matches current."""
        from ezscreen import version_check, __version__

        # Patch _latest to current version
        version_check._latest = __version__
        version_check._done_evt.set()

        assert version_check.banner() is None

    def test_banner_outdated(self) -> None:
        """banner() returns a string when outdated."""
        from ezscreen import version_check

        version_check._latest = "99.99.99"
        version_check._done_evt.set()

        b = version_check.banner()
        assert b is not None
        assert "99.99.99" in b
        assert "pip install" in b

    def teardown_method(self, _method) -> None:
        """Reset module state between tests."""
        from ezscreen import version_check
        import threading
        version_check._latest = None
        version_check._done_evt = threading.Event()
