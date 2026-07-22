from __future__ import annotations

import json
from pathlib import Path

import ezscreen.checkpoint as checkpoint
from ezscreen.benchmark.kaggle_planning import (
    build_account_assignments,
    plan_kaggle_shards,
)
from ezscreen.benchmark.workspace import create_cell_scaffold
from ezscreen.commands import benchmark as benchmark_cmd


def test_adaptive_kaggle_shard_plan_does_not_use_all_slots_for_small_panel():
    plan = plan_kaggle_shards(1_000, 6)

    assert plan.total_notebooks == 2
    assert plan.notebooks_per_account == (1, 1, 0, 0, 0, 0)


def test_adaptive_kaggle_shard_plan_uses_full_capacity_above_20k():
    plan = plan_kaggle_shards(20_001, 6)

    assert plan.total_notebooks == 12
    assert plan.notebooks_per_account == (2, 2, 2, 2, 2, 2)


def test_adaptive_kaggle_shard_plan_uses_8_until_20k():
    plan = plan_kaggle_shards(20_000, 6)

    assert plan.total_notebooks == 8
    assert plan.notebooks_per_account == (2, 2, 1, 1, 1, 1)


def test_build_account_assignments_preserves_zero_exclusions():
    accounts = [{"name": f"acct{i}", "kaggle_json_path": f"k{i}.json"} for i in range(3)]

    assignments = build_account_assignments(accounts, (1, 0, 2))

    assert [a["shard_count"] for a in assignments] == [1, 0, 2]
    assert assignments[1]["account"]["name"] == "acct1"


def test_multi_account_benchmark_submission_creates_recent_run_checkpoint(
    monkeypatch,
    tmp_path: Path,
):
    db_file = tmp_path / "checkpoints.db"
    monkeypatch.setattr(checkpoint, "DB_DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "DB_PATH", db_file)
    checkpoint.init_db()

    cell = create_cell_scaffold("bench", "abl1", "crystal_holo", "run-001", root=tmp_path)
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("RECEPTOR\n", encoding="utf-8")
    ligands = tmp_path / "ligands.smi"
    ligands.write_text(
        "\n".join(f"CC ligand_{idx:04d}" for idx in range(1_000)),
        encoding="utf-8",
    )
    (cell / "inputs" / "actives.smi").write_text("CC ligand_0000\n", encoding="utf-8")

    accounts = [
        {"name": f"acct{i}", "kaggle_json_path": str(tmp_path / f"k{i}.json")}
        for i in range(6)
    ]
    captured: dict[str, object] = {}

    output_dir = tmp_path / "remote_output"
    output_dir.mkdir()
    (output_dir / "scores.csv").write_text(
        "ligand,name,smiles,docking_score\n"
        "lig_0001,ligand_0000,CC,-8.0\n"
        "lig_0002,decoy,CCC,-6.0\n",
        encoding="utf-8",
    )
    (output_dir / "results_meta.json").write_text("{}", encoding="utf-8")

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "complete", "output_dir": str(output_dir)}

    monkeypatch.setattr("ezscreen.auth.get_all_kaggle_accounts", lambda: accounts)
    monkeypatch.setattr(
        "ezscreen.config.load",
        lambda: {
            "run": {"default_ph": 7.4},
            "prep": {},
            "results": {},
        },
    )
    monkeypatch.setattr("ezscreen.backends.kaggle.runner.run_multi_account_screening", fake_runner)
    monkeypatch.setattr(
        benchmark_cmd,
        "run_cell_benchmark",
        lambda *args, **kwargs: {"metrics_summary": cell / "metrics_summary.json"},
    )

    result = benchmark_cmd.submit_multi_account_kaggle_cell(
        cell,
        receptor_pdbqt=receptor,
        ligands=ligands,
        box_center="1,2,3",
        box_size="20,20,20",
        run_id="ezs-test-multiacct",
        min_accounts=6,
    )

    assert result["status"] == "complete"
    assert len(captured["shard_paths"]) == 2
    assert [a["shard_count"] for a in captured["account_assignments"]] == [1, 1, 0, 0, 0, 0]

    run = checkpoint.get_run("ezs-test-multiacct")
    assert run is not None
    assert run["status"] == "complete"
    assert run["total_compounds"] == 1_000

    config_json = json.loads(run["config_json"])
    assert config_json["source"] == "benchmark_multi_account_kaggle"
    assert config_json["shard_plan"]["total_notebooks"] == 2
