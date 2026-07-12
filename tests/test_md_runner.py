from __future__ import annotations

import csv
import json
import os

from ezscreen.backends.kaggle import md_runner


def _write(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_render_md_notebook_is_valid_json():
    src = md_runner.render_md_notebook(
        run_id="ezs-test",
        ligand_id="lig_00000",
        preset="setup",
        pose_source="unidock",
    )
    nb = json.loads(src)
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "LIGAND_ID = \"lig_00000\"" in joined
    assert "STEPS = 0" in joined
    assert "PARAMETERIZATION_MODE = 'restrained_ligand_geometry'" in joined
    assert "ligand_charge_model': 'zero_charge_lj_only" in joined
    assert "md_summary.json" in joined

def test_resolve_kaggle_credentials_sets_env(tmp_path, monkeypatch):
    kj = tmp_path / "kaggle.json"
    kj.write_text('{"username": "alice", "key": "secret"}')
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    username, path = md_runner._resolve_kaggle_credentials(kj)

    assert username == "alice"
    assert path == kj
    assert os.environ["KAGGLE_USERNAME"] == "alice"
    assert os.environ["KAGGLE_KEY"] == "secret"



def test_resolve_kaggle_credentials_uses_saved_auth_path(tmp_path, monkeypatch):
    from ezscreen import auth

    kj = tmp_path / "kaggle.json"
    kj.write_text('{"username": "saved-user", "key": "secret"}')
    monkeypatch.setattr(auth, "get_kaggle_json_path", lambda creds=None: kj)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    username, path = md_runner._resolve_kaggle_credentials()

    assert username == "saved-user"
    assert path == kj
    assert os.environ["KAGGLE_USERNAME"] == "saved-user"
    assert os.environ["KAGGLE_KEY"] == "secret"

def test_find_pose_sdf_prefers_consensus_source(tmp_path):
    run = tmp_path / "ezs-x"
    u = run / "consensus_unidock" / "output"
    g = run / "consensus_gnina" / "output"
    u.mkdir(parents=True)
    g.mkdir(parents=True)
    (u / "poses.sdf").write_text("unidock pose")
    (g / "poses.sdf").write_text("gnina pose")

    assert md_runner._find_pose_sdf(run, "unidock") == u / "poses.sdf"
    assert md_runner._find_pose_sdf(run, "gnina") == g / "poses.sdf"


def test_run_md_simulation_validates_inputs(tmp_path):
    run = tmp_path / "ezs-x"
    (run / "receptor").mkdir(parents=True)
    (run / "output").mkdir(parents=True)
    (run / "receptor" / "receptor_prep.pdb").write_text("HEADER\nEND\n")
    _write(run / "output" / "scores.csv", ["ligand", "name"], [
        {"ligand": "lig_00000", "name": "A"},
    ])

    result = md_runner.run_md_simulation(
        run_id="ezs-x",
        ligand_id="missing",
        preset="setup",
        run_dir=run,
    )
    assert result["status"] == "failed"
    assert "poses.sdf" in result["error"]




