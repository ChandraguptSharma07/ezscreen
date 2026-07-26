import csv
import json
from pathlib import Path

from ezscreen.benchmark.smina_output import recover_smina_directory


def test_recover_smina_directory_uses_log_affinity_and_pose_models(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    logs = output / "logs"
    poses = output / "poses"
    logs.mkdir(parents=True)
    poses.mkdir()
    (logs / "dock_0000_vina.log").write_text(
        "mode | affinity\n"
        "1       -7.0       0.000      0.000\n"
        "2       -6.8       1.585      4.871\n",
        encoding="utf-8",
    )
    (poses / "0000_vina.pdbqt").write_text(
        "MODEL        1\nENDMDL\nMODEL        2\nENDMDL\n",
        encoding="utf-8",
    )

    result = recover_smina_directory(output, expected_attempts=1)

    assert result["gate_pass"] is True
    assert result["coverage"] == 1.0
    rows = list(csv.DictReader((output / "scores_recovered.csv").open()))
    assert rows[0]["score"] == "-7.0"
    assert rows[0]["pose_count"] == "2"
    audit = json.loads((output / "smina_output_audit.json").read_text())
    assert audit["parser"].startswith("smina mode-table")
