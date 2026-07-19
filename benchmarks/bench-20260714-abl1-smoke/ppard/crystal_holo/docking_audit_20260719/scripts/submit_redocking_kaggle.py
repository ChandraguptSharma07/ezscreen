from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[1]
REPO = AUDIT.parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))

BOX_CENTER = [15.656, 2.308, 39.483]
BOX_SIZE = [21.337, 22.566, 22.752]
RECEPTOR_PDBQT = AUDIT / "inputs" / "receptor_fixed.pdbqt"
REDOCK_LIGAND_PDBQT = AUDIT / "redocking" / "cocrystal_2znp.pdbqt"


def _load_primary_kaggle() -> tuple[str, Path]:
    from ezscreen.auth import get_kaggle_json_path, load_credentials

    path = get_kaggle_json_path(load_credentials())
    if path is None or not path.exists():
        raise RuntimeError("No primary Kaggle credential path configured")
    data = json.loads(path.read_text(encoding="utf-8"))
    os.environ["KAGGLE_USERNAME"] = data["username"]
    os.environ["KAGGLE_KEY"] = data["key"]
    return data["username"], path


def submit(ablation_id: str, search_mode: str, accelerator: str) -> dict[str, object]:
    from ezscreen.backends.kaggle.runner import render_vina_notebook, run_screening_job
    from scripts.evaluate_redocking import evaluate

    username, key_path = _load_primary_kaggle()
    run_id = f"ezs-ppard-redock-{ablation_id}".lower().replace("_", "-")
    work = AUDIT / "redocking" / f"kaggle_{ablation_id}"
    work.mkdir(parents=True, exist_ok=True)
    notebook = work / "redock_unidock.ipynb"
    notebook.write_text(
        render_vina_notebook(
            run_id=run_id,
            shard_filenames=["cocrystal_2znp.pdbqt"],
            notebook_index=0,
            total_notebooks=1,
            box_center=BOX_CENTER,
            box_size=BOX_SIZE,
            search_mode=search_mode,
            ph=7.4,
            prep_on_kaggle=False,
            accelerator=accelerator,
            results_cfg={"poses_returned": 25, "prolif_rescore_enabled": False},
            engine="unidock",
            scoring="vina",
            receptor_filename="receptor.pdbqt",
        ),
        encoding="utf-8",
    )
    started = time.time()
    result = run_screening_job(
        run_id=run_id,
        receptor_pdbqt=RECEPTOR_PDBQT,
        shard_paths=[REDOCK_LIGAND_PDBQT],
        notebook_path=notebook,
        username=username,
        work_dir=work,
        retry_limit=1,
        accelerator=accelerator,
        engine="unidock",
    )
    runtime = round(time.time() - started, 3)
    summary = {
        "status": result.get("status"),
        "ablation_id": ablation_id,
        "run_id": run_id,
        "username": username,
        "kaggle_key_path": str(key_path),
        "work_dir": str(work),
        "runtime_seconds": runtime,
        "result": {k: str(v) if isinstance(v, Path) else v for k, v in result.items()},
    }
    (work / "submission_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if result.get("status") == "complete" and result.get("output_dir"):
        output_dir = Path(str(result["output_dir"]))
        archived = AUDIT / "redocking" / f"output_{ablation_id}"
        if archived.exists():
            shutil.rmtree(archived)
        shutil.copytree(output_dir, archived)
        eval_result = evaluate(archived, ablation_id, runtime)
        summary["evaluation"] = eval_result
        (work / "submission_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-id", default="current_box_balance")
    parser.add_argument("--search-mode", default="balance", choices=["fast", "balance", "detail"])
    parser.add_argument("--accelerator", default="nvidiaTeslaT4")
    parser.add_argument("--receptor-pdbqt", type=Path, default=RECEPTOR_PDBQT)
    parser.add_argument("--redock-ligand-pdbqt", type=Path, default=REDOCK_LIGAND_PDBQT)
    args = parser.parse_args()
    print(json.dumps(submit(args.ablation_id, args.search_mode, args.accelerator), indent=2))


if __name__ == "__main__":
    main()
