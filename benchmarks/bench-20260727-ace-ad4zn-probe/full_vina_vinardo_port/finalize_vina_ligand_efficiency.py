"""Create an exploratory ACE Vina ligand-efficiency secondary ranking."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

from finalize_full_vina_vinardo import (
    PUBLISHED,
    bayes_ef_max_proxy,
    bayes_ef_proxy,
    enrichment,
    pr_metrics,
)

from ezscreen.benchmark.metrics import compute_metrics

HERE = Path(__file__).resolve().parent
PRIMARY = HERE / "final_results" / "vina" / "ranking_1846.csv"
OUTPUT = HERE / "final_results" / "vina_ligand_efficiency"
AD4_WORK = HERE.parent / "full_ad4_port" / "work"
TOTAL, ACTIVES, DECOYS = 1846, 49, 1797


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heavy_atom_count(payload: bytes) -> int:
    count = 0
    for line in payload.decode(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")) and line.split()[-1] not in {
            "H",
            "HD",
        }:
            count += 1
    if count <= 0:
        raise RuntimeError("pose has no heavy atoms")
    return count


def pose_counts(work: Path, pose_zip_name: str) -> tuple[dict[int, int], dict[str, str]]:
    counts: dict[int, int] = {}
    hashes = {}
    for bundle in sorted(work.glob("*/output/output.zip")):
        hashes[bundle.parents[1].name] = sha256(bundle)
        with zipfile.ZipFile(bundle) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read(pose_zip_name))) as poses:
                for name in poses.namelist():
                    index = int(Path(name).stem.split("_")[0])
                    if index in counts:
                        raise RuntimeError(f"duplicate pose index: {index}")
                    counts[index] = heavy_atom_count(poses.read(name))
    if set(counts) != set(range(TOTAL)):
        raise RuntimeError("pose heavy-atom accounting mismatch")
    return counts, hashes


def main() -> None:
    with PRIMARY.open(newline="", encoding="utf-8") as handle:
        primary = list(csv.DictReader(handle))
    if len(primary) != TOTAL or any(row["status"] != "scored" for row in primary):
        raise RuntimeError("primary Vina ranking is not complete")
    if [int(row["rank"]) for row in primary] != list(range(1, TOTAL + 1)):
        raise RuntimeError("primary rank accounting mismatch")

    vina_counts, raw_hashes = pose_counts(HERE / "work", "docked_vina.zip")
    ad4_counts, _ad4_hashes = pose_counts(AD4_WORK, "docked_pdbqt.zip")
    if vina_counts != ad4_counts:
        raise RuntimeError("Vina and AD4 heavy-atom counts differ")

    ranking = []
    for row in primary:
        index = int(row["sample_index"])
        raw_score = float(row["score"])
        heavy_atoms = vina_counts[index]
        ligand_efficiency = raw_score / heavy_atoms
        if not math.isfinite(ligand_efficiency):
            raise RuntimeError(f"nonfinite ligand efficiency: {index}")
        ranking.append(
            {
                "sample_index": index,
                "source_record_key": row["source_record_key"],
                "source_id": row["source_id"],
                "label": int(row["label"]),
                "raw_score": raw_score,
                "heavy_atoms": heavy_atoms,
                "ligand_efficiency": ligand_efficiency,
                "status": row["status"],
                "prep_method": row["prep_method"],
                "failure_reason": row["failure_reason"],
            }
        )
    ranking.sort(key=lambda row: (row["ligand_efficiency"], row["sample_index"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    scores = [row["ligand_efficiency"] for row in ranking]
    labels = [row["label"] for row in ranking]
    metric = compute_metrics(scores, labels)
    scored = [{"score": row["ligand_efficiency"], "label": row["label"]} for row in ranking]
    metrics = {
        "protocol": "Exploratory post-hoc ligand-efficiency reranking of the UniDock Vina accelerated port",
        "formula": "raw UniDock Vina score / non-hydrogen PDBQT atom count; lower is better",
        "primary_status": "secondary analysis; primary raw-score Vina ranking remains unchanged",
        "selection_caveat": "This standard label-free formula was selected after inspecting ACE benchmark metrics; improvement requires held-out-target validation and is not a paper reproduction claim.",
        "total": TOTAL,
        "actives": ACTIVES,
        "decoys": DECOYS,
        "scored": TOTAL,
        "coverage": 1.0,
        "failures": 0,
        "ef1": metric.ef1,
        "ef5": metric.ef5,
        "ef10": enrichment(labels, 0.10),
        "ef20": metric.ef20,
        "roc_auc": metric.auc_roc,
        "bedroc20": metric.bedroc,
        **pr_metrics(scores, labels),
        "bayes_ef_decoy_proxy_1pct": bayes_ef_proxy(scored, 0.01),
        "bayes_ef_decoy_proxy_5pct": bayes_ef_proxy(scored, 0.05),
        "bayes_ef_max_decoy_proxy": bayes_ef_max_proxy(scored),
        "bayes_ef_caveat": "DUD2 decoys are not random compounds; values are decoy-distribution proxies, not canonical Bayes EF.",
        "published_context_only": PUBLISHED["vina"],
        "published_delta_context_only": {
            "roc_auc": round(metric.auc_roc - PUBLISHED["vina"]["roc_auc"], 6),
            "bedroc20": round(metric.bedroc - PUBLISHED["vina"]["bedroc20"], 6),
        },
        "primary_raw_vina_context": {"roc_auc": 0.414, "bedroc20": 0.1565},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "sample_index",
        "source_record_key",
        "source_id",
        "label",
        "raw_score",
        "heavy_atoms",
        "ligand_efficiency",
        "status",
        "prep_method",
        "failure_reason",
    ]
    with (OUTPUT / "ranking_1846.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranking)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUTPUT / "provenance.json").write_text(
        json.dumps(
            {
                "primary_ranking": str(PRIMARY),
                "primary_ranking_sha256": sha256(PRIMARY),
                "raw_output_bundle_sha256": raw_hashes,
                "heavy_atom_count_crosscheck": "1,846/1,846 counts match Vina and AD4 output poses",
                "formula_uses_labels": False,
                "selection_was_post_hoc": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
