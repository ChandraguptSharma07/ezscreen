from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def merge_shard_results(shard_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    fieldnames: list[str] = []

    for d in shard_dirs:
        sf = d / "scores.csv"
        if not sf.exists():
            continue
        with sf.open() as f:
            reader = csv.DictReader(f)
            if not fieldnames and reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            all_rows.extend(reader)

    id_col = fieldnames[0] if fieldnames else None
    score_col = next(
        (h for h in fieldnames if "score" in h.lower() or "affinity" in h.lower()),
        fieldnames[-1] if fieldnames else None,
    )

    def _score(row: dict) -> float:
        try:
            return float(row.get(score_col, 0)) if score_col else 0.0
        except (ValueError, TypeError):
            return 0.0

    best: dict[str, dict] = {}
    for row in all_rows:
        key = row.get(id_col, "") if id_col else str(id(row))
        if key not in best or _score(row) < _score(best[key]):
            best[key] = row

    deduped = sorted(best.values(), key=_score)

    scores_out = output_dir / "scores.csv"
    if deduped and fieldnames:
        with scores_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(deduped)

    poses_out = output_dir / "poses.sdf"
    with poses_out.open("w") as f:
        for d in shard_dirs:
            pf = d / "poses.sdf"
            if pf.exists():
                content = pf.read_text()
                f.write(content)
                if content and not content.endswith("\n"):
                    f.write("\n")

    failed_out = output_dir / "failed_prep.sdf"
    has_failures = False
    with failed_out.open("w") as f:
        for d in shard_dirs:
            ff = d / "failed_prep.sdf"
            if ff.exists() and ff.stat().st_size > 0:
                f.write(ff.read_text())
                has_failures = True

    return {
        "scores_csv": scores_out,
        "poses_sdf": poses_out,
        "failed_prep_sdf": failed_out if has_failures else None,
        "total_hits": len(deduped),
        "score_col": score_col,
    }
