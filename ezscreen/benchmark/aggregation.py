from __future__ import annotations

import math
from dataclasses import dataclass

from ezscreen.results.variants import source_compound_id


@dataclass
class SourceAggregation:
    records: list[dict]
    counts: dict[str, int]


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate_source_compounds(
    rows: list[dict],
    *,
    score_key: str = '_ranking_score',
    name_key: str = 'name',
) -> SourceAggregation:
    grouped: dict[str, list[dict]] = {}
    unmatched_rows = 0
    for index, row in enumerate(rows):
        source_id = source_compound_id(row, name_key=name_key)
        if not source_id:
            source_id = f'__unmatched_source_{index:09d}'
            unmatched_rows += 1
        grouped.setdefault(source_id, []).append(row)

    valid_scores = [
        score
        for row in rows
        if row.get('_scored', True) and (score := _finite(row.get(score_key))) is not None
    ]
    worst_score = math.nextafter(max(valid_scores), math.inf) if valid_scores else 0.0
    records: list[dict] = []
    scored_count = failed_count = filtered_count = 0
    for source_id, variants in grouped.items():
        scored_variants = [
            row
            for row in variants
            if row.get('_scored', True) and _finite(row.get(score_key)) is not None
        ]
        if scored_variants:
            kept = dict(
                min(scored_variants, key=lambda row: float(row[score_key]))
            )
            kept['_scored'] = True
            kept['_failed'] = False
            scored_count += 1
        else:
            kept = dict(variants[0])
            kept[score_key] = worst_score
            kept['_scored'] = False
            kept['_failed'] = True
            failed_count += 1
        filtered = any(bool(row.get('_filtered')) for row in variants)
        if filtered and not scored_variants:
            filtered_count += 1
        kept['_filtered'] = filtered and not bool(scored_variants)
        kept['source_compound_id'] = source_id
        kept['variant_count'] = len(variants)
        kept['receptor_count'] = len(
            {str(row.get('receptor_variant', '')) for row in variants}
        )
        kept['conformer_count'] = len(
            {str(row.get('conformer_id', '')) for row in variants}
        )
        records.append(kept)
    records.sort(key=lambda row: float(row[score_key]))
    return SourceAggregation(
        records=records,
        counts={
            'input_rows': len(rows),
            'after_aggregation': len(records),
            'scored': scored_count,
            'failed': failed_count,
            'filtered': filtered_count,
            'unmatched': unmatched_rows,
            'variant_rows_collapsed': len(rows) - len(records),
        },
    )
