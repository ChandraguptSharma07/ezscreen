from __future__ import annotations

import re

# Enumerated variants are tagged "<source>_v1", "<source>_v2", … during prep
# (see prep/ligands.py and the Kaggle prep cell). Stripping that suffix recovers
# the source molecule so its forms can be grouped back together.
_VARIANT_RE = re.compile(r"_v\d+$")


def source_name(name: str) -> str:
    return _VARIANT_RE.sub("", name or "")


def has_variants(rows: list[dict], name_key: str = "name") -> bool:
    """True when at least one source molecule appears as >1 enumerated form."""
    seen: set[str] = set()
    for r in rows:
        nm = r.get(name_key, "") or ""
        if _VARIANT_RE.search(nm):
            return True
        base = source_name(nm)
        if base in seen:
            return True
        seen.add(base)
    return False


def collapse_variants(rows: list[dict], name_key: str = "name") -> list[dict]:
    """Collapse enumerated forms back to one row per source molecule.

    Keeps the first row seen per source — and since scores.csv is written
    best-first by the merger, that is the best-scoring form. Rank order is
    preserved. Each kept row gains a "variant_count" of how many forms collapsed.
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        base = source_name(r.get(name_key, "") or "")
        if base not in best:
            kept = dict(r)
            kept["variant_count"] = 1
            best[base] = kept
            order.append(base)
        else:
            best[base]["variant_count"] += 1
    return [best[b] for b in order]
