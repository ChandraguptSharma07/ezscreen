from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KaggleShardPlan:
    ligand_count: int
    account_count: int
    total_notebooks: int
    notebooks_per_account: tuple[int, ...]
    rationale: str


def recommend_kaggle_notebook_count(
    ligand_count: int,
    account_count: int,
    *,
    max_notebooks_per_account: int = 2,
) -> int:
    """Choose notebook count from work size, not from available quota alone."""
    if account_count <= 0:
        raise ValueError("account_count must be positive")
    if max_notebooks_per_account <= 0:
        raise ValueError("max_notebooks_per_account must be positive")

    ligand_count = max(0, int(ligand_count))
    capacity = account_count * max_notebooks_per_account
    if ligand_count <= 0:
        desired = 1
    elif ligand_count <= 1_500:
        desired = 2
    elif ligand_count <= 6_000:
        desired = 4
    elif ligand_count <= 15_000:
        desired = 6
    elif ligand_count <= 20_000:
        desired = 8
    else:
        desired = capacity

    return max(1, min(desired, capacity))


def distribute_notebooks_across_accounts(
    total_notebooks: int,
    account_count: int,
    *,
    max_notebooks_per_account: int = 2,
) -> tuple[int, ...]:
    """Fill one notebook per account before assigning second notebooks."""
    if account_count <= 0:
        raise ValueError("account_count must be positive")
    if max_notebooks_per_account <= 0:
        raise ValueError("max_notebooks_per_account must be positive")
    capacity = account_count * max_notebooks_per_account
    if total_notebooks < 1 or total_notebooks > capacity:
        raise ValueError("total_notebooks must be between 1 and account capacity")

    counts = [0] * account_count
    remaining = total_notebooks
    for _round in range(max_notebooks_per_account):
        for idx in range(account_count):
            if remaining == 0:
                return tuple(counts)
            counts[idx] += 1
            remaining -= 1
    return tuple(counts)


def plan_kaggle_shards(
    ligand_count: int,
    account_count: int,
    *,
    max_notebooks_per_account: int = 2,
) -> KaggleShardPlan:
    total = recommend_kaggle_notebook_count(
        ligand_count,
        account_count,
        max_notebooks_per_account=max_notebooks_per_account,
    )
    counts = distribute_notebooks_across_accounts(
        total,
        account_count,
        max_notebooks_per_account=max_notebooks_per_account,
    )
    per_notebook = math.ceil(max(0, int(ligand_count)) / total) if total else 0
    rationale = (
        f"{ligand_count} ligand(s) -> {total} notebook(s), about "
        f"{per_notebook} ligand(s)/notebook; capped at "
        f"{max_notebooks_per_account} notebook(s)/account"
    )
    return KaggleShardPlan(
        ligand_count=max(0, int(ligand_count)),
        account_count=account_count,
        total_notebooks=total,
        notebooks_per_account=counts,
        rationale=rationale,
    )


def build_account_assignments(
    accounts: list[dict[str, Any]],
    notebooks_per_account: tuple[int, ...],
) -> list[dict[str, Any]]:
    if len(accounts) != len(notebooks_per_account):
        raise ValueError("accounts and notebooks_per_account length mismatch")
    return [
        {"account": account, "shard_count": notebook_count}
        for account, notebook_count in zip(accounts, notebooks_per_account)
    ]
