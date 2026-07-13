from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ezscreen import auth
from ezscreen.errors import KaggleAuthError


@dataclass(frozen=True)
class WorkerKaggleCredentials:
    username: str
    key: str
    source: str
    path: Path | None = None


def resolve_worker_kaggle_credentials(
    *,
    team_account_name: str | None = None,
    kaggle_username: str | None = None,
    kaggle_key: str | None = None,
    use_local: bool = False,
    creds: dict[str, Any] | None = None,
) -> WorkerKaggleCredentials:
    """Resolve credentials for GitHub worker secrets without mutating local files."""
    creds = creds or auth.load_credentials()
    env_username = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    if kaggle_key:
        explicit_username = kaggle_username or env_username
        if not explicit_username:
            raise KaggleAuthError(
                "Kaggle username is required when using --kaggle-key."
            )
        return WorkerKaggleCredentials(
            username=explicit_username,
            key=kaggle_key,
            source="explicit",
        )

    if team_account_name:
        team_account_name = team_account_name.strip()
        if not team_account_name:
            raise KaggleAuthError("Saved team account name is empty.")
        accounts = {acct["name"]: acct for acct in auth.list_team_accounts(creds)}
        account = accounts.get(team_account_name)
        if not account:
            raise KaggleAuthError(f"Saved team account not found: {team_account_name}")
        kaggle_json = Path(str(account.get("kaggle_json_path", ""))).expanduser()
        data = auth.validate_kaggle_json(kaggle_json)
        return WorkerKaggleCredentials(
            username=str(account.get("username") or data["username"]),
            key=data["key"],
            source=f"saved team account: {team_account_name}",
            path=kaggle_json,
        )

    if use_local:
        kaggle_json = auth.get_kaggle_json_path(creds)
        if kaggle_json is None:
            raise KaggleAuthError("No local kaggle_json_path is saved in ezscreen credentials.")
        data = auth.validate_kaggle_json(kaggle_json)
        username = kaggle_username or data["username"]
        return WorkerKaggleCredentials(
            username=username,
            key=data["key"],
            source="local ezscreen kaggle.json",
            path=kaggle_json,
        )

    if env_key:
        if not env_username:
            raise KaggleAuthError(
                "Kaggle username is required when using KAGGLE_KEY."
            )
        return WorkerKaggleCredentials(
            username=env_username,
            key=env_key,
            source="environment",
        )

    raise KaggleAuthError(
        "Choose a saved team account, use local Kaggle credentials, or provide a Kaggle username and key for this worker."
    )



def local_kaggle_summary(creds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return non-secret local Kaggle credential metadata for status/UI messages."""
    creds = creds or auth.load_credentials()
    path = auth.get_kaggle_json_path(creds)
    if path is None or not path.exists():
        return {"configured": False, "path": str(path) if path else ""}
    try:
        data = auth.validate_kaggle_json(path)
    except Exception:
        return {"configured": False, "path": str(path)}
    return {"configured": True, "path": str(path), "username": data.get("username", "")}
