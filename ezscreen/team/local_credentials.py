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


_GITHUB_TOKEN_NAMESPACE = "team_runner"
_GITHUB_TOKEN_KEY = "github_dispatch_tokens"


def resolve_worker_kaggle_credentials(
    *,
    kaggle_username: str | None = None,
    kaggle_key: str | None = None,
    use_local: bool = True,
    creds: dict[str, Any] | None = None,
) -> WorkerKaggleCredentials:
    """Resolve credentials for GitHub worker secrets without mutating local files."""
    env_username = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    explicit_key = kaggle_key or env_key
    explicit_username = kaggle_username or env_username
    if explicit_key:
        if not explicit_username:
            raise KaggleAuthError(
                "Kaggle username is required when using --kaggle-key or KAGGLE_KEY."
            )
        return WorkerKaggleCredentials(
            username=explicit_username,
            key=explicit_key,
            source="explicit" if kaggle_key else "environment",
        )

    if not use_local:
        raise KaggleAuthError(
            "No Kaggle key provided. Re-run with --use-local-kaggle or pass --kaggle-key."
        )

    creds = creds or auth.load_credentials()
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


def load_github_dispatch_token(repo: str, *, creds: dict[str, Any] | None = None) -> str:
    creds = creds or auth.load_credentials()
    team_runner = creds.get(_GITHUB_TOKEN_NAMESPACE, {})
    tokens = team_runner.get(_GITHUB_TOKEN_KEY, {})
    return str(tokens.get(repo, "") or "")


def save_github_dispatch_token(repo: str, token: str, *, creds: dict[str, Any] | None = None) -> None:
    if not repo.strip():
        raise ValueError("repo is required")
    if not token.strip():
        raise ValueError("token is required")
    creds = creds or auth.load_credentials()
    team_runner = dict(creds.get(_GITHUB_TOKEN_NAMESPACE, {}))
    tokens = dict(team_runner.get(_GITHUB_TOKEN_KEY, {}))
    tokens[repo] = token.strip()
    team_runner[_GITHUB_TOKEN_KEY] = tokens
    creds[_GITHUB_TOKEN_NAMESPACE] = team_runner
    auth.save_credentials(creds)


def remove_github_dispatch_token(repo: str, *, creds: dict[str, Any] | None = None) -> None:
    creds = creds or auth.load_credentials()
    team_runner = dict(creds.get(_GITHUB_TOKEN_NAMESPACE, {}))
    tokens = dict(team_runner.get(_GITHUB_TOKEN_KEY, {}))
    if repo in tokens:
        tokens.pop(repo, None)
        team_runner[_GITHUB_TOKEN_KEY] = tokens
        creds[_GITHUB_TOKEN_NAMESPACE] = team_runner
        auth.save_credentials(creds)
