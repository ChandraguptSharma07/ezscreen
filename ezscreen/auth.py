from __future__ import annotations

import json
import os
import stat
import tomllib
from pathlib import Path
from typing import Any

import questionary
import requests
import tomli_w
from rich.console import Console
from rich.panel import Panel

from ezscreen.errors import (
    CredentialPermissionError,
    KaggleAuthError,
    NetworkTimeoutError,
    NIMAuthError,
)

CREDS_DIR: Path = Path.home() / ".ezscreen"
CREDS_PATH: Path = CREDS_DIR / "credentials"
NIM_HEALTH_ENDPOINT = "https://health.api.nvidia.com/v1/biology/mit/diffdock"

console = Console()


# ---------------------------------------------------------------------------
# Credential I/O
# ---------------------------------------------------------------------------

def load_credentials() -> dict[str, Any]:
    if not CREDS_PATH.exists():
        return {}
    with CREDS_PATH.open("rb") as f:
        return tomllib.load(f)


def save_credentials(creds: dict[str, Any]) -> None:
    CREDS_DIR.mkdir(parents=True, exist_ok=True)
    with CREDS_PATH.open("wb") as f:
        tomli_w.dump(creds, f)
    try:
        os.chmod(CREDS_PATH, 0o600)
    except OSError:
        pass


def get_kaggle_json_path(creds: dict[str, Any] | None = None) -> Path | None:
    creds = creds or load_credentials()
    raw = creds.get("kaggle_json_path")
    return Path(raw).expanduser() if raw else None


def get_nim_key(creds: dict[str, Any] | None = None) -> str | None:
    creds = creds or load_credentials()
    return creds.get("nim_api_key") or None


def has_kaggle_credentials() -> bool:
    path = get_kaggle_json_path()
    return path is not None and path.exists()


def has_nim_key() -> bool:
    return get_nim_key() is not None


# ---------------------------------------------------------------------------
# Team accounts
# ---------------------------------------------------------------------------

def list_team_accounts(creds: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    creds = creds or load_credentials()
    team  = creds.get("team", {})
    return [{"name": k, **v} for k, v in team.items()]


def add_team_account(name: str, email: str, kaggle_json_path: Path) -> dict[str, Any]:
    data  = validate_kaggle_json(kaggle_json_path)
    creds = load_credentials()
    creds.setdefault("team", {})[name] = {
        "email":            email,
        "kaggle_json_path": str(kaggle_json_path),
        "username":         data["username"],
    }
    save_credentials(creds)
    return creds["team"][name]


def remove_team_account(name: str) -> None:
    creds = load_credentials()
    creds.get("team", {}).pop(name, None)
    save_credentials(creds)


def get_all_kaggle_accounts(creds: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    creds   = creds or load_credentials()
    primary = get_kaggle_json_path(creds)
    accounts: list[dict[str, Any]] = []
    if primary:
        accounts.append({"name": "primary", "kaggle_json_path": str(primary)})
    for acct in list_team_accounts(creds):
        accounts.append(acct)
    return accounts


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _warn_env_overrides() -> None:
    if os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_USERNAME"):
        console.print(
            "[yellow]⚠  Found KAGGLE_KEY / KAGGLE_USERNAME env vars "
            "— these override your kaggle.json[/yellow]"
        )


def _check_json_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        if mode & 0o177:
            raise CredentialPermissionError(
                f"{path} has insecure permissions ({oct(mode)}). "
                "Fix with: chmod 600 ~/.kaggle/kaggle.json"
            )
    except (OSError, NotImplementedError):
        pass


def validate_kaggle_json(path: Path) -> dict[str, str]:
    if not path.exists():
        raise KaggleAuthError(f"kaggle.json not found at {path}")

    _check_json_permissions(path)

    try:
        data: dict[str, str] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise KaggleAuthError(f"kaggle.json is not valid JSON: {exc}") from exc

    for field in ("username", "key"):
        if field not in data:
            raise KaggleAuthError(f"kaggle.json is missing the '{field}' field")

    return data


def _live_kaggle_check(kaggle_data: dict[str, str]) -> None:
    import kaggle as kaggle_pkg  # lazy import — kaggle is slow to load

    os.environ.setdefault("KAGGLE_USERNAME", kaggle_data["username"])
    os.environ.setdefault("KAGGLE_KEY", kaggle_data["key"])
    try:
        kaggle_pkg.api.authenticate()
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg:
            raise KaggleAuthError(
                "API key rejected — go to kaggle.com/settings/account "
                "→ API → Create New Token"
            ) from exc
        if "403" in msg or "forbidden" in msg:
            raise KaggleAuthError(
                "Account needs phone verification — "
                "complete at kaggle.com/settings"
            ) from exc
        raise KaggleAuthError(str(exc)) from exc


def validate_nim_key(key: str) -> None:
    try:
        resp = requests.post(
            NIM_HEALTH_ENDPOINT,
            headers={"Authorization": f"Bearer {key}"},
            json={},
            timeout=10,
        )
    except requests.Timeout as exc:
        raise NetworkTimeoutError("NIM endpoint timed out") from exc
    except requests.ConnectionError as exc:
        raise NetworkTimeoutError(f"Could not reach NIM API: {exc}") from exc

    if resp.status_code == 401:
        raise NIMAuthError(
            "NIM key rejected — get a free key at build.nvidia.com"
        )


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def _step_kaggle(creds: dict[str, Any]) -> dict[str, Any]:
    _warn_env_overrides()

    default = get_kaggle_json_path(creds) or Path("~/.kaggle/kaggle.json").expanduser()
    raw = questionary.text("Path to kaggle.json:", default=str(default)).ask()
    if raw is None:
        raise KeyboardInterrupt

    path = Path(raw).expanduser()

    try:
        kaggle_data = validate_kaggle_json(path)
    except CredentialPermissionError as exc:
        fix = questionary.confirm(f"\n  {exc}\n  Auto-fix permissions?", default=True).ask()
        if fix:
            os.chmod(path, 0o600)
        kaggle_data = validate_kaggle_json(path)

    console.print("  [dim]Checking Kaggle API...[/dim]")
    _live_kaggle_check(kaggle_data)
    console.print(f"  [green]Kaggle ✓[/green]  [dim]{kaggle_data['username']}[/dim]")

    creds["kaggle_json_path"] = str(path)
    return creds


def _step_nim(creds: dict[str, Any]) -> dict[str, Any]:
    console.print("  [dim]optional — only needed for ezscreen validate[/dim]")
    raw = questionary.password("NIM API key (Enter to skip):", default="").ask()
    if raw is None:
        raise KeyboardInterrupt

    if not raw.strip():
        console.print("  [dim]NIM — skipped[/dim]")
        return creds

    console.print("  [dim]Checking NIM API...[/dim]")
    validate_nim_key(raw.strip())
    console.print("  [green]NIM ✓[/green]")

    creds["nim_api_key"] = raw.strip()
    return creds


# ---------------------------------------------------------------------------
# Public wizard entry point
# ---------------------------------------------------------------------------

def run_wizard(update: str | None = None) -> None:
    existing = load_credentials()

    if existing and update is None:
        choice = questionary.select(
            "Credentials already set. What would you like to update?",
            choices=["Kaggle credentials", "NIM API key", "Both", "← Cancel"],
        ).ask()
        if choice is None or choice == "← Cancel":
            return
        update = choice

    creds = dict(existing)
    run_kaggle = update in (None, "Kaggle credentials", "Both")
    run_nim = update in (None, "NIM API key", "Both")

    if run_kaggle:
        console.print("\n[bold]Step 1 — Kaggle[/bold]")
        creds = _step_kaggle(creds)

    if run_nim:
        console.print("\n[bold]Step 2 — NIM[/bold]  [dim](optional)[/dim]")
        creds = _step_nim(creds)

    save_credentials(creds)
    console.print(
        Panel(
            f"  [green]Credentials saved[/green]  [dim]{CREDS_PATH}[/dim]",
            title="[bold]Done[/bold]",
        )
    )
