from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class GitHubError(RuntimeError):
    pass


WORKER_REPO_DEFAULT_NAME = "ezscreen-team-worker"
WORKER_REPO_MARKER_PATH = ".ezscreen-worker.json"
WORKER_WORKFLOW_PATH = ".github/workflows/team-worker.yml"


def _github_request(
    method: str,
    path: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(f"GitHub API failed: {exc.code} {detail}") from exc
    return json.loads(raw) if raw else {}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _github_app_jwt(*, app_id: str, private_key_pem: str) -> str:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise GitHubError("cryptography is required for GitHub App authentication") from exc

    header = {"alg": "RS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (9 * 60),
        "iss": str(app_id),
    }
    signing_input = (
        f"{_b64url(json.dumps(header, separators=(',', ':')).encode('utf-8'))}."
        f"{_b64url(json.dumps(payload, separators=(',', ':')).encode('utf-8'))}"
    )
    key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    signature = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(signature)}"


def _github_request_optional(
    method: str,
    path: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any | None:
    try:
        return _github_request(method, path, token=token, payload=payload)
    except GitHubError as exc:
        if "GitHub API failed: 404" in str(exc):
            return None
        raise


def github_app_installation_token(
    *,
    app_id: str,
    private_key_pem: str,
    installation_id: int,
    permissions: dict[str, str] | None = None,
) -> str:
    jwt_token = _github_app_jwt(app_id=str(app_id), private_key_pem=private_key_pem)
    payload = {"permissions": permissions} if permissions else None
    result = _github_request(
        "POST",
        f"/app/installations/{int(installation_id)}/access_tokens",
        token=jwt_token,
        payload=payload,
    )
    token = str(result.get("token", "")).strip()
    if not token:
        raise GitHubError("GitHub App did not return an installation access token")
    return token


def get_repo_installation(
    *,
    repo: str,
    app_id: str,
    private_key_pem: str,
) -> dict[str, Any] | None:
    owner, name = repo.split("/", 1)
    jwt_token = _github_app_jwt(app_id=str(app_id), private_key_pem=private_key_pem)
    result = _github_request_optional(
        "GET",
        f"/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}/installation",
        token=jwt_token,
    )
    return result if isinstance(result, dict) else None


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    try:
        from nacl import encoding, public
    except Exception as exc:  # pragma: no cover - depends on optional local package
        raise GitHubError("PyNaCl is required to set GitHub Actions secrets") from exc

    public_key = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def set_actions_secret(repo: str, name: str, value: str, *, token: str) -> None:
    key = _github_request("GET", f"/repos/{repo}/actions/secrets/public-key", token=token)
    encrypted = encrypt_secret(key["key"], value)
    _github_request(
        "PUT",
        f"/repos/{repo}/actions/secrets/{name}",
        token=token,
        payload={"encrypted_value": encrypted, "key_id": key["key_id"]},
    )


def configure_worker_repo(
    *,
    repo: str,
    github_token: str,
    kaggle_username: str,
    kaggle_key: str,
    team_token: str,
    server_url: str,
) -> None:
    secrets = {
        "KAGGLE_USERNAME": kaggle_username,
        "KAGGLE_KEY": kaggle_key,
        "EZSCREEN_TEAM_TOKEN": team_token,
        "EZSCREEN_TEAM_SERVER_URL": server_url,
    }
    for name, value in secrets.items():
        set_actions_secret(repo, name, value, token=github_token)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _worker_runtime_files() -> dict[str, str]:
    repo_root = _repo_root()
    file_map = {
        "ezscreen/__init__.py": repo_root / "ezscreen" / "__init__.py",
        "ezscreen/errors.py": repo_root / "ezscreen" / "errors.py",
        "ezscreen/backends/__init__.py": repo_root / "ezscreen" / "backends" / "__init__.py",
        "ezscreen/backends/kaggle/__init__.py": repo_root / "ezscreen" / "backends" / "kaggle" / "__init__.py",
        "ezscreen/backends/kaggle/kernel.py": repo_root / "ezscreen" / "backends" / "kaggle" / "kernel.py",
        "ezscreen/backends/kaggle/poller.py": repo_root / "ezscreen" / "backends" / "kaggle" / "poller.py",
        "ezscreen/backends/kaggle/runner.py": repo_root / "ezscreen" / "backends" / "kaggle" / "runner.py",
        "ezscreen/backends/kaggle/templates/vina_shard.ipynb.j2": repo_root / "ezscreen" / "backends" / "kaggle" / "templates" / "vina_shard.ipynb.j2",
        "ezscreen/team/__init__.py": repo_root / "ezscreen" / "team" / "__init__.py",
        "ezscreen/team/client.py": repo_root / "ezscreen" / "team" / "client.py",
        "ezscreen/team/kaggle_upload_ticket.py": repo_root / "ezscreen" / "team" / "kaggle_upload_ticket.py",
        "ezscreen/team/state.py": repo_root / "ezscreen" / "team" / "state.py",
        "ezscreen/team/worker.py": repo_root / "ezscreen" / "team" / "worker.py",
        WORKER_WORKFLOW_PATH: repo_root / ".github" / "workflows" / "team-worker.yml",
    }
    files = {path: src.read_text(encoding="utf-8") for path, src in file_map.items()}
    files["pyproject.toml"] = """[build-system]
requires = [\"setuptools>=68\", \"wheel\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"ezscreen-team-worker\"
version = \"0.1.0\"
description = \"Dedicated private ezscreen team worker runtime\"
readme = \"README.md\"
requires-python = \">=3.11\"
dependencies = [
  \"kaggle>=1.7.4.5\",
  \"jinja2>=3.1.4\",
  \"requests>=2.32.3\",
  \"rich>=13.7.0\",
]

[tool.setuptools]
packages = [\"ezscreen\", \"ezscreen.backends\", \"ezscreen.backends.kaggle\", \"ezscreen.team\"]

[tool.setuptools.package-data]
\"ezscreen.backends.kaggle\" = [\"templates/*.j2\"]
\"ezscreen.tui\" = [\"theme.tcss\"]
"""
    files["README.md"] = """# ezscreen Team Worker

This is a dedicated private repository for ezscreen team-runner GitHub Actions jobs.

Do not store research code, manuscripts, prompts, datasets, notebooks, or unpublished work in this repository.
It exists only to host the worker workflow and the minimal ezscreen runtime needed for team jobs.
"""
    files[".gitignore"] = """__pycache__/
*.pyc
team_worker_runs/
.venv/
"""
    files[WORKER_REPO_MARKER_PATH] = json.dumps(
        {
            "managed_by": "ezscreen",
            "kind": "team_worker_repo",
            "private_required": True,
            "purpose": "Dedicated private GitHub Actions worker repository. Research work must not be stored here.",
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    return files


def _repo_content_path(path: str) -> str:
    return urllib.parse.quote(path, safe="/")


def _get_repo(repo: str, *, token: str) -> dict[str, Any] | None:
    result = _github_request_optional("GET", f"/repos/{repo}", token=token)
    return result if isinstance(result, dict) else None


def _get_repo_file(repo: str, path: str, *, token: str) -> dict[str, Any] | None:
    result = _github_request_optional("GET", f"/repos/{repo}/contents/{_repo_content_path(path)}", token=token)
    return result if isinstance(result, dict) else None


def _put_repo_text_file(repo: str, path: str, text: str, *, token: str, message: str) -> None:
    existing = _get_repo_file(repo, path, token=token)
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
    }
    if existing and existing.get("sha"):
        payload["sha"] = existing["sha"]
    _github_request(
        "PUT",
        f"/repos/{repo}/contents/{_repo_content_path(path)}",
        token=token,
        payload=payload,
    )


def _normalize_repo_name(repo_name: str | None, owner_login: str) -> str:
    candidate = (repo_name or WORKER_REPO_DEFAULT_NAME).strip()
    if not candidate:
        candidate = WORKER_REPO_DEFAULT_NAME
    if "/" in candidate:
        owner, name = candidate.split("/", 1)
        if owner.strip().lower() != owner_login.strip().lower():
            raise GitHubError("Worker repo must be owned by the signed-in GitHub account")
        candidate = name.strip()
    if not candidate:
        raise GitHubError("Worker repo name is empty")
    return candidate


def _candidate_repo_names(base_name: str) -> list[str]:
    return [base_name, *[f"{base_name}-{index}" for index in range(2, 21)]]


def _create_private_repo(name: str, *, token: str) -> dict[str, Any]:
    result = _github_request(
        "POST",
        "/user/repos",
        token=token,
        payload={
            "name": name,
            "private": True,
            "auto_init": True,
            "description": "Dedicated private ezscreen team worker repository",
        },
    )
    if not isinstance(result, dict) or not result.get("full_name"):
        raise GitHubError("GitHub did not return the created worker repository")
    return result


def sync_managed_worker_repo(repo: str, *, token: str) -> None:
    for path, text in _worker_runtime_files().items():
        _put_repo_text_file(repo, path, text, token=token, message=f"ezscreen: sync {path}")


def dispatch_worker_workflow(
    *,
    repo: str,
    token: str,
    mode: str,
    worker_job_id: str,
    workflow_file: str = "team-worker.yml",
) -> None:
    repo_info = _github_request("GET", f"/repos/{repo}", token=token)
    default_branch = str(repo_info.get("default_branch", "")).strip()
    if not default_branch:
        raise GitHubError(f"GitHub did not return a default branch for {repo}")
    _github_request(
        "POST",
        f"/repos/{repo}/actions/workflows/{urllib.parse.quote(workflow_file, safe='')}/dispatches",
        token=token,
        payload={
            "ref": default_branch,
            "inputs": {
                "mode": mode,
                "worker_job_id": worker_job_id,
            },
        },
    )


def dispatch_worker_workflow_via_app(
    *,
    repo: str,
    app_id: str,
    private_key_pem: str,
    installation_id: int,
    mode: str,
    worker_job_id: str,
    workflow_file: str = "team-worker.yml",
) -> None:
    token = github_app_installation_token(
        app_id=app_id,
        private_key_pem=private_key_pem,
        installation_id=installation_id,
        permissions={"actions": "write", "contents": "read", "metadata": "read"},
    )
    dispatch_worker_workflow(
        repo=repo,
        token=token,
        mode=mode,
        worker_job_id=worker_job_id,
        workflow_file=workflow_file,
    )


def provision_managed_worker_repo(*, github_token: str, repo_name: str | None = None) -> dict[str, Any]:
    user = _github_request("GET", "/user", token=github_token)
    login = str(user.get("login", "")).strip()
    if not login:
        raise GitHubError("GitHub login could not be resolved from the provided token")

    explicit_name = repo_name is not None and repo_name.strip() != ""
    base_name = _normalize_repo_name(repo_name, login)
    for candidate_name in ([base_name] if explicit_name else _candidate_repo_names(base_name)):
        full_repo = f"{login}/{candidate_name}"
        repo = _get_repo(full_repo, token=github_token)
        if repo is None:
            created = _create_private_repo(candidate_name, token=github_token)
            sync_managed_worker_repo(full_repo, token=github_token)
            return {"repo": full_repo, "created": True, "private": bool(created.get("private", True))}
        if not bool(repo.get("private")):
            if explicit_name:
                raise GitHubError("Existing worker repo must be private. ezscreen will not use a public repository.")
            continue
        if _get_repo_file(full_repo, WORKER_REPO_MARKER_PATH, token=github_token) is None:
            if explicit_name:
                raise GitHubError(
                    "Existing repo is not an ezscreen-managed worker repo. Use a fresh name or let ezscreen create one automatically."
                )
            continue
        sync_managed_worker_repo(full_repo, token=github_token)
        return {"repo": full_repo, "created": False, "private": True}
    raise GitHubError("Could not find an available dedicated ezscreen worker repo name")


def provision_and_configure_worker_repo(
    *,
    github_token: str,
    kaggle_username: str,
    kaggle_key: str,
    team_token: str,
    server_url: str,
    repo_name: str | None = None,
) -> dict[str, Any]:
    repo_info = provision_managed_worker_repo(github_token=github_token, repo_name=repo_name)
    configure_worker_repo(
        repo=repo_info["repo"],
        github_token=github_token,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        team_token=team_token,
        server_url=server_url,
    )
    return repo_info


