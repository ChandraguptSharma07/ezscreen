from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

_DATASET_CREATE_RETRY_LIMIT = 4


class UploadTicketUnavailable(RuntimeError):
    """Raised when the Kaggle SDK no longer exposes upload-ticket internals."""


@dataclass(frozen=True)
class UploadTicket:
    filename: str
    size_bytes: int
    upload_url: str
    blob_token: str
    expires_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "upload_url": self.upload_url,
            "blob_token": self.blob_token,
            "expires_at": self.expires_at,
        }


def _get_api(api: Any | None = None) -> Any:
    if api is not None:
        return api
    import kaggle

    kaggle.api.authenticate()
    return kaggle.api


def mint_upload_ticket(filename: str, size_bytes: int, api: Any | None = None) -> UploadTicket:
    """Mint a signed Kaggle upload URL.

    This intentionally uses a narrow internal adapter. If Kaggle changes the
    low-level method shape, callers get upload_ticket_unavailable rather than a
    silent server-mediated fallback that would expose member credentials.
    """
    kaggle_api = _get_api(api)

    starter = getattr(kaggle_api, "start_blob_upload", None)
    if starter is not None:
        result = starter(filename=filename, content_length=size_bytes)
    else:
        result = _start_current_kaggle_blob_upload(kaggle_api, filename, size_bytes)

    if not isinstance(result, dict):
        result = {
            "upload_url": getattr(result, "upload_url", None) or getattr(result, "create_url", None),
            "blob_token": getattr(result, "blob_token", None) or getattr(result, "token", None),
            "expires_at": getattr(result, "expires_at", None),
        }
    upload_url = result.get("upload_url") or result.get("url") or result.get("create_url")
    blob_token = result.get("blob_token") or result.get("token")
    if not upload_url or not blob_token:
        raise UploadTicketUnavailable("upload_ticket_unavailable: Kaggle upload ticket shape is unsupported")
    return UploadTicket(
        filename=filename,
        size_bytes=size_bytes,
        upload_url=upload_url,
        blob_token=blob_token,
        expires_at=result.get("expires_at"),
    )


def _start_current_kaggle_blob_upload(kaggle_api: Any, filename: str, size_bytes: int) -> Any:
    builder = getattr(kaggle_api, "build_kaggle_client", None)
    retry = getattr(kaggle_api, "with_retry", lambda method: method)
    if builder is None:
        raise UploadTicketUnavailable("upload_ticket_unavailable: Kaggle SDK blob upload entrypoint is missing")
    try:
        from kagglesdk.blobs.types.blob_api_service import (
            ApiBlobType,
            ApiStartBlobUploadRequest,
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle SDK internals
        raise UploadTicketUnavailable("upload_ticket_unavailable: Kaggle SDK blob request types are missing") from exc

    request = ApiStartBlobUploadRequest()
    request.type = ApiBlobType.DATASET
    request.name = filename
    request.content_length = size_bytes
    request.last_modified_epoch_seconds = 0

    try:
        with builder() as kaggle_client:
            method = kaggle_client.blobs.blob_api_client.start_blob_upload
            return retry(method)(request)
    except UploadTicketUnavailable:
        raise
    except Exception as exc:
        raise UploadTicketUnavailable(f"upload_ticket_unavailable: Kaggle blob upload start failed: {exc}") from exc


def upload_file_to_signed_url(path: Path, upload_url: str) -> None:
    with path.open("rb") as f:
        resp = requests.put(upload_url, data=f, timeout=300)
    resp.raise_for_status()


def commit_dataset_from_blob_tokens(
    *,
    username: str,
    slug: str,
    title: str,
    files: list[dict[str, Any]],
    api: Any | None = None,
) -> str:
    kaggle_api = _get_api(api)
    committer = getattr(kaggle_api, "dataset_create_new_from_blob_tokens", None)
    last_error: UploadTicketUnavailable | None = None
    transient_retries = 0
    slug_suffix_index = 0
    while slug_suffix_index < _DATASET_CREATE_RETRY_LIMIT:
        attempt_slug = _slug_with_suffix(slug, slug_suffix_index)
        attempt_title = _title_with_suffix(title, slug_suffix_index)
        try:
            if committer is not None:
                committer(
                    owner_slug=username,
                    dataset_slug=attempt_slug,
                    title=attempt_title,
                    files=[{"name": item["filename"], "blobToken": item["blob_token"]} for item in files],
                    public=False,
                )
            else:
                _commit_dataset_blob_tokens_current_kaggle(
                    kaggle_api,
                    username=username,
                    slug=attempt_slug,
                    title=attempt_title,
                    files=files,
                )
            return f"{username}/{attempt_slug}"
        except UploadTicketUnavailable as exc:
            last_error = exc
            message = str(exc)
            if _is_slug_taken_error(message):
                slug_suffix_index += 1
                transient_retries = 0
                continue
            if _is_transient_dataset_create_error(message) and transient_retries < (_DATASET_CREATE_RETRY_LIMIT - 1):
                transient_retries += 1
                continue
            raise
    if last_error is not None:
        raise last_error
    raise UploadTicketUnavailable("upload_ticket_unavailable: Kaggle dataset create exhausted retries")


def _slug_with_suffix(slug: str, suffix_index: int) -> str:
    if suffix_index <= 0:
        return slug
    suffix = f"-r{suffix_index}"
    max_len = 50
    base = slug[: max_len - len(suffix)].rstrip("-")
    return f"{base}{suffix}" if base else f"ezscreen{suffix}"


def _title_with_suffix(title: str, suffix_index: int) -> str:
    if suffix_index <= 0:
        return title
    suffix = f" r{suffix_index}"
    max_len = 50
    base = title[: max_len - len(suffix)].rstrip()
    return f"{base}{suffix}" if base else f"ezscreen{suffix}"


def _is_slug_taken_error(message: str) -> bool:
    lowered = message.lower()
    return "path must be unique" in lowered or "already exists" in lowered


def _is_transient_dataset_create_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "500 server error" in lowered
        or "502 bad gateway" in lowered
        or "503 service unavailable" in lowered
        or "504 gateway timeout" in lowered
        or "connection aborted" in lowered
        or "connection reset" in lowered
        or "temporarily unavailable" in lowered
        or "timed out" in lowered
    )


def _commit_dataset_blob_tokens_current_kaggle(
    kaggle_api: Any,
    *,
    username: str,
    slug: str,
    title: str,
    files: list[dict[str, Any]],
) -> None:
    builder = getattr(kaggle_api, "build_kaggle_client", None)
    retry = getattr(kaggle_api, "with_retry", lambda method: method)
    if builder is None:
        raise UploadTicketUnavailable("upload_ticket_unavailable: Kaggle SDK dataset create entrypoint is missing")
    try:
        from kagglesdk.datasets.types.dataset_api_service import (
            ApiCreateDatasetRequest,
            ApiDatasetNewFile,
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle SDK internals
        raise UploadTicketUnavailable("upload_ticket_unavailable: Kaggle SDK dataset request types are missing") from exc

    request = ApiCreateDatasetRequest()
    request.owner_slug = username
    request.slug = slug
    request.title = title
    request.is_private = True
    request.files = []
    for item in files:
        dataset_file = ApiDatasetNewFile()
        dataset_file.token = item["blob_token"]
        request.files.append(dataset_file)

    try:
        with builder() as kaggle_client:
            method = kaggle_client.datasets.dataset_api_client.create_dataset
            response = retry(method)(request)
    except UploadTicketUnavailable:
        raise
    except Exception as exc:
        detail = f'{type(exc).__name__}: {exc}'
        response_text = getattr(getattr(exc, 'response', None), 'text', None)
        if response_text:
            detail = f'{detail} | response={response_text[:500]}'
        raise UploadTicketUnavailable(f"upload_ticket_unavailable: Kaggle dataset create failed: {detail}") from exc

    if getattr(response, "error", None):
        raise UploadTicketUnavailable(f"upload_ticket_unavailable: Kaggle dataset create failed: {response.error}")
