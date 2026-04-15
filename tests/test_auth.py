from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from ezscreen.auth import validate_kaggle_json, validate_nim_key
from ezscreen.errors import KaggleAuthError, NIMAuthError, NetworkTimeoutError


# ---------------------------------------------------------------------------
# validate_kaggle_json
# ---------------------------------------------------------------------------

def _write_kaggle_json(path: Path, content: dict | str | None) -> Path:
    kj = path / "kaggle.json"
    if content is None:
        return kj  # don't write — used for missing-file test
    if isinstance(content, dict):
        kj.write_text(json.dumps(content))
    else:
        kj.write_text(content)
    return kj


def test_validate_kaggle_json_valid(tmp_path):
    kj = _write_kaggle_json(tmp_path, {"username": "testuser", "key": "abc123xyz"})
    result = validate_kaggle_json(kj)
    assert result["username"] == "testuser"
    assert result["key"] == "abc123xyz"


def test_validate_kaggle_json_missing_file(tmp_path):
    kj = tmp_path / "kaggle.json"
    with pytest.raises(KaggleAuthError, match="not found"):
        validate_kaggle_json(kj)


def test_validate_kaggle_json_malformed_json(tmp_path):
    kj = _write_kaggle_json(tmp_path, "this is { not valid json")
    with pytest.raises(KaggleAuthError, match="not valid JSON"):
        validate_kaggle_json(kj)


def test_validate_kaggle_json_missing_username_field(tmp_path):
    kj = _write_kaggle_json(tmp_path, {"key": "abc123xyz"})
    with pytest.raises(KaggleAuthError, match="'username'"):
        validate_kaggle_json(kj)


def test_validate_kaggle_json_missing_key_field(tmp_path):
    kj = _write_kaggle_json(tmp_path, {"username": "testuser"})
    with pytest.raises(KaggleAuthError, match="'key'"):
        validate_kaggle_json(kj)


def test_validate_kaggle_json_empty_object(tmp_path):
    kj = _write_kaggle_json(tmp_path, {})
    with pytest.raises(KaggleAuthError):
        validate_kaggle_json(kj)


# ---------------------------------------------------------------------------
# validate_nim_key — mocked requests
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, raise_for: type | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if raise_for == requests.exceptions.Timeout:
        resp.raise_for_status.side_effect = requests.exceptions.Timeout()
    elif raise_for == requests.exceptions.ConnectionError:
        resp.raise_for_status.side_effect = requests.exceptions.ConnectionError()
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_validate_nim_key_accepts_valid_key():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(200)
        # should not raise
        validate_nim_key("valid-key-abc")


def test_validate_nim_key_rejects_401():
    with patch("requests.post") as mock_post:
        resp = MagicMock()
        resp.status_code = 401
        mock_post.return_value = resp
        with pytest.raises(NIMAuthError):
            validate_nim_key("bad-key")


def test_validate_nim_key_timeout():
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.Timeout()
        with pytest.raises(NetworkTimeoutError):
            validate_nim_key("any-key")


def test_validate_nim_key_sends_bearer_header():
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(200)
        validate_nim_key("my-secret-key")
        _, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert "Authorization" in headers
        assert "my-secret-key" in headers["Authorization"]
