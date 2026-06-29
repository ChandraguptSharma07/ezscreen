from __future__ import annotations

import pytest
import requests

from ezscreen.errors import LibrarySourceUnavailableError
from ezscreen.libraries import zinc


class _Resp:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class _Session:
    def __init__(self, resp: _Resp) -> None:
        self._resp = resp

    def get(self, *a, **k) -> _Resp:
        return self._resp


def test_fetch_page_raises_on_antibot_page():
    sess = _Session(_Resp("<html><body>Verification Required</body></html>"))
    with pytest.raises(LibrarySourceUnavailableError):
        zinc._fetch_page(sess, {}, 1)


def test_fetch_page_raises_on_http_403():
    sess = _Session(_Resp("forbidden", status=403))
    with pytest.raises(LibrarySourceUnavailableError):
        zinc._fetch_page(sess, {}, 1)


def test_download_does_not_write_empty_file_when_source_dead(tmp_path, monkeypatch):
    def _dead(*a, **k):
        raise LibrarySourceUnavailableError("dead")

    monkeypatch.setattr(zinc, "_fetch_page", _dead)
    out = tmp_path / "lib.smi"
    with pytest.raises(LibrarySourceUnavailableError):
        zinc.download_zinc_library(out, size="1k")
    assert not out.exists()


def test_fetch_page_parses_valid_response():
    body = "smiles\tzinc_id\nCCO\tZINC1\nc1ccccc1\tZINC2\n"
    rows = zinc._fetch_page(_Session(_Resp(body)), {}, 1)
    assert rows == [("CCO", "ZINC1"), ("c1ccccc1", "ZINC2")]
