import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ezscreen.backends.kaggle import poller


@pytest.fixture
def polling(monkeypatch):
    api = SimpleNamespace(authenticate=Mock(), kernels_status=Mock())
    monkeypatch.setitem(sys.modules, "kaggle", SimpleNamespace(api=api))
    monkeypatch.setattr(poller.time, "sleep", Mock())
    monkeypatch.setattr(poller, "_notify", Mock())
    monkeypatch.setattr(poller, "_fetch_and_show_logs", Mock())
    monkeypatch.setattr(poller, "_TIMEOUT", 150)
    return api


@pytest.mark.parametrize("show_live", [False, True])
def test_status_connection_error_keeps_same_job(polling, show_live):
    polling.kernels_status.side_effect = [
        ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING"),
        SimpleNamespace(status="running"),
        SimpleNamespace(status="complete"),
    ]
    result = poller.poll_until_done(
        "owner/job", "run", retry_limit=0, show_live=show_live
    )
    assert result == {"status": "complete", "error_type": None, "retry_count": 0}
    assert polling.kernels_status.call_count == 3
    assert all(
        call.args == ("owner/job",) for call in polling.kernels_status.call_args_list
    )
    poller._fetch_and_show_logs.assert_not_called()
    poller._notify.assert_called_once_with("run", "complete")


def test_persistent_status_outage_times_out_without_kernel_failure(polling):
    polling.kernels_status.side_effect = ConnectionError("unavailable")
    result = poller.poll_until_done("owner/job", "run", show_live=False)
    assert result == {"status": "timeout", "error_type": "timeout", "retry_count": 0}
    assert polling.kernels_status.call_count == 5
    poller._fetch_and_show_logs.assert_not_called()
    poller._notify.assert_called_once_with("run", "timeout")


def test_missing_remote_status_remains_unconfirmed(polling):
    polling.kernels_status.side_effect = [
        SimpleNamespace(),
        SimpleNamespace(status="complete"),
    ]
    assert (
        poller.poll_until_done("owner/job", "run", show_live=False)["status"]
        == "complete"
    )
    poller._fetch_and_show_logs.assert_not_called()


@pytest.mark.parametrize(
    "message,expected",
    [("CUDA out of memory", "retry"), ("compilation terminated", "failed")],
)
def test_confirmed_remote_error_preserves_failure_policy(polling, message, expected):
    polling.kernels_status.return_value = SimpleNamespace(
        status="error", failure_message=message
    )
    result = poller.poll_until_done("owner/job", "run", show_live=False)
    assert result["status"] == expected
    poller._fetch_and_show_logs.assert_called_once_with("owner/job")


def test_authentication_exception_is_not_remote_failure(polling):
    polling.authenticate.side_effect = [
        RuntimeError("auth transport unavailable"),
        None,
    ]
    polling.kernels_status.return_value = SimpleNamespace(status="complete")
    assert (
        poller.poll_until_done("owner/job", "run", show_live=False)["status"]
        == "complete"
    )
    poller._fetch_and_show_logs.assert_not_called()
