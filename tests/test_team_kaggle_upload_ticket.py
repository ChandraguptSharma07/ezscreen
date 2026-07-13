from __future__ import annotations

from ezscreen.team.kaggle_upload_ticket import (
    UploadTicketUnavailable,
    commit_dataset_from_blob_tokens,
)


def test_commit_dataset_retries_with_unique_slug_after_collision():
    calls: list[tuple[str, str]] = []

    class FakeApi:
        def dataset_create_new_from_blob_tokens(self, **kwargs):
            calls.append((kwargs["dataset_slug"], kwargs["title"]))
            if len(calls) == 1:
                raise UploadTicketUnavailable(
                    "upload_ticket_unavailable: Kaggle dataset create failed: Path must be unique"
                )

    dataset_ref = commit_dataset_from_blob_tokens(
        username="worker-user",
        slug="ezscreen-job",
        title="ezscreen job",
        files=[{"filename": "shard_000.smi", "blob_token": "blob-1"}],
        api=FakeApi(),
    )

    assert dataset_ref == "worker-user/ezscreen-job-r1"
    assert calls == [
        ("ezscreen-job", "ezscreen job"),
        ("ezscreen-job-r1", "ezscreen job r1"),
    ]



def test_commit_dataset_retries_transient_create_failure_without_changing_slug():
    calls: list[str] = []

    class FakeApi:
        def dataset_create_new_from_blob_tokens(self, **kwargs):
            calls.append(kwargs["dataset_slug"])
            if len(calls) == 1:
                raise UploadTicketUnavailable(
                    "upload_ticket_unavailable: Kaggle dataset create failed: HTTPError: 500 Server Error"
                )

    dataset_ref = commit_dataset_from_blob_tokens(
        username="worker-user",
        slug="ezscreen-job",
        title="ezscreen job",
        files=[{"filename": "shard_000.smi", "blob_token": "blob-1"}],
        api=FakeApi(),
    )

    assert dataset_ref == "worker-user/ezscreen-job"
    assert calls == ["ezscreen-job", "ezscreen-job"]



def test_commit_dataset_raises_after_exhausting_collision_suffixes():
    class FakeApi:
        def dataset_create_new_from_blob_tokens(self, **kwargs):
            raise UploadTicketUnavailable(
                "upload_ticket_unavailable: Kaggle dataset create failed: Path must be unique"
            )

    try:
        commit_dataset_from_blob_tokens(
            username="worker-user",
            slug="ezscreen-job",
            title="ezscreen job",
            files=[{"filename": "shard_000.smi", "blob_token": "blob-1"}],
            api=FakeApi(),
        )
    except UploadTicketUnavailable as exc:
        assert "Path must be unique" in str(exc)
    else:
        raise AssertionError("expected UploadTicketUnavailable")
