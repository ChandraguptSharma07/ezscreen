from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from ezscreen.team.kaggle_upload_ticket import mint_upload_ticket, upload_file_to_signed_url
from ezscreen.team.state import TeamStateError, TeamStore, extract_result_zip


def test_default_team_server_url_prefers_env(monkeypatch):
    from ezscreen.team.client import default_server_url

    monkeypatch.setenv("EZSCREEN_TEAM_SERVER_URL", "https://team.example.test/")

    assert default_server_url() == "https://team.example.test"


def test_one_team_only_create_join_leave_rules(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")

    with pytest.raises(TeamStateError):
        store.create_team(owner_name="alice", owner_user_id="u1", team_name="Other")

    joined = store.join_team(
        invite_code=team["invite_code"],
        user_name="bob",
        user_id="u2",
        github_repo="bob/worker",
    )
    assert joined["team_id"] == team["team_id"]

    with pytest.raises(TeamStateError):
        store.join_team(
            invite_code=team["invite_code"],
            user_name="bob",
            user_id="u2",
            github_repo="bob/worker2",
        )

    store.leave_team("u2")
    joined_again = store.join_team(
        invite_code=team["invite_code"],
        user_name="bob",
        user_id="u2",
        github_repo="bob/worker2",
    )
    assert joined_again["member_id"] != joined["member_id"]


def test_github_session_uses_hashed_server_token(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    user = store.ensure_github_user(github_id=123, login="octocat", name="Octo Cat")
    session = store.create_session(user_id=user["user_id"], provider="github", provider_login="octocat")

    assert session["session_token"]
    loaded = store.session_from_token(session["session_token"])
    assert loaded["user_id"] == "gh_123"
    assert loaded["provider_login"] == "octocat"
    with store.connection() as conn:
        row = conn.execute("SELECT token_hash FROM auth_sessions WHERE session_id = ?", (session["session_id"],)).fetchone()
    assert row["token_hash"] != session["session_token"]


def test_revoke_session_marks_token_revoked(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    user = store.ensure_github_user(github_id=123, login="octocat", name="Octo Cat")
    session = store.create_session(user_id=user["user_id"], provider="github", provider_login="octocat")

    result = store.revoke_session(session["session_token"])

    assert result["ok"] is True
    assert result["user_id"] == "gh_123"
    with pytest.raises(TeamStateError):
        store.session_from_token(session["session_token"])
    with store.connection() as conn:
        row = conn.execute("SELECT revoked_at FROM auth_sessions WHERE session_id = ?", (session["session_id"],)).fetchone()
    assert row["revoked_at"]



def test_connect_worker_rotates_member_repo_and_token(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")

    updated = store.connect_worker("u1", github_repo="alice/worker", kaggle_username="alice_kaggle")

    assert updated["github_repo"] == "alice/worker"
    assert updated["kaggle_username"] == "alice_kaggle"
    assert updated["team_token"]
    status = store.status("u1")
    assert status["members"][0]["github_repo"] == "alice/worker"
    assert status["members"][0]["kaggle_username"] == "alice_kaggle"
    assert store.worker_from_token(updated["team_token"])["member_id"] == team["member_id"]


def test_disconnect_worker_keeps_membership_but_makes_worker_unavailable(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    store.connect_worker("u1", github_repo="alice/worker", kaggle_username="alice_kaggle")

    result = store.disconnect_worker("u1")

    assert result["disconnected"] is True
    status = store.status("u1")
    assert status["members"][0]["github_repo"] == ""
    assert status["members"][0]["kaggle_username"] == ""
    assert status["members"][0]["paused"] == 1


def test_init_releases_stale_pre_run_jobs_even_if_not_active_slot(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    store.connect_worker("u1", github_repo="alice/worker")
    old_job = store.create_team_job(
        user_id="u1",
        run_id="ezs-stale-background",
        payload={"files": [{"filename": "old.smi", "size_bytes": 1}]},
    )
    old_worker_job_id = old_job["worker_jobs"][0]["worker_job_id"]
    with store.connection() as conn:
        conn.execute(
            "UPDATE worker_jobs SET status = ?, created_at = ? WHERE worker_job_id = ?",
            ("uploads_ready", "2020-01-01T00:00:00Z", old_worker_job_id),
        )
        conn.execute(
            "UPDATE members SET active_worker_job_id = NULL WHERE user_id = ?",
            ("u1",),
        )

    reopened = TeamStore(tmp_path / "team.db")
    with reopened.connection() as conn:
        row = conn.execute(
            "SELECT status, completed_at FROM worker_jobs WHERE worker_job_id = ?",
            (old_worker_job_id,),
        ).fetchone()
    assert row["status"] == "timeout"
    assert row["completed_at"] is not None


def test_init_releases_stale_pre_run_worker_slots(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    connected = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    store.connect_worker("u1", github_repo="alice/worker")
    job = store.create_team_job(
        user_id="u1",
        run_id="ezs-stale-pre-run",
        payload={"files": [{"filename": "shard.smi", "size_bytes": 1}]},
    )
    worker_job_id = job["worker_jobs"][0]["worker_job_id"]
    with store.connection() as conn:
        conn.execute(
            "UPDATE worker_jobs SET created_at = ? WHERE worker_job_id = ?",
            ("2020-01-01T00:00:00Z", worker_job_id),
        )

    reopened = TeamStore(tmp_path / "team.db")
    status = reopened.status("u1")

    assert status["members"][0]["active_worker_job_id"] is None
    with reopened.connection() as conn:
        row = conn.execute(
            "SELECT status, completed_at FROM worker_jobs WHERE worker_job_id = ?",
            (worker_job_id,),
        ).fetchone()
        callback = conn.execute(
            "SELECT event_type, payload_json FROM callbacks WHERE worker_job_id = ? ORDER BY created_at DESC LIMIT 1",
            (worker_job_id,),
        ).fetchone()
    assert row["status"] == "timeout"
    assert row["completed_at"] is not None
    assert callback["event_type"] == "timeout"
    assert "stale_pre_run_timeout" in callback["payload_json"]


def test_init_clears_stale_active_worker_job_for_disconnected_member(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    store.connect_worker("u1", github_repo="alice/worker")
    store.disconnect_worker("u1")
    with store.connection() as conn:
        conn.execute(
            "UPDATE members SET active_worker_job_id = ? WHERE user_id = ?",
            ("wjob_stale", "u1"),
        )

    reopened = TeamStore(tmp_path / "team.db")
    status = reopened.status("u1")

    assert status["members"][0]["github_repo"] == ""
    assert status["members"][0]["active_worker_job_id"] is None


def test_paused_worker_skipped_and_active_worker_gets_one_job(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    store.join_team(
        invite_code=team["invite_code"],
        user_name="bob",
        user_id="u2",
        github_repo="bob/worker",
    )
    store.set_worker_paused("u1", True)

    job = store.create_team_job(
        user_id="u2",
        run_id="ezs-test",
        payload={"files": [{"filename": f"shard_{i}.smi", "size_bytes": i + 1} for i in range(4)]},
    )
    assert len(job["worker_jobs"]) == 1
    assert len(job["worker_jobs"][0]["files"]) == 4

    with pytest.raises(TeamStateError):
        store.create_team_job(
            user_id="u2",
            run_id="ezs-test-2",
            payload={"files": [{"filename": "next.smi", "size_bytes": 1}]},
        )


def test_team_jobs_ignore_members_without_connected_worker_repo(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    bob = store.join_team(
        invite_code=team["invite_code"],
        user_name="bob",
        user_id="u2",
        github_repo="bob/worker",
    )
    carol = store.join_team(
        invite_code=team["invite_code"],
        user_name="carol",
        user_id="u3",
        github_repo="",
    )
    store.connect_worker("u2", github_repo="bob/worker")

    job = store.create_team_job(
        user_id="u1",
        run_id="ezs-connected-only",
        payload={"files": [{"filename": "shard.smi", "size_bytes": 1}]},
    )

    assert [item["member_id"] for item in job["worker_jobs"]] == [bob["member_id"]]
    status = store.status("u3")
    assert status["members"][-1]["github_repo"] == ""


def test_team_member_assignment_sends_common_files_and_requested_shards(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    bob = store.join_team(
        invite_code=team["invite_code"],
        user_name="bob",
        user_id="u2",
        github_repo="bob/worker",
    )
    carol = store.join_team(
        invite_code=team["invite_code"],
        user_name="carol",
        user_id="u3",
        github_repo="carol/worker",
    )
    store.connect_worker("u2", github_repo="bob/worker")
    store.connect_worker("u3", github_repo="carol/worker")

    job = store.create_team_job(
        user_id="u1",
        run_id="ezs-team-assign",
        payload={
            "files": [
                {"filename": "receptor.pdbqt", "size_bytes": 10, "role": "receptor"},
                {"filename": "team_notebook.ipynb", "size_bytes": 20, "role": "notebook"},
                {"filename": "shard_000.smi", "size_bytes": 1, "role": "ligand_shard"},
                {"filename": "shard_001.smi", "size_bytes": 1, "role": "ligand_shard"},
                {"filename": "shard_002.smi", "size_bytes": 1, "role": "ligand_shard"},
            ],
            "team_member_assignments": [
                {"member_id": bob["member_id"], "shard_count": 2},
                {"member_id": carol["member_id"], "shard_count": 1},
            ],
        },
    )

    by_member = {item["member_id"]: [f["filename"] for f in item["files"]] for item in job["worker_jobs"]}
    assert team["member_id"] not in by_member
    assert by_member[bob["member_id"]] == [
        "receptor.pdbqt",
        "team_notebook.ipynb",
        "shard_000.smi",
        "shard_001.smi",
    ]
    assert by_member[carol["member_id"]] == [
        "receptor.pdbqt",
        "team_notebook.ipynb",
        "shard_002.smi",
    ]


def test_start_team_job_sends_team_member_assignments(tmp_path):
    from ezscreen.team.coordinator import start_team_job

    receptor = tmp_path / "receptor.pdbqt"
    notebook = tmp_path / "team_notebook.ipynb"
    shard = tmp_path / "shard_000.smi"
    receptor.write_text("rec", encoding="utf-8")
    notebook.write_text("nb", encoding="utf-8")
    shard.write_text("CC ligand", encoding="utf-8")
    calls = []

    class FakeClient:
        def request(self, method, path, payload):
            calls.append((method, path, payload))
            return {"team_job_id": "tjob_1", "worker_jobs": []}

    assignments = [{"member_id": "mem_b", "shard_count": 1}]
    result = start_team_job(
        client=FakeClient(),
        user_id="u1",
        run_id="ezs-test",
        receptor_pdbqt=receptor,
        shard_paths=[shard],
        notebook_path=notebook,
        engine="unidock",
        team_member_assignments=assignments,
    )

    assert result["team_job_id"] == "tjob_1"
    payload = calls[0][2]["payload"]
    assert payload["team_member_assignments"] == assignments
    assert [item["role"] for item in payload["files"]] == ["receptor", "notebook", "ligand_shard"]


def test_upload_job_files_waits_for_all_expected_tickets(monkeypatch, tmp_path):
    from ezscreen.team.coordinator import upload_job_files

    receptor = tmp_path / "receptor.pdbqt"
    notebook = tmp_path / "team_notebook.ipynb"
    shard_a = tmp_path / "shard_000.smi"
    shard_b = tmp_path / "shard_001.smi"
    for path in (receptor, notebook, shard_a, shard_b):
        path.write_text(path.name, encoding="utf-8")

    tickets_round_1 = [
        {"ticket_id": "t1", "filename": "receptor.pdbqt", "upload_url": "https://upload.invalid/1"},
        {"ticket_id": "t2", "filename": "team_notebook.ipynb", "upload_url": "https://upload.invalid/2"},
        {"ticket_id": "t3", "filename": "shard_000.smi", "upload_url": "https://upload.invalid/3"},
    ]
    tickets_round_2 = [
        *tickets_round_1,
        {"ticket_id": "t4", "filename": "receptor.pdbqt", "upload_url": "https://upload.invalid/4"},
        {"ticket_id": "t5", "filename": "team_notebook.ipynb", "upload_url": "https://upload.invalid/5"},
        {"ticket_id": "t6", "filename": "shard_001.smi", "upload_url": "https://upload.invalid/6"},
    ]
    calls = {"poll": 0, "acks": [], "uploads": []}

    class FakeClient:
        def request(self, method, path, payload=None):
            if method == "GET":
                calls["poll"] += 1
                return {"tickets": tickets_round_1 if calls["poll"] == 1 else tickets_round_2}
            calls["acks"].append(payload["ticket_id"])
            return {"ok": True}

    monkeypatch.setattr(
        "ezscreen.team.coordinator.upload_file_to_signed_url",
        lambda path, url: calls["uploads"].append((Path(path).name, url)),
    )
    monkeypatch.setattr("ezscreen.team.coordinator.time.sleep", lambda _: None)

    tickets = upload_job_files(
        client=FakeClient(),
        user_id="u1",
        team_job_id="tjob_1",
        file_paths=[receptor, notebook, shard_a, shard_b],
        expected_ticket_count=6,
        poll_timeout_s=5,
    )

    assert len(tickets) == 6
    assert calls["poll"] == 2
    assert calls["acks"] == ["t1", "t2", "t3", "t4", "t5", "t6"]
    assert calls["uploads"] == [
        ("receptor.pdbqt", "https://upload.invalid/1"),
        ("team_notebook.ipynb", "https://upload.invalid/2"),
        ("shard_000.smi", "https://upload.invalid/3"),
        ("receptor.pdbqt", "https://upload.invalid/4"),
        ("team_notebook.ipynb", "https://upload.invalid/5"),
        ("shard_001.smi", "https://upload.invalid/6"),
    ]


def test_upload_ticket_rejects_wrong_worker_and_kaggle_secret_payload(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    bob = store.join_team(
        invite_code=team["invite_code"],
        user_name="bob",
        user_id="u2",
        github_repo="bob/worker",
    )
    store.connect_worker("u1", github_repo="alice/worker")
    job = store.create_team_job(
        user_id="u1",
        run_id="ezs-test",
        payload={"files": [{"filename": "shard.smi", "size_bytes": 10}]},
    )
    worker_job_id = job["worker_jobs"][0]["worker_job_id"]

    with pytest.raises(TeamStateError):
        store.record_upload_tickets(
            token=bob["team_token"],
            worker_job_id=worker_job_id,
            tickets=[{"filename": "shard.smi", "upload_url": "https://x", "blob_token": "b", "size_bytes": 10}],
        )

    with pytest.raises(TeamStateError):
        store.create_team_job(
            user_id="u1",
            run_id="bad",
            payload={"KAGGLE_KEY": "secret", "files": [{"filename": "x", "size_bytes": 1}]},
        )


def test_extract_result_zip_blocks_path_traversal(tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    with pytest.raises(TeamStateError):
        extract_result_zip(zip_path, tmp_path / "out")


def test_kaggle_upload_ticket_adapter_shape():
    class FakeApi:
        def start_blob_upload(self, filename, content_length):
            return {
                "upload_url": f"https://upload.invalid/{filename}",
                "blob_token": "blob-1",
                "expires_at": "soon",
            }

    ticket = mint_upload_ticket("receptor.pdbqt", 123, api=FakeApi())
    assert ticket.as_dict() == {
        "filename": "receptor.pdbqt",
        "size_bytes": 123,
        "upload_url": "https://upload.invalid/receptor.pdbqt",
        "blob_token": "blob-1",
        "expires_at": "soon",
    }


def test_kaggle_upload_ticket_current_sdk_shape(monkeypatch):
    class Response:
        create_url = "https://upload.invalid/current"
        token = "blob-current"

    class BlobClient:
        def start_blob_upload(self, request):
            assert request.name == "notebook.ipynb"
            assert request.content_length == 321
            return Response()

    class Blobs:
        blob_api_client = BlobClient()

    class KaggleClient:
        blobs = Blobs()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeApi:
        def build_kaggle_client(self):
            return KaggleClient()

        def with_retry(self, method):
            return method

    ticket = mint_upload_ticket("notebook.ipynb", 321, api=FakeApi())

    assert ticket.upload_url == "https://upload.invalid/current"
    assert ticket.blob_token == "blob-current"
    assert ticket.filename == "notebook.ipynb"
    assert ticket.size_bytes == 321


def test_commit_dataset_from_blob_tokens_current_sdk_shape():
    from ezscreen.team.kaggle_upload_ticket import commit_dataset_from_blob_tokens

    calls = {}

    class Response:
        error = ""

    class DatasetClient:
        def create_dataset(self, request):
            calls["owner_slug"] = request.owner_slug
            calls["slug"] = request.slug
            calls["title"] = request.title
            calls["is_private"] = request.is_private
            calls["tokens"] = [item.token for item in request.files]
            return Response()

    class Datasets:
        dataset_api_client = DatasetClient()

    class KaggleClient:
        datasets = Datasets()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeApi:
        def build_kaggle_client(self):
            return KaggleClient()

        def with_retry(self, method):
            return method

    dataset_ref = commit_dataset_from_blob_tokens(
        username="worker-user",
        slug="ezscreen-job",
        title="ezscreen job",
        files=[
            {"filename": "receptor.pdbqt", "blob_token": "blob-rec"},
            {"filename": "shard_000.smi", "blob_token": "blob-shard"},
        ],
        api=FakeApi(),
    )

    assert dataset_ref == "worker-user/ezscreen-job"
    assert calls == {
        "owner_slug": "worker-user",
        "slug": "ezscreen-job",
        "title": "ezscreen job",
        "is_private": True,
        "tokens": ["blob-rec", "blob-shard"],
    }

def test_direct_upload_uses_http_put(monkeypatch, tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    calls = []

    class Response:
        def raise_for_status(self):
            calls.append("raised")

    def fake_put(url, data, timeout):
        calls.append((url, data.read(), timeout))
        return Response()

    monkeypatch.setattr("ezscreen.team.kaggle_upload_ticket.requests.put", fake_put)
    upload_file_to_signed_url(payload, "https://upload.invalid/blob")

    assert calls == [("https://upload.invalid/blob", b"abc", 300), "raised"]

def test_resolve_worker_kaggle_credentials_reads_local_without_mutating(tmp_path):
    from ezscreen.team.local_credentials import resolve_worker_kaggle_credentials

    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text('{"username": "local_user", "key": "local_key"}', encoding="utf-8")
    creds = {"kaggle_json_path": str(kaggle_json)}
    before = kaggle_json.read_text(encoding="utf-8")

    resolved = resolve_worker_kaggle_credentials(use_local=True, creds=creds)

    assert resolved.username == "local_user"
    assert resolved.key == "local_key"
    assert resolved.source == "local ezscreen kaggle.json"
    assert resolved.path == kaggle_json
    assert kaggle_json.read_text(encoding="utf-8") == before


def test_resolve_worker_kaggle_credentials_prefers_explicit_key(tmp_path):
    from ezscreen.team.local_credentials import resolve_worker_kaggle_credentials

    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text('{"username": "local_user", "key": "local_key"}', encoding="utf-8")

    resolved = resolve_worker_kaggle_credentials(
        kaggle_username="explicit_user",
        kaggle_key="explicit_key",
        use_local=True,
        creds={"kaggle_json_path": str(kaggle_json)},
    )

    assert resolved.username == "explicit_user"
    assert resolved.key == "explicit_key"
    assert resolved.source == "explicit"
    assert resolved.path is None


def test_resolve_worker_kaggle_credentials_reads_saved_team_account(monkeypatch, tmp_path):
    from ezscreen.team.local_credentials import resolve_worker_kaggle_credentials

    kaggle_json = tmp_path / "team-kaggle.json"
    kaggle_json.write_text('{"username": "team_user", "key": "team_key"}', encoding="utf-8")
    monkeypatch.setattr(
        "ezscreen.auth.list_team_accounts",
        lambda _creds: [
            {
                "name": "Worker A",
                "username": "team_user",
                "kaggle_json_path": str(kaggle_json),
            }
        ],
    )

    resolved = resolve_worker_kaggle_credentials(team_account_name="Worker A", creds={})

    assert resolved.username == "team_user"
    assert resolved.key == "team_key"
    assert resolved.source == "saved team account: Worker A"
    assert resolved.path == kaggle_json

def test_worker_selects_explicit_worker_job():
    from ezscreen.team.worker import _select_job

    class FakeClient:
        def request(self, method, path):
            return {
                "jobs": [
                    {"worker_job_id": "wjob_a"},
                    {"worker_job_id": "wjob_b"},
                ]
            }

    assert _select_job(FakeClient(), "wjob_b")["worker_job_id"] == "wjob_b"
    with pytest.raises(SystemExit):
        _select_job(FakeClient())
    with pytest.raises(SystemExit):
        _select_job(FakeClient(), "missing")


def test_prepare_upload_reports_adapter_unavailable(monkeypatch):
    import argparse

    from ezscreen.team.kaggle_upload_ticket import UploadTicketUnavailable
    from ezscreen.team.worker import prepare_upload

    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            calls.append((method, path, payload))
            if path == "/v1/worker/jobs":
                return {
                    "jobs": [
                        {
                            "worker_job_id": "wjob_1",
                            "required_files": [{"filename": "shard.smi", "size_bytes": 12}],
                        }
                    ]
                }
            return {"ok": True}

    def unavailable(filename, size_bytes):
        raise UploadTicketUnavailable("upload_ticket_unavailable: missing")

    monkeypatch.setattr("ezscreen.team.worker.mint_upload_ticket", unavailable)

    rc = prepare_upload(argparse.Namespace(worker_job_id="wjob_1"), FakeClient())

    assert rc == 2
    assert calls[-1] == (
        "POST",
        "/v1/worker/status",
        {
            "worker_job_id": "wjob_1",
            "status": "failed",
            "payload": {
                "error_type": "upload_ticket_unavailable",
                "message": "upload_ticket_unavailable: missing",
            },
        },
    )



def test_kaggle_dataset_slug_sanitizes_team_and_worker_ids():
    from ezscreen.team.worker import _kaggle_dataset_slug

    assert _kaggle_dataset_slug('tjob_16f73f7d6a644c0e', 'wjob_1578f745788f4ec5') == (
        'ezscreen-tjob-16f73f7d6a644c0e-wjob-1578f745788f4ec5'
    )



def test_kaggle_dataset_title_stays_within_kaggle_limit():
    from ezscreen.team.worker import _kaggle_dataset_title

    title = _kaggle_dataset_title('ezscreen-tjob-16f73f7d6a644c0e-wjob-1578f745788f4ec5')
    assert len(title) <= 50
    assert title.startswith('ezscreen ')



def test_run_kaggle_uses_sanitized_dataset_slug(monkeypatch):
    import argparse
    import os

    from ezscreen.team import worker as worker_mod

    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            calls.append((method, path, payload))
            if path == '/v1/worker/jobs':
                return {
                    'jobs': [
                        {
                            'worker_job_id': 'wjob_1578f745788f4ec5',
                            'team_job_id': 'tjob_16f73f7d6a644c0e',
                            'required_files': [],
                        }
                    ]
                }
            if path == '/v1/worker/upload-tickets?worker_job_id=wjob_1578f745788f4ec5':
                return {
                    'tickets': [
                        {'filename': 'shard_001.smi', 'blob_token': 'blob-1'},
                    ]
                }
            return {'ok': True}

    def fake_commit_dataset_from_blob_tokens(**kwargs):
        calls.append(('commit', kwargs))
        return f"{kwargs['username']}/{kwargs['slug']}"

    monkeypatch.setattr(worker_mod, 'commit_dataset_from_blob_tokens', fake_commit_dataset_from_blob_tokens)
    monkeypatch.setenv('KAGGLE_USERNAME', 'worker-user')

    rc = worker_mod.run_kaggle(argparse.Namespace(worker_job_id='wjob_1578f745788f4ec5'), FakeClient())

    assert rc == 0
    commit = next(item for item in calls if item[0] == 'commit')[1]
    assert commit['slug'] == 'ezscreen-tjob-16f73f7d6a644c0e-wjob-1578f745788f4ec5'


def test_run_kaggle_passes_worker_accelerator_to_push_kernel(monkeypatch):
    import argparse
    import os

    from ezscreen.team import worker as worker_mod

    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            if path == '/v1/worker/jobs':
                return {
                    'jobs': [
                        {
                            'worker_job_id': 'wjob_1578f745788f4ec5',
                            'team_job_id': 'tjob_16f73f7d6a644c0e',
                            'required_files': [],
                            'payload': {'accelerator': 'nvidiaTeslaT4'},
                        }
                    ]
                }
            if path == '/v1/worker/upload-tickets?worker_job_id=wjob_1578f745788f4ec5':
                return {'tickets': [{'filename': 'shard_001.smi', 'blob_token': 'blob-1'}]}
            calls.append((method, path, payload))
            return {'ok': True}

    monkeypatch.setattr(worker_mod, 'commit_dataset_from_blob_tokens', lambda **kwargs: f"{kwargs['username']}/{kwargs['slug']}")
    monkeypatch.setattr(worker_mod, '_wait_for_dataset', lambda _ref: None)
    monkeypatch.setattr(worker_mod, '_download_dataset_file', lambda *_args, **_kwargs: Path('team_notebook.ipynb'))
    monkeypatch.setattr(worker_mod, '_finalize_kernel_run', lambda *args, **kwargs: {'status': 'complete'})

    def fake_push_kernel(**kwargs):
        calls.append(('push', kwargs))
        return f"{kwargs['username']}/{kwargs['run_id']}"

    monkeypatch.setattr(worker_mod, 'push_kernel', fake_push_kernel)
    monkeypatch.setenv('KAGGLE_USERNAME', 'worker-user')

    rc = worker_mod.run_kaggle(argparse.Namespace(worker_job_id='wjob_1578f745788f4ec5'), FakeClient())

    assert rc == 0
    push = next(item for item in calls if item[0] == 'push')[1]
    assert push['accelerator'] == 'nvidiaTeslaT4'


def test_worker_runtime_files_include_notebook_renderer_and_template():
    from ezscreen.team.github import _worker_runtime_files

    files = _worker_runtime_files()

    assert "ezscreen/backends/kaggle/runner.py" in files
    assert "ezscreen/backends/kaggle/templates/vina_shard.ipynb.j2" in files
    assert "jinja2>=3.1.4" in files["pyproject.toml"]
    assert 'templates/*.j2' in files["pyproject.toml"]
    assert '\"ezscreen.tui\" = [\"theme.tcss\"]' in files["pyproject.toml"]


def test_pyproject_includes_runtime_package_data_contract():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.hatch.build.targets.wheel.force-include]" in pyproject
    assert '"ezscreen/tui/theme.tcss" = "ezscreen/tui/theme.tcss"' in pyproject


def test_acknowledge_upload_queues_run_once_when_worker_uploads_complete(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    team = store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    connected = store.connect_worker("u1", github_repo="alice/worker")
    job = store.create_team_job(
        user_id="u1",
        run_id="ezs-team-ack",
        payload={
            "files": [
                {"filename": "receptor.pdbqt", "size_bytes": 10, "role": "receptor"},
                {"filename": "shard_000.smi", "size_bytes": 5, "role": "ligand_shard"},
            ]
        },
    )
    worker_job_id = job["worker_jobs"][0]["worker_job_id"]
    store.record_upload_tickets(
        token=connected["team_token"],
        worker_job_id=worker_job_id,
        tickets=[
            {"filename": "receptor.pdbqt", "size_bytes": 10, "upload_url": "https://upload.invalid/rec", "blob_token": "blob-rec"},
            {"filename": "shard_000.smi", "size_bytes": 5, "upload_url": "https://upload.invalid/shard", "blob_token": "blob-shard"},
        ],
    )
    tickets = store.list_upload_tickets(user_id="u1", team_job_id=job["team_job_id"])

    first = store.acknowledge_upload(user_id="u1", ticket_id=tickets[0]["ticket_id"])
    assert first == {"ok": True}

    second = store.acknowledge_upload(user_id="u1", ticket_id=tickets[1]["ticket_id"])
    assert second == {"ok": True}
    assert store.list_worker_jobs(connected["team_token"])[0]["status"] == "queued_run"

    repeated = store.acknowledge_upload(user_id="u1", ticket_id=tickets[1]["ticket_id"])
    assert repeated == {"ok": True}


def test_create_team_job_carries_accelerator_into_worker_payload(tmp_path):
    store = TeamStore(tmp_path / "team.db")
    store.create_team(owner_name="alice", owner_user_id="u1", team_name="Lab")
    connected = store.connect_worker("u1", github_repo="alice/worker")

    job = store.create_team_job(
        user_id="u1",
        run_id="ezs-team-accel",
        payload={
            "accelerator": "nvidiaTeslaT4",
            "files": [
                {"filename": "receptor.pdbqt", "size_bytes": 10, "role": "receptor"},
                {"filename": "shard_000.smi", "size_bytes": 5, "role": "ligand_shard"},
            ],
        },
    )

    worker_job = store.list_worker_jobs(connected["team_token"])[0]
    assert worker_job["worker_job_id"] == job["worker_jobs"][0]["worker_job_id"]
    assert worker_job["payload"]["accelerator"] == "nvidiaTeslaT4"


def test_team_http_server_control_plane_flow(tmp_path):
    import json
    import threading
    import urllib.request

    from ezscreen.team.client import TeamClient
    import ezscreen.team.server as team_server_mod
    from ezscreen.team.server import build_server


    server = build_server(
        host="127.0.0.1",
        port=0,
        db_path=tmp_path / "team.db",
        storage_root=tmp_path / "results",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        base = f"http://{host}:{port}"
        client = TeamClient(base)
        user = server.RequestHandlerClass.store.ensure_github_user(github_id=456, login="alice")
        session = server.RequestHandlerClass.store.create_session(
            user_id=user["user_id"], provider="github", provider_login="alice"
        )
        signed_client = TeamClient(base, token=session["session_token"])
        session_payload = signed_client.session()
        assert session_payload["user_id"] == "gh_456"
        logout_payload = signed_client.logout()
        assert logout_payload["ok"] is True
        with pytest.raises(Exception):
            signed_client.session()

        second_session = server.RequestHandlerClass.store.create_session(
            user_id=user["user_id"], provider="github", provider_login="alice"
        )
        signed_client = TeamClient(base, token=second_session["session_token"])

        team = client.create_team(owner_name="alice", team_name="Lab", github_repo="alice/worker")
        signed_team = signed_client.create_team(owner_name="signed alice", team_name="Signed Lab", github_repo="")
        assert signed_team["owner_user_id"] == "gh_456"
        assert signed_client.status("gh_456")["members"][0]["display_name"] == "alice"

        job = client.request(
            "POST",
            "/v1/jobs",
            {
                "user_id": team["owner_user_id"],
                "run_id": "ezs-http-test",
                "payload": {"files": [{"filename": "shard.smi", "size_bytes": 5}]},
            },
        )
        worker_job_id = job["worker_jobs"][0]["worker_job_id"]
        worker = TeamClient(base, token=team["team_token"])
        jobs = worker.request("GET", "/v1/worker/jobs")["jobs"]
        assert jobs[0]["worker_job_id"] == worker_job_id

        worker.request(
            "POST",
            "/v1/worker/upload-tickets",
            {
                "worker_job_id": worker_job_id,
                "tickets": [
                    {
                        "filename": "shard.smi",
                        "size_bytes": 5,
                        "upload_url": "https://upload.invalid/shard.smi",
                        "blob_token": "blob-token",
                    }
                ],
            },
        )
        tickets = client.request(
            "GET",
            f"/v1/jobs/{job['team_job_id']}/tickets?user_id={team['owner_user_id']}",
        )["tickets"]
        assert tickets[0]["filename"] == "shard.smi"
        assert tickets[0]["blob_token"] == "blob-token"

        ack = client.request(
            "POST",
            "/v1/uploads/ack",
            {"user_id": team["owner_user_id"], "ticket_id": tickets[0]["ticket_id"]},
        )
        assert ack == {"ok": True}

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("scores.csv", "ligand,score\nL1,-7.0\n")
        req = urllib.request.Request(
            f"{base}/v1/worker/result/{worker_job_id}",
            data=zip_buf.getvalue(),
            method="PUT",
            headers={"Authorization": f"Bearer {team['team_token']}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["ok"] is True
        assert Path(payload["path"]).exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_merge_team_result_zips_namespaces_colliding_worker_ids(tmp_path):
    import csv

    from ezscreen.team.coordinator import merge_team_result_zips

    def write_worker_zip(path: Path, score: str, name: str, smiles: str) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "scores.csv",
                f"ligand,score,receptor_variant\nnb0_lig_00000,{score},default\n",
            )
            zf.writestr(
                "index.csv",
                f"ligand,name,smiles,conformer_qc\nnb0_lig_00000,{name},{smiles},\n",
            )
            zf.writestr(
                "poses.sdf",
                "nb0_lig_00000\n  ezscreen\n\n0  0  0  0  0  0            999 V2000\nM  END\n$$$$\n",
            )

    zip_a = tmp_path / "worker_a.zip"
    zip_b = tmp_path / "worker_b.zip"
    write_worker_zip(zip_a, "-7.1", "A", "CCO")
    write_worker_zip(zip_b, "-6.8", "B", "CCN")

    merged = merge_team_result_zips([zip_a, zip_b], tmp_path / "merged", engine="unidock")

    with merged["scores_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"A", "B"}
    assert all(row["ligand"].startswith("worker_") for row in rows)

def test_team_join_cli_uses_local_kaggle_for_worker_secrets(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from ezscreen import auth
    from ezscreen.cli import app
    from ezscreen.team.client import TeamClient
    import ezscreen.team.github as github_mod

    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text('{"username": "local_user", "key": "local_key"}', encoding="utf-8")
    monkeypatch.setattr(auth, "load_credentials", lambda: {"kaggle_json_path": str(kaggle_json)})
    monkeypatch.setattr("ezscreen.team.client.LOCAL_STATE_PATH", tmp_path / "client.json")

    joined_payloads = []
    configured_payloads = []

    def fake_join_team(self, *, invite_code, user_name, github_repo, user_id=None):
        joined_payloads.append(
            {"invite_code": invite_code, "user_name": user_name, "github_repo": github_repo}
        )
        return {"team_id": "team_1", "user_id": "u_1", "member_id": "m_1", "team_token": "team_token"}

    def fake_configure_worker_repo(**kwargs):
        configured_payloads.append(kwargs)

    monkeypatch.setattr(TeamClient, "join_team", fake_join_team)
    monkeypatch.setattr(github_mod, "configure_worker_repo", fake_configure_worker_repo)
    monkeypatch.setattr(github_mod, "provision_managed_worker_repo", lambda **kwargs: {"repo": "alice/ezscreen-team-worker", "created": True})

    result = CliRunner().invoke(
        app,
        [
            "team",
            "join",
            "--server-url",
            "http://team.invalid",
            "--invite",
            "invite_1",
            "--user",
            "alice",
            "--use-local-kaggle",
        ],
        env={"GITHUB_TOKEN": "gh_token"},
    )

    assert result.exit_code == 0
    assert joined_payloads == [
        {"invite_code": "invite_1", "user_name": "alice", "github_repo": "alice/ezscreen-team-worker"}
    ]
    assert configured_payloads == [
        {
            "repo": "alice/ezscreen-team-worker",
            "github_token": "gh_token",
            "kaggle_username": "local_user",
            "kaggle_key": "local_key",
            "team_token": "team_token",
            "server_url": "http://team.invalid",
        }
    ]
    assert "local_key" not in result.output
    assert kaggle_json.read_text(encoding="utf-8") == '{"username": "local_user", "key": "local_key"}'


def test_team_connect_worker_cli_updates_repo_and_secrets(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from ezscreen import auth
    from ezscreen.cli import app
    from ezscreen.team.client import TeamClient
    import ezscreen.team.github as github_mod

    kaggle_json = tmp_path / "kaggle.json"
    kaggle_json.write_text('{"username": "local_user", "key": "local_key"}', encoding="utf-8")
    monkeypatch.setattr(auth, "load_credentials", lambda: {"kaggle_json_path": str(kaggle_json)})
    monkeypatch.setattr("ezscreen.team.client.LOCAL_STATE_PATH", tmp_path / "client.json")
    from ezscreen.team.client import save_local_state

    save_local_state(
        {
            "server_url": "http://team.invalid",
            "session_token": "session-token",
            "user_id": "u_1",
            "user_name": "alice",
        }
    )

    connect_calls = []
    configure_calls = []

    def fake_connect_worker(self, *, github_repo, kaggle_username="", workflow_file="team-worker.yml"):
        connect_calls.append(
            {
                "github_repo": github_repo,
                "kaggle_username": kaggle_username,
                "workflow_file": workflow_file,
                "token": self.token,
            }
        )
        return {"member_id": "m_1", "github_repo": github_repo, "workflow_file": workflow_file, "team_token": "team-token"}

    def fake_configure_worker_repo(**kwargs):
        configure_calls.append(kwargs)

    monkeypatch.setattr(TeamClient, "connect_worker", fake_connect_worker)
    monkeypatch.setattr(github_mod, "configure_worker_repo", fake_configure_worker_repo)
    monkeypatch.setattr(github_mod, "provision_managed_worker_repo", lambda **kwargs: {"repo": "alice/ezscreen-team-worker", "created": True})

    result = CliRunner().invoke(
        app,
        [
            "team",
            "connect-worker",
            "--use-local-kaggle",
        ],
        env={"GITHUB_TOKEN": "gh_token"},
    )

    assert result.exit_code == 0
    assert connect_calls == [
        {
            "github_repo": "alice/ezscreen-team-worker",
            "kaggle_username": "local_user",
            "workflow_file": "team-worker.yml",
            "token": "session-token",
        }
    ]
    assert configure_calls == [
        {
            "repo": "alice/ezscreen-team-worker",
            "github_token": "gh_token",
            "kaggle_username": "local_user",
            "kaggle_key": "local_key",
            "team_token": "team-token",
            "server_url": "http://team.invalid",
        }
    ]



def test_team_logout_cli_revokes_server_session(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from ezscreen.cli import app
    from ezscreen.team import client as team_client

    monkeypatch.setattr(team_client, "LOCAL_STATE_PATH", tmp_path / "client.json")
    team_client.save_local_state(
        {
            "server_url": "http://team.invalid",
            "session_token": "session-token",
            "user_name": "alice",
            "github_login": "octocat",
        }
    )

    logout_calls = []

    def fake_logout(self):
        logout_calls.append({"server_url": self.server_url, "token": self.token})
        return {"ok": True}

    monkeypatch.setattr(team_client.TeamClient, "logout", fake_logout)

    result = CliRunner().invoke(app, ["team", "logout"])

    assert result.exit_code == 0
    assert logout_calls == [{"server_url": "http://team.invalid", "token": "session-token"}]
    assert team_client.load_local_state() == {"server_url": "http://team.invalid", "user_name": "alice"}


def test_team_disconnect_worker_cli_clears_repo(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from ezscreen.cli import app
    from ezscreen.team import client as team_client

    monkeypatch.setattr(team_client, "LOCAL_STATE_PATH", tmp_path / "client.json")
    team_client.save_local_state(
        {
            "server_url": "http://team.invalid",
            "session_token": "session-token",
            "user_id": "u_1",
            "user_name": "alice",
            "github_repo": "alice/worker",
        }
    )

    calls = []

    def fake_disconnect_worker(self, *, user_id=None):
        calls.append({"server_url": self.server_url, "token": self.token, "user_id": user_id})
        return {"member_id": "m_1", "disconnected": True}

    monkeypatch.setattr(team_client.TeamClient, "disconnect_worker", fake_disconnect_worker)

    result = CliRunner().invoke(app, ["team", "disconnect-worker"])

    assert result.exit_code == 0
    assert calls == [{"server_url": "http://team.invalid", "token": "session-token", "user_id": "u_1"}]
    assert team_client.load_local_state()["github_repo"] == ""


def test_provision_managed_worker_repo_rejects_explicit_nonmanaged_repo(monkeypatch):
    import ezscreen.team.github as github_mod

    def fake_request(method, path, *, token, payload=None):
        if path == "/user":
            return {"login": "alice"}
        if path == "/repos/alice/existing-repo":
            return {"full_name": "alice/existing-repo", "private": True}
        raise AssertionError(path)

    monkeypatch.setattr(github_mod, "_github_request", fake_request)
    monkeypatch.setattr(
        github_mod,
        "_github_request_optional",
        lambda method, path, **kwargs: fake_request(method, path, **kwargs) if path == "/repos/alice/existing-repo" else None,
    )
    monkeypatch.setattr(github_mod, "_get_repo_file", lambda repo, path, token: None)

    with pytest.raises(github_mod.GitHubError):
        github_mod.provision_managed_worker_repo(github_token="gh_token", repo_name="existing-repo")



def test_worker_auto_processes_pending_statuses(monkeypatch):
    import argparse
    from ezscreen.team import worker as worker_mod

    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            if path == "/v1/worker/jobs":
                return {
                    "jobs": [
                        {"worker_job_id": "w1", "status": "needs_upload_tickets", "required_files": []},
                        {"worker_job_id": "w2", "status": "queued_run", "required_files": []},
                        {"worker_job_id": "w3", "status": "collecting", "required_files": []},
                    ]
                }
            raise AssertionError(path)

    monkeypatch.setattr(worker_mod, "prepare_upload", lambda args, client: calls.append(("prepare", args.worker_job_id)) or 0)
    monkeypatch.setattr(worker_mod, "run_kaggle", lambda args, client: calls.append(("run", args.worker_job_id)) or 0)
    monkeypatch.setattr(worker_mod, "collect", lambda args, client: calls.append(("collect", args.worker_job_id)) or 0)

    rc = worker_mod.auto(argparse.Namespace(worker_job_id=None), FakeClient())

    assert rc == 0
    assert calls == [("prepare", "w1"), ("run", "w2"), ("collect", "w3")]



def test_mark_worker_job_updates_parent_team_job_status(tmp_path):
    store = TeamStore(tmp_path / 'team.db')
    store.create_team(owner_name='alice', owner_user_id='u1', team_name='Lab')
    connected = store.connect_worker('u1', github_repo='alice/worker')
    job = store.create_team_job(
        user_id='u1',
        run_id='ezs-parent-status',
        payload={'files': [{'filename': 'shard.smi', 'size_bytes': 1}]},
    )
    worker_job_id = job['worker_jobs'][0]['worker_job_id']
    store.mark_worker_job(
        token=connected['team_token'],
        worker_job_id=worker_job_id,
        status='running',
        payload={'kernel_ref': 'alice/kernel'},
    )
    with store.connection() as conn:
        row = conn.execute('SELECT status, completed_at FROM team_jobs WHERE team_job_id = ?', (job['team_job_id'],)).fetchone()
    assert row['status'] == 'running'
    assert row['completed_at'] is None
    store.mark_worker_job(
        token=connected['team_token'],
        worker_job_id=worker_job_id,
        status='failed',
        payload={'error_type': 'kernel_failed'},
    )
    with store.connection() as conn:
        row = conn.execute('SELECT status, completed_at FROM team_jobs WHERE team_job_id = ?', (job['team_job_id'],)).fetchone()
    assert row['status'] == 'failed'
    assert row['completed_at'] is not None


def test_save_result_zip_marks_parent_team_job_complete(tmp_path):
    store = TeamStore(tmp_path / 'team.db')
    store.create_team(owner_name='alice', owner_user_id='u1', team_name='Lab')
    connected = store.connect_worker('u1', github_repo='alice/worker')
    job = store.create_team_job(
        user_id='u1',
        run_id='ezs-parent-complete',
        payload={'files': [{'filename': 'shard.smi', 'size_bytes': 1}]},
    )
    worker_job_id = job['worker_jobs'][0]['worker_job_id']
    result = store.save_result_zip(
        token=connected['team_token'],
        worker_job_id=worker_job_id,
        data=(b'PK\x05\x06' + (b'\x00' * 18)),
        storage_root=tmp_path / 'results',
    )
    assert Path(result['path']).exists()
    with store.connection() as conn:
        team_row = conn.execute('SELECT status, completed_at FROM team_jobs WHERE team_job_id = ?', (job['team_job_id'],)).fetchone()
        worker_row = conn.execute('SELECT status_payload_json FROM worker_jobs WHERE worker_job_id = ?', (worker_job_id,)).fetchone()
    assert team_row['status'] == 'complete'
    assert team_row['completed_at'] is not None
    assert 'result_zip_path' in worker_row['status_payload_json']


def test_run_kaggle_pushes_kernel_and_uploads_result(monkeypatch, tmp_path):
    import argparse

    from ezscreen.team import worker as worker_mod

    notebook = tmp_path / 'team_notebook.ipynb'
    notebook.write_text('{}', encoding='utf-8')
    out_zip = tmp_path / 'output.zip'
    out_zip.write_bytes(b'zip-bytes')
    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            calls.append((method, path, payload))
            if path == '/v1/worker/jobs':
                return {
                    'jobs': [
                        {
                            'worker_job_id': 'wjob_1',
                            'team_job_id': 'tjob_1',
                            'required_files': [],
                            'payload': {},
                        }
                    ]
                }
            if path == '/v1/worker/upload-tickets?worker_job_id=wjob_1':
                return {'tickets': [{'filename': 'team_notebook.ipynb', 'blob_token': 'blob-1'}]}
            return {'ok': True}

        def upload_result_zip(self, *, worker_job_id, data):
            calls.append(('PUT', worker_job_id, data))
            return {'ok': True}

    monkeypatch.setenv('KAGGLE_USERNAME', 'worker-user')
    monkeypatch.setattr(worker_mod, 'commit_dataset_from_blob_tokens', lambda **kwargs: 'worker-user/ezscreen-team-job')
    monkeypatch.setattr(worker_mod, '_wait_for_dataset', lambda dataset_ref: None)
    monkeypatch.setattr(worker_mod, '_download_dataset_file', lambda dataset_ref, filename, dest_dir: notebook)
    monkeypatch.setattr(worker_mod, 'push_kernel', lambda **kwargs: 'worker-user/ezs-team-123')
    monkeypatch.setattr(worker_mod, 'poll_until_done', lambda **kwargs: {'status': 'complete', 'retry_count': 0, 'error_type': None})
    monkeypatch.setattr(worker_mod, '_download_kernel_output_zip', lambda kernel_ref, work_dir: out_zip)

    rc = worker_mod.run_kaggle(argparse.Namespace(worker_job_id='wjob_1'), FakeClient())

    assert rc == 0
    running = [item for item in calls if item[1] == '/v1/worker/status']
    assert running[-1][2]['status'] == 'running'
    assert running[-1][2]['payload']['kernel_ref'] == 'worker-user/ezs-team-123'
    put_call = next(item for item in calls if item[0] == 'PUT')
    assert put_call[1] == 'wjob_1'
    assert put_call[2] == b'zip-bytes'



def test_download_dataset_file_uses_dataset_ref_signature(monkeypatch, tmp_path):
    from ezscreen.team import worker as worker_mod

    calls = []

    class FakeApi:
        def dataset_download_file(self, dataset, file_name, path=None, force=False, quiet=True, licenses=None):
            calls.append({
                'dataset': dataset,
                'file_name': file_name,
                'path': path,
                'force': force,
                'quiet': quiet,
            })
            target = Path(path) / file_name
            target.write_text('{}', encoding='utf-8')

    monkeypatch.setattr(worker_mod, '_api', lambda: FakeApi())

    result = worker_mod._download_dataset_file('worker-user/ezscreen-dataset', 'team_notebook.ipynb', tmp_path)

    assert result == tmp_path / 'team_notebook.ipynb'
    assert calls == [{
        'dataset': 'worker-user/ezscreen-dataset',
        'file_name': 'team_notebook.ipynb',
        'path': str(tmp_path),
        'force': True,
        'quiet': True,
    }]


def test_collect_can_resume_from_dataset_ref_only(monkeypatch, tmp_path):
    import argparse

    from ezscreen.team import worker as worker_mod

    notebook = tmp_path / 'team_notebook.ipynb'
    notebook.write_text('{}', encoding='utf-8')
    out_zip = tmp_path / 'output.zip'
    out_zip.write_bytes(b'zip-bytes')
    calls = []

    class FakeClient:
        def request(self, method, path, payload=None):
            calls.append((method, path, payload))
            if path == '/v1/worker/jobs':
                return {
                    'jobs': [
                        {
                            'worker_job_id': 'wjob_2',
                            'team_job_id': 'tjob_2',
                            'required_files': [],
                            'payload': {'dataset_ref': 'worker-user/ezscreen-team-job'},
                        }
                    ]
                }
            return {'ok': True}

        def upload_result_zip(self, *, worker_job_id, data):
            calls.append(('PUT', worker_job_id, data))
            return {'ok': True}

    monkeypatch.setenv('KAGGLE_USERNAME', 'worker-user')
    monkeypatch.setattr(worker_mod, '_download_dataset_file', lambda dataset_ref, filename, dest_dir: notebook)
    monkeypatch.setattr(worker_mod, 'push_kernel', lambda **kwargs: 'worker-user/ezs-team-456')
    monkeypatch.setattr(worker_mod, 'poll_until_done', lambda **kwargs: {'status': 'complete', 'retry_count': 0, 'error_type': None})
    monkeypatch.setattr(worker_mod, '_download_kernel_output_zip', lambda kernel_ref, work_dir: out_zip)

    rc = worker_mod.collect(argparse.Namespace(worker_job_id='wjob_2'), FakeClient())

    assert rc == 0
    collecting = [item for item in calls if item[1] == '/v1/worker/status']
    assert collecting[-1][2]['status'] == 'collecting'
    assert collecting[-1][2]['payload']['kernel_ref'] == 'worker-user/ezs-team-456'
    put_call = next(item for item in calls if item[0] == 'PUT')
    assert put_call[1] == 'wjob_2'


def test_list_team_job_workers_and_read_result_zip(tmp_path):
    store = TeamStore(tmp_path / 'team.db')
    store.create_team(owner_name='alice', owner_user_id='u1', team_name='Lab')
    connected = store.connect_worker('u1', github_repo='alice/worker')
    job = store.create_team_job(
        user_id='u1',
        run_id='ezs-results',
        payload={'files': [{'filename': 'shard.smi', 'size_bytes': 1}]},
    )
    worker_job_id = job['worker_jobs'][0]['worker_job_id']
    payload = b'PK\x05\x06' + (b'\x00' * 18)
    store.save_result_zip(
        token=connected['team_token'],
        worker_job_id=worker_job_id,
        data=payload,
        storage_root=tmp_path / 'results',
    )

    workers = store.list_team_job_workers(user_id='u1', team_job_id=job['team_job_id'])

    assert workers[0]['worker_job_id'] == worker_job_id
    assert workers[0]['status'] == 'complete'
    assert store.read_result_zip(user_id='u1', team_job_id=job['team_job_id'], worker_job_id=worker_job_id) == payload


def test_dispatch_team_job_workers_uses_server_response():
    from ezscreen.team import coordinator as coordinator_mod

    class FakeClient:
        def dispatch_team_job(self, *, user_id, team_job_id, mode):
            assert user_id == 'u1'
            assert team_job_id == 'tjob_1'
            assert mode == 'prepare_upload'
            return {
                'dispatched': [{'worker_job_id': 'w1', 'repo': 'alice/worker', 'mode': 'prepare_upload'}],
                'skipped': [],
                'failed': [],
                'dispatch_backend': 'github_app',
            }

    result = coordinator_mod.dispatch_team_job_workers(
        client=FakeClient(),
        user_id='u1',
        team_job={'team_job_id': 'tjob_1', 'worker_jobs': [{'worker_job_id': 'w1', 'member_id': 'mem1'}]},
    )

    assert result['dispatched'] == [{'worker_job_id': 'w1', 'repo': 'alice/worker', 'mode': 'prepare_upload'}]


def test_dispatch_team_job_runs_preserves_server_skip_reason():
    from ezscreen.team import coordinator as coordinator_mod

    class FakeClient:
        def dispatch_team_job(self, *, user_id, team_job_id, mode):
            assert user_id == 'u1'
            assert team_job_id == 'tjob_1'
            assert mode == 'run_kaggle'
            return {
                'dispatched': [],
                'skipped': [{'worker_job_id': 'w1', 'repo': 'alice/worker', 'reason': 'app_installation_missing'}],
                'failed': [],
                'dispatch_backend': 'github_app',
            }

    result = coordinator_mod.dispatch_team_job_runs(
        client=FakeClient(),
        user_id='u1',
        team_job={'team_job_id': 'tjob_1', 'worker_jobs': [{'worker_job_id': 'w1', 'member_id': 'mem1'}]},
    )

    assert result['skipped'] == [{'worker_job_id': 'w1', 'repo': 'alice/worker', 'reason': 'app_installation_missing'}]


def test_connect_worker_tracks_installation_id(tmp_path):
    store = TeamStore(tmp_path / 'team.db')
    store.create_team(owner_name='alice', owner_user_id='u1', team_name='Lab')

    connected = store.connect_worker('u1', github_repo='alice/worker', github_installation_id=123456)

    assert connected['github_installation_id'] == 123456
    status = store.status('u1')
    assert status['members'][0]['github_installation_id'] == 123456


def test_team_job_dispatch_targets_include_installation_id(tmp_path):
    store = TeamStore(tmp_path / 'team.db')
    store.create_team(owner_name='alice', owner_user_id='u1', team_name='Lab')
    connected = store.connect_worker('u1', github_repo='alice/worker', github_installation_id=987)
    job = store.create_team_job(
        user_id='u1',
        run_id='ezs-dispatch-targets',
        payload={'files': [{'filename': 'shard.smi', 'size_bytes': 1}]},
    )

    targets = store.team_job_dispatch_targets(user_id='u1', team_job_id=job['team_job_id'])

    assert targets == [{
        'worker_job_id': job['worker_jobs'][0]['worker_job_id'],
        'member_id': connected['member_id'],
        'status': 'needs_upload_tickets',
        'github_repo': 'alice/worker',
        'github_installation_id': 987,
        'workflow_file': 'team-worker.yml',
    }]


def test_dispatch_worker_workflow_via_app_uses_installation_token(monkeypatch):
    import ezscreen.team.github as github_mod

    calls = []

    monkeypatch.setattr(
        github_mod,
        'github_app_installation_token',
        lambda **kwargs: calls.append(('token', kwargs)) or 'inst-token',
    )
    monkeypatch.setattr(
        github_mod,
        'dispatch_worker_workflow',
        lambda **kwargs: calls.append(('dispatch', kwargs)),
    )

    github_mod.dispatch_worker_workflow_via_app(
        repo='alice/worker',
        app_id='123',
        private_key_pem='-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----\n',
        installation_id=456,
        mode='prepare_upload',
        worker_job_id='wjob_1',
    )

    assert calls[0] == (
        'token',
        {
            'app_id': '123',
            'private_key_pem': '-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----\n',
            'installation_id': 456,
            'permissions': {'actions': 'write', 'contents': 'read', 'metadata': 'read'},
        },
    )
    assert calls[1] == (
        'dispatch',
        {
            'repo': 'alice/worker',
            'token': 'inst-token',
            'mode': 'prepare_upload',
            'worker_job_id': 'wjob_1',
            'workflow_file': 'team-worker.yml',
        },
    )


def test_refresh_status_installations_fills_missing_ids():
    from ezscreen.team.server import refresh_status_installations

    persisted = []
    status = {
        'team': {'team_id': 'team_1'},
        'members': [
            {'member_id': 'mem1', 'github_repo': 'alice/worker', 'github_installation_id': None},
            {'member_id': 'mem2', 'github_repo': 'bob/worker', 'github_installation_id': 22},
            {'member_id': 'mem3', 'github_repo': '', 'github_installation_id': None},
        ],
    }

    result = refresh_status_installations(
        status=status,
        app_config={'app_id': '123', 'private_key': 'pem', 'install_url': 'https://github.com/apps/x/installations/new'},
        resolve_installation_id=lambda repo: {'alice/worker': 11}.get(repo),
        persist_installation_id=lambda **kwargs: persisted.append(kwargs),
    )

    assert result['members'][0]['github_installation_id'] == 11
    assert result['members'][1]['github_installation_id'] == 22
    assert persisted == [{'member_id': 'mem1', 'github_installation_id': 11}]


def test_refresh_status_installations_ignores_lookup_errors():
    from ezscreen.team.server import refresh_status_installations

    status = {
        'team': {'team_id': 'team_1'},
        'members': [
            {'member_id': 'mem1', 'github_repo': 'alice/worker', 'github_installation_id': None},
        ],
    }

    result = refresh_status_installations(
        status=status,
        app_config={'app_id': '123', 'private_key': 'pem', 'install_url': 'https://github.com/apps/x/installations/new'},
        resolve_installation_id=lambda repo: (_ for _ in ()).throw(RuntimeError('missing dependency')),
        persist_installation_id=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    assert result['members'][0]['github_installation_id'] is None

def test_cli_help_hides_legacy_tui_subcommand():
    from typer.testing import CliRunner

    from ezscreen.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "auth" in result.output
    assert "status" in result.output
    assert "tui       Open the full-screen interactive TUI." not in result.output



