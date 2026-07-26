from pathlib import Path

import pytest

from ezscreen.benchmark import kaggle_notebook


def test_submit_notebook_uses_cpu_backend_and_supervisor(
    tmp_path: Path, monkeypatch
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    campaign = tmp_path / "campaign"
    account = {
        "name": "primary",
        "kaggle_json_path": str(tmp_path / "kaggle.json"),
    }
    monkeypatch.setattr(
        kaggle_notebook, "_healthy_account", lambda _name: (account, "owner")
    )

    calls = []
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.register_job",
        lambda _campaign, job: calls.append(("register", job)) or job,
    )
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.update_registered_job",
        lambda _campaign, job_id, **updates: calls.append(
            ("update", job_id, updates)
        ),
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._set_account_creds", lambda _path: "owner"
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._wait_for_dataset",
        lambda ref: calls.append(("ready", ref)),
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._wait_for_dataset_sync",
        lambda ref, **kwargs: calls.append(("sync", ref)),
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.dataset.upload_files_dataset",
        lambda *args, **kwargs: "owner/data",
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.kernel.push_kernel",
        lambda *args, **kwargs: calls.append(("push", kwargs)) or "owner/kernel",
    )

    result = kaggle_notebook.submit_notebook(
        campaign,
        notebook,
        [source],
        run_id="paper-cpu-smoke",
        backend="cpu",
    )

    assert result["status"] == "submitted"
    assert result["compute_kind"] == "cpu"
    assert result["planned_gpu_hours"] == 0.0
    registered = calls[0][1]
    assert registered["compute_kind"] == "cpu"
    assert registered["accelerator"] == "none"
    assert next(call for call in calls if call[0] == "push")[1]["backend"] == "cpu"


def test_submit_notebook_rejects_gpu_without_reservation(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="positive planned GPU hours"):
        kaggle_notebook.submit_notebook(
            tmp_path / "campaign",
            notebook,
            [source],
            run_id="paper-gpu",
            backend="p100",
        )


def test_submit_notebook_reuses_immutable_dataset_without_waiting(
    tmp_path: Path, monkeypatch
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    account = {
        "name": "primary",
        "kaggle_json_path": str(tmp_path / "kaggle.json"),
    }
    monkeypatch.setattr(
        kaggle_notebook, "_healthy_account", lambda _name: (account, "owner")
    )
    calls = []
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.register_job", lambda *_args: None
    )
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.update_registered_job",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._set_account_creds", lambda _path: None
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._wait_for_dataset",
        lambda ref: calls.append(("wait", ref)),
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.dataset.upload_files_dataset",
        lambda *_args, **_kwargs: calls.append(("upload",)) or "owner/new",
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.kernel.push_kernel",
        lambda *args, **kwargs: calls.append(("push", args[2])) or "owner/kernel",
    )

    result = kaggle_notebook.submit_notebook(
        tmp_path / "campaign",
        notebook,
        [source],
        run_id="paper-cpu-retry",
        backend="cpu",
        dataset_ref="owner/frozen-inputs",
    )

    assert result["dataset_ref"] == "owner/frozen-inputs"
    assert result["dataset_reused"] is True
    assert calls == [("push", "owner/frozen-inputs")]


def test_submit_notebook_marks_prior_attempt_superseded(
    tmp_path: Path, monkeypatch
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    account = {
        "name": "primary",
        "kaggle_json_path": str(tmp_path / "kaggle.json"),
    }
    monkeypatch.setattr(
        kaggle_notebook, "_healthy_account", lambda _name: (account, "owner")
    )
    updates = []
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.update_registered_job",
        lambda _campaign, job_id, **kwargs: updates.append((job_id, kwargs)),
    )
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.register_job", lambda *_args: None
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._set_account_creds", lambda _path: None
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.kernel.push_kernel",
        lambda *_args, **_kwargs: "owner/kernel",
    )

    kaggle_notebook.submit_notebook(
        tmp_path / "campaign",
        notebook,
        [source],
        run_id="retry-r7",
        job_id="job-r7",
        dataset_ref="owner/frozen",
        supersedes_job_ids=["job-r5", "job-r6"],
    )

    assert updates[:2] == [
        ("job-r5", {"status": "superseded", "superseded_by": "job-r7"}),
        ("job-r6", {"status": "superseded", "superseded_by": "job-r7"}),
    ]


def test_submit_notebook_registers_bayesbind_finalizer_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    intended = tmp_path / "intended.csv"
    intended.write_text("ligand,label\nL1,1\n", encoding="utf-8")
    account = {
        "name": "primary",
        "kaggle_json_path": str(tmp_path / "kaggle.json"),
    }
    monkeypatch.setattr(
        kaggle_notebook, "_healthy_account", lambda _name: (account, "owner")
    )
    calls = []
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.register_job",
        lambda _campaign, job: calls.append(("register", job)) or job,
    )
    monkeypatch.setattr(
        "ezscreen.benchmark.supervisor.update_registered_job",
        lambda _campaign, job_id, **updates: calls.append(
            ("update", job_id, updates)
        ),
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._set_account_creds", lambda _path: None
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.dataset.upload_files_dataset",
        lambda *_args, **_kwargs: "owner/data",
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._wait_for_dataset", lambda _ref: None
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.runner._wait_for_dataset_sync",
        lambda _ref, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ezscreen.backends.kaggle.kernel.push_kernel",
        lambda *_args, **_kwargs: "owner/kernel",
    )

    result = kaggle_notebook.submit_notebook(
        tmp_path / "campaign",
        notebook,
        [source],
        run_id="paper-cpu-calib",
        backend="cpu",
        finalizer="bayesbind-vina",
        intended_csv=intended,
    )

    registered = calls[0][1]
    assert result["status"] == "submitted"
    assert registered["finalizer"] == "bayesbind-vina"
    assert registered["intended_csv"] == str(intended.resolve())
