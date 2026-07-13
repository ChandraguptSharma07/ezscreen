from __future__ import annotations

from pathlib import Path

from ezscreen.backends.kaggle import kernel


def test_delete_kernel_uses_existing_run_id_slug(monkeypatch):
    calls = []

    class FakeApi:
        def kernel_delete(self, username, slug):
            calls.append((username, slug))

    monkeypatch.setattr(kernel, '_api', lambda: FakeApi())

    kernel.delete_kernel('ezs-abc123', 'worker-user')

    assert calls == [('worker-user', 'ezs-abc123')]


def test_push_kernel_uses_run_id_as_kernel_slug(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def kernels_push(self, kernel_dir):
            calls.append(Path(kernel_dir))

    monkeypatch.setattr(kernel, '_api', lambda: FakeApi())

    notebook = tmp_path / 'run.ipynb'
    notebook.write_text('{}', encoding='utf-8')
    ref = kernel.push_kernel(
        run_id='ezs-run',
        notebook_path=notebook,
        dataset_ref='user/dataset',
        username='worker-user',
        work_dir=tmp_path / 'work',
    )

    assert ref == 'worker-user/ezs-run'
    metadata = (calls[0] / 'kernel-metadata.json').read_text(encoding='utf-8')
    assert '"id": "worker-user/ezs-run"' in metadata
