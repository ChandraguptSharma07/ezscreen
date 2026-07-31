from __future__ import annotations

import json
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
        def kernels_push(self, kernel_dir, acc=None):
            calls.append((Path(kernel_dir), acc))

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
    metadata = (calls[0][0] / 'kernel-metadata.json').read_text(encoding='utf-8')
    assert '"id": "worker-user/ezs-run"' in metadata
    assert calls[0][1] == 'NvidiaTeslaP100'


def test_push_kernel_bounds_long_run_id_slug(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def kernels_push(self, kernel_dir, acc=None):
            calls.append((Path(kernel_dir), acc))

    monkeypatch.setattr(kernel, '_api', lambda: FakeApi())
    notebook = tmp_path / 'run.ipynb'
    notebook.write_text('{}', encoding='utf-8')

    run_id = 'bayesbind-vina-exact-sdf-full-al1a1-human-4-501-0-001-of-002'
    ref = kernel.push_kernel(
        run_id=run_id,
        notebook_path=notebook,
        dataset_ref='user/dataset',
        username='worker-user',
        work_dir=tmp_path / 'work',
    )

    slug = ref.split('/')[1]
    assert len(slug) <= kernel.MAX_KERNEL_SLUG_LEN
    assert slug == kernel.kernel_slug(run_id)
    assert calls[0][0].name == slug
    metadata = json.loads((calls[0][0] / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert metadata['id'] == ref
    assert len(metadata['title']) <= kernel.MAX_KERNEL_SLUG_LEN


def test_push_kernel_disables_gpu_for_cpu_accelerator(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def kernels_push(self, kernel_dir, acc=None):
            calls.append((Path(kernel_dir), acc))

    monkeypatch.setattr(kernel, '_api', lambda: FakeApi())
    notebook = tmp_path / 'cpu.ipynb'
    notebook.write_text('{}', encoding='utf-8')

    kernel.push_kernel(
        run_id='ezs-cpu',
        notebook_path=notebook,
        dataset_ref='user/dataset',
        username='worker-user',
        work_dir=tmp_path / 'work',
        accelerator='none',
    )

    metadata = json.loads(
        (calls[0][0] / 'kernel-metadata.json').read_text(encoding='utf-8')
    )
    assert metadata['enable_gpu'] is False
    assert 'machine_shape' not in metadata
    assert calls[0][1] is None


def test_push_kernel_accepts_cpu_backend_option(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def kernels_push(self, kernel_dir, acc=None):
            calls.append((Path(kernel_dir), acc))

    monkeypatch.setattr(kernel, '_api', lambda: FakeApi())
    notebook = tmp_path / 'cpu.ipynb'
    notebook.write_text('{}', encoding='utf-8')
    kernel.push_kernel(
        run_id='ezs-cpu-backend',
        notebook_path=notebook,
        dataset_ref='user/dataset',
        username='worker-user',
        work_dir=tmp_path / 'work',
        backend='cpu',
    )
    metadata = json.loads(
        (calls[0][0] / 'kernel-metadata.json').read_text(encoding='utf-8')
    )
    assert metadata['enable_gpu'] is False
    assert 'machine_shape' not in metadata
    assert calls[0][1] is None


def test_unknown_kaggle_backend_is_rejected():
    import pytest

    with pytest.raises(ValueError, match='choose cpu, p100, or t4x2'):
        kernel.accelerator_for_backend('quantum')


def test_push_kernel_passes_t4_accelerator_to_save_api(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def kernels_push(self, kernel_dir, acc=None):
            calls.append((Path(kernel_dir), acc))

    monkeypatch.setattr(kernel, '_api', lambda: FakeApi())
    notebook = tmp_path / 't4.ipynb'
    notebook.write_text('{}', encoding='utf-8')

    kernel.push_kernel(
        run_id='ezs-t4',
        notebook_path=notebook,
        dataset_ref='user/dataset',
        username='worker-user',
        work_dir=tmp_path / 'work',
        accelerator='nvidiaTeslaT4',
    )

    metadata = json.loads(
        (calls[0][0] / 'kernel-metadata.json').read_text(encoding='utf-8')
    )
    assert calls[0][1] == 'NvidiaTeslaT4'
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['enable_gpu'] is True
