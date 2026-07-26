import json
import tempfile
from pathlib import Path

import pytest

from ezscreen.backends.kaggle import dataset


class _Api:
    def __init__(self) -> None:
        self.created: list[tuple[str, bool, bool]] = []

    def dataset_create_new(
        self, path: str, public: bool, quiet: bool, dir_mode: str = "skip"
    ) -> None:
        self.created.append((path, public, quiet))


def test_upload_files_dataset_stages_hashed_bundle(tmp_path: Path, monkeypatch) -> None:
    api = _Api()
    monkeypatch.setattr(dataset, '_api', lambda: api)
    monkeypatch.setattr(dataset, 'MANIFEST_PATH', tmp_path / 'manifest.json')
    a = tmp_path / 'a.txt'
    b = tmp_path / 'b.txt'
    a.write_text('a', encoding='utf-8')
    b.write_text('b', encoding='utf-8')

    ref = dataset.upload_files_dataset('paper-smoke', [a, b], 'owner', tmp_path)

    assert ref.startswith('owner/ezscreen-paper-smoke-')
    staged = Path(api.created[0][0])
    assert staged.name.startswith('ezscreen-paper-smoke-')
    assert str(staged).startswith(str(Path(tempfile.gettempdir())))
    assert (staged / 'a.txt').read_text() == 'a'
    metadata = json.loads((staged / 'dataset-metadata.json').read_text())
    assert metadata['id'] == ref
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest[str(a.resolve())] == dataset.sha256(a)
    assert len(api.created) == 1


def test_dataset_slug_is_safe_and_bounded() -> None:
    slug = dataset.dataset_slug("Very Long Run ID " * 8)
    assert len(slug) <= 50
    assert slug == dataset.dataset_slug("Very Long Run ID " * 8)
    assert slug != dataset.dataset_slug("Very Long Run ID " * 8 + "x")


def test_upload_files_dataset_rejects_sdk_error_response(
    tmp_path: Path, monkeypatch
) -> None:
    class ErrorApi:
        def dataset_create_new(self, *args, **kwargs):
            return type("Response", (), {"status": "error", "error": "title exists"})()

    monkeypatch.setattr(dataset, '_api', lambda: ErrorApi())
    monkeypatch.setattr(dataset, 'MANIFEST_PATH', tmp_path / 'manifest.json')
    source = tmp_path / 'a.txt'
    source.write_text('a')
    with pytest.raises(dataset.KaggleBadRequestError, match='title exists'):
        dataset.upload_files_dataset('paper-smoke', [source], 'owner', tmp_path)


def test_upload_files_dataset_rejects_duplicate_basenames(tmp_path: Path) -> None:
    a = tmp_path / 'x' / 'same.txt'
    b = tmp_path / 'y' / 'same.txt'
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text('a')
    b.write_text('b')
    with pytest.raises(ValueError, match='basenames'):
        dataset.upload_files_dataset('dup', [a, b], 'owner', tmp_path)
