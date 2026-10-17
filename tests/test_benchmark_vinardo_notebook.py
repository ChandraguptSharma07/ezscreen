import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).parents[1]
    / 'benchmarks'
    / 'bench-20260723-vinardo-dud28'
    / 'notebooks'
    / 'vinardo_preflight.ipynb'
)

pytestmark = pytest.mark.skipif(
    not NOTEBOOK.exists(),
    reason='rendered preflight notebook is kept outside the repository',
)


def test_vinardo_preflight_notebook_is_clean_and_parseable() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert payload['metadata']['ezscreen']['accelerator'] == 'none'
    assert payload['metadata']['ezscreen']['seed'] == 20260723
    assert len(payload['cells']) >= 10
    for cell in payload['cells']:
        assert not cell.get('outputs')
        assert cell.get('execution_count') is None
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))


def test_vinardo_preflight_contains_scientific_gates() -> None:
    text = NOTEBOOK.read_text(encoding='utf-8')
    for required in (
        '--exhaustiveness',
        '--autobox_add',
        'dkoes_scoring',
        'vinardo',
        'source_hashes_match',
        'prep_coverage_at_least_95pct',
        'sam_atoms_removed',
        'rglob',
        'EMBEDDED_MANIFEST',
        'urllib.request',
        'compatibility_patches',
        'restore paper-era effective optional archive version=0',
        'binary_archive_input_used',
        'boost_optional_overlay',
        '/kaggle/temp/vinardo_preflight',
        'mode_pattern',
        'model_pattern',
    ):
        assert required in text
