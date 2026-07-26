from ezscreen.benchmark.aggregation import aggregate_source_compounds
from ezscreen.benchmark.runner import run_benchmark


def test_collapses_variants_receptors_and_conformers_to_best_source_score() -> None:
    rows = [
        {'name': 'A_v1', '_ranking_score': -7.0, 'receptor_variant': 'r1'},
        {'name': 'A_v2', '_ranking_score': -9.0, 'receptor_variant': 'r2'},
        {'name': 'B', '_ranking_score': -8.0, 'conformer_id': 'c1'},
    ]
    result = aggregate_source_compounds(rows)
    assert len(result.records) == 2
    source_a = next(
        row for row in result.records if row['source_compound_id'] == 'A'
    )
    assert source_a['_ranking_score'] == -9.0
    assert source_a['variant_count'] == 2
    assert source_a['receptor_count'] == 2
    assert result.counts['variant_rows_collapsed'] == 1


def test_failed_sources_share_worst_rank_and_remain_in_denominator() -> None:
    rows = [
        {'name': 'A', '_ranking_score': -8.0, '_scored': True},
        {'name': 'B', '_scored': False, 'reason': 'prep_failed'},
        {'name': 'C', '_scored': False, 'reason': 'dock_failed'},
    ]
    result = aggregate_source_compounds(rows)
    failed = [row for row in result.records if row['_failed']]
    assert len(result.records) == 3
    assert len(failed) == 2
    assert failed[0]['_ranking_score'] == failed[1]['_ranking_score']
    assert failed[0]['_ranking_score'] > -8.0
    assert result.counts['failed'] == 2


def test_runner_uses_one_intended_row_per_source_and_keeps_failures(tmp_path) -> None:
    actives = tmp_path / 'actives.smi'
    actives.write_text('CCO A\n', encoding='utf-8')
    intended = tmp_path / 'intended.csv'
    intended.write_text(
        'source_compound_id,name,smiles\nA,A,CCO\nD,D,CCC\nF,F,CCN\n',
        encoding='utf-8',
    )
    scores = tmp_path / 'scores.csv'
    scores.write_text(
        'name,smiles,score,receptor_variant\n'
        'A_v1,CCO,-7,r1\nA_v2,CCO,-9,r2\nD,CCC,-8,r1\n',
        encoding='utf-8',
    )
    (tmp_path / 'unscored_reasons.csv').write_text(
        'name,smiles,reason\nF,CCN,dock_failed\n', encoding='utf-8'
    )
    result = run_benchmark(
        actives,
        scores,
        score_col='score',
        intended_compounds_path=intended,
    )
    assert result.total_screened == 3
    assert result.n_actives == 1
    assert result.auc_roc == 1.0
    assert result.coverage == 0.666667
    assert result.label_diagnostics['counts_before_prep'] == 3
    assert result.label_diagnostics['counts_after_aggregation'] == 3
    assert result.label_diagnostics['counts_scored'] == 2
    assert result.label_diagnostics['counts_failed'] == 1
    assert result.label_diagnostics['variant_rows_collapsed'] == 4
