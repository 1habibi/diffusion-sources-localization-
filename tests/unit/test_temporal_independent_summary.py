import copy
import numpy as np
import pytest
from diffusion_sources.temporal_statistics import paired_bootstrap_ci


def synthetic_reports():
    reports = []
    for seed in (7026, 7027, 7028):
        rows = []
        for i in range(1998):
            row = {'index': i, 'k': i % 3 + 1, 'candidate_count': 10,
                   'true_sources': list(range(i % 3 + 1)),
                   'snapshot_f1': .3, 'early_expected_f1': .4, 'temporal_f1': .5,
                   'temporal_minus_snapshot': .2, 'temporal_minus_early': .1}
            for label, f1, exact, distance, hit in [('snapshot', .3, 0., 1., .6), ('temporal', .5, .2, .5, .8)]:
                row[label] = {'precision': f1, 'recall': f1, 'f1': f1, 'exact_set_accuracy': exact,
                              'count_accuracy': 1., 'count_mae': 0., 'symmetric_set_distance': distance,
                              'hit_at_1_hop': hit, 'hit_at_2_hop': .9}
            rows.append(row)
        reports.append({'seed': seed, 'beta': .5, 'n': 1998, 'early_mask_hash': 'same',
                        'dataset_hash': 'same', 'freeze_hash': 'same', 'count_unchanged': True, 'rows': rows})
    return reports


def set_temporal(row, f1):
    row['temporal_f1'] = row['temporal']['f1'] = f1
    row['temporal_minus_snapshot'] = f1 - row['snapshot_f1']
    row['temporal_minus_early'] = f1 - row['early_expected_f1']


def test_aggregate_cascade_ci():
    from diffusion_sources.temporal_independent_summary import summarize_reports
    result = summarize_reports(synthetic_reports())
    assert result['bootstrap_n'] == 1998
    assert result['primary']['delta_f1'] == pytest.approx(.2)
    assert result['primary']['ci'] == pytest.approx([.2, .2])
    assert result['primary']['passed'] and result['secondary']['passed']
    assert result['means']['temporal_f1'] == .5
    assert result['metrics']['temporal']['mean']['exact_set_accuracy'] == pytest.approx(.2)


def test_asymmetric_ci_averages_cascades_not_flattened():
    from diffusion_sources.temporal_independent_summary import summarize_reports
    reports = synthetic_reports()
    for s, report in enumerate(reports):
        for i, row in enumerate(report['rows']):
            set_temporal(row, .35 + .08*s + .1*((i*7+s) % 11)/10)
    matrix = np.array([[row['temporal_minus_snapshot'] for row in r['rows']] for r in reports])
    k = np.array([row['k'] for row in reports[0]['rows']])
    expected = paired_bootstrap_ci(matrix.mean(axis=0), k)
    flattened = paired_bootstrap_ci(matrix.ravel(), np.tile(k, 3))
    assert expected != flattened
    assert summarize_reports(reports)['primary']['ci'] == list(expected)
    assert summarize_reports(reports)['primary']['ci'] == list(expected)


@pytest.mark.parametrize('case', ['duplicates', 'shuffle', 'mask', 'dataset', 'beta', 'nan', 'delta', 'missing', 'truth'])
def test_reject_unpaired_reports(case):
    from diffusion_sources.temporal_independent_summary import summarize_reports
    reports = synthetic_reports()
    if case == 'duplicates': reports[1]['seed'] = 7026
    if case == 'shuffle': reports[1]['rows'].reverse()
    if case == 'mask': reports[1]['early_mask_hash'] = 'changed'
    if case == 'dataset': reports[1]['dataset_hash'] = 'changed'
    if case == 'beta': reports[1]['beta'] = .25
    if case == 'nan': reports[1]['rows'][0]['early_expected_f1'] = float('nan')
    if case == 'delta': reports[1]['rows'][0]['temporal_minus_snapshot'] = 999
    if case == 'missing': reports[1]['rows'].pop()
    if case == 'truth': reports[1]['rows'][0]['true_sources'] = [9]
    with pytest.raises(ValueError): summarize_reports(reports)


@pytest.mark.parametrize('case', ['mean', 'seed', 'k_loss', 'count', 'secondary'])
def test_fixed_negative_gates(case):
    from diffusion_sources.temporal_independent_summary import summarize_reports
    reports = synthetic_reports()
    if case == 'mean':
        for r in reports:
            for row in r['rows']: set_temporal(row, .31)
    if case == 'seed':
        for row in reports[0]['rows']: set_temporal(row, .2)
    if case == 'k_loss':
        for r in reports:
            for row in r['rows']: set_temporal(row, .2 if row['k'] == 2 else .6)
    if case == 'count':
        reports[0]['count_unchanged'] = False
    if case == 'secondary':
        for r in reports:
            for row in r['rows']:
                row['early_expected_f1'] = .6
                row['temporal_minus_early'] = -.1
    result = summarize_reports(reports)
    assert result['primary']['passed'] == (case == 'secondary')
    assert not result['secondary']['passed']
