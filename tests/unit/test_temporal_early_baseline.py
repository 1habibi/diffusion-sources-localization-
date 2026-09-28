import itertools

import numpy as np
import pytest

from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord


def candidates(early, count=1):
    ids = tuple(range(len(early)))
    return CandidateScores(ids, tuple(.9 - .1 * i for i in ids), tuple(early),
                           frozenset(ids[:min(count, len(ids))]), count)


@pytest.mark.parametrize('early,count,true,expected', [
    ([True, True, False], 1, {0}, .5),
    ([True, False, False, False], 2, {0, 1}, 2 / 3),
    ([False] * 4, 2, {0, 1}, .5),
    ([True] * 4, 2, {0, 1}, .5),
    ([False], 3, {0}, 1.),
])
def test_exact_expectation_with_ties_and_count_clamping(early, count, true, expected):
    from diffusion_sources.temporal_early_baseline import expected_early_f1
    assert expected_early_f1(candidates(early, count), frozenset(true)) == pytest.approx(expected)


def test_expectation_equals_exhaustive_tie_enumeration():
    from diffusion_sources.metrics import set_metrics
    from diffusion_sources.temporal_early_baseline import expected_early_f1
    c = candidates([True, True, False, False], 3)
    true = frozenset({0, 2, 3})
    legal = [frozenset({0, 1, *tail}) for tail in itertools.combinations([2, 3], 1)]
    enumerated = np.mean([set_metrics(true, prediction)['f1'] for prediction in legal])
    assert expected_early_f1(c, true) == enumerated == 2 / 3


def test_expectation_independent_of_scores_and_node_ids():
    from diffusion_sources.temporal_early_baseline import expected_early_f1
    a = candidates([True, True, False], 1)
    b = CandidateScores((100, 7, 99), (.01, .99, .8), (True, True, False), frozenset({7}), 1)
    assert expected_early_f1(a, frozenset({0})) == expected_early_f1(b, frozenset({100})) == .5


@pytest.mark.parametrize('true', [frozenset(), frozenset({99})])
def test_invalid_targets_rejected(true):
    from diffusion_sources.temporal_early_baseline import expected_early_f1
    with pytest.raises(ValueError):
        expected_early_f1(candidates([True, False]), true)


def test_summary_compares_gcn_against_expected_early_baseline():
    from diffusion_sources.temporal_early_baseline import evaluate_early_baseline
    c = CandidateScores((0, 1, 2), (.9, .1, .5), (True, True, False), frozenset({0}), 1)
    records = [PilotRecord(0, frozenset({0}), c, False)]
    report = evaluate_early_baseline(records, .5)
    assert report['all']['early_expected_f1'] == .5
    assert report['all']['snapshot_f1'] == 1.
    assert report['all']['temporal_f1'] == 1.
    assert report['all']['temporal_minus_early'] == .5
    assert report['f1_ci'] == [.5, .5]
    assert report['by_k']['2']['n'] == 0
    assert report['by_k']['2']['temporal_minus_early'] is None
    assert report['by_candidates']['1-10']['n'] == 1
    assert report['count_unchanged']
    assert report['early_baseline_uses_gcn_count']


def test_nonfinite_beta_and_duplicate_indices_rejected():
    from diffusion_sources.temporal_early_baseline import evaluate_early_baseline
    r = PilotRecord(0, frozenset({0}), candidates([True, False]), False)
    for records, beta in (([], .5), ([r, r], .5), ([r], float('nan')), ([r], -.5)):
        with pytest.raises(ValueError):
            evaluate_early_baseline(records, beta)
