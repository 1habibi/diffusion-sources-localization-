import networkx as nx
import numpy as np
import pytest


def test_constant_delta_ci():
    from diffusion_sources.temporal_statistics import paired_bootstrap_ci
    assert paired_bootstrap_ci(np.full(6, .125), np.array([1, 1, 2, 2, 3, 3])) == (.125, .125)


def test_ci_reproducible_and_paired():
    from diffusion_sources.temporal_statistics import paired_bootstrap_ci
    args = (np.array([-.2, .1, .3, .4]), np.array([1, 1, 2, 2]))
    assert paired_bootstrap_ci(*args) == paired_bootstrap_ci(*args)
    lo, hi = paired_bootstrap_ci(*args)
    assert lo <= .15 <= hi


@pytest.mark.parametrize('d,k', [([], []), ([0.], [1, 2]), ([float('nan')], [1]), ([0.], [4])])
def test_invalid_ci(d, k):
    from diffusion_sources.temporal_statistics import paired_bootstrap_ci
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array(d), np.array(k))


def report(delta=.03, ci=(.01, .05), k2=.03, k3=.03, count=True):
    return {'delta_f1': delta, 'f1_ci': ci,
            'delta_by_k': {'1': delta, '2': k2, '3': k3}, 'count_unchanged': count}


@pytest.mark.parametrize('bad', [report(delta=0., ci=(0., 0.)), report(ci=(-.01, .1)),
                               report(k2=-.021), report(k3=-.021), report(count=False)])
def test_gate_rejects_bad_results(bad):
    from diffusion_sources.temporal_statistics import primary_gate
    assert not primary_gate(bad)['passed']


def test_gates_and_missing_strata():
    from diffusion_sources.temporal_statistics import primary_gate, confirmation_gate
    assert primary_gate(report())['passed']
    bad = report(); bad['delta_by_k']['2'] = None
    assert not primary_gate(bad)['passed']
    assert confirmation_gate(report(), [report(), report()])['passed']
    assert not confirmation_gate(report(), [report(delta=0.), report()])['passed']
    assert not confirmation_gate(report(), [report()])['passed']


def test_geometry_empty_groups_and_no_filter():
    from diffusion_sources.temporal_statistics import evaluate_pairs
    from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord
    c = CandidateScores((0, 1, 2), (.9, .5, .1), (False, False, True), frozenset({0}), 1)
    r = PilotRecord(0, frozenset({2}), c, False)
    result = evaluate_pairs([r], 1., nx.path_graph(3))
    assert result['baseline']['all']['f1'] == 0.
    assert result['temporal']['all']['f1'] == 1.
    assert result['baseline']['all']['symmetric_set_distance'] == 2.
    assert result['baseline']['all']['hit_at_1_hop'] == 0.
    assert result['baseline']['all']['hit_at_2_hop'] == 1.
    assert result['baseline']['by_k']['2']['n'] == 0
    assert result['baseline']['by_candidates']['51+']['f1'] is None
    empty = PilotRecord(1, frozenset({2}), CandidateScores((0, 1, 2), (.9, .5, .1),
                        (False, False, False), frozenset({0}), 1), True)
    zero = evaluate_pairs([r, empty], 0., nx.path_graph(3))
    assert zero['delta_f1'] == 0. and len(zero['rows']) == 2
    assert zero['early_coverage']['empty_fraction'] == .5
