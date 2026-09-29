import inspect
import json

import networkx as nx
import numpy as np
import pytest

from diffusion_sources.temporal_learned_features import (
    FEATURE_NAMES, LinearHead, build_candidate_features, rank_candidates,
)


def graph():
    g = nx.path_graph(3)
    g.add_node(3)
    return g


def head(**changes):
    args = dict(feature_names=FEATURE_NAMES, mean=(0.,) * 6, scale=(1.,) * 6,
                coefficients=(0.,) * 6, intercept=0., C=1.)
    args.update(changes)
    return LinearHead(**args)


def test_features_manual_graph():
    x = build_candidate_features(graph(), (1, 3), np.array([.4, .2]),
                                 np.array([1, 0, 0, 0]), np.array([1, 1, 1, 0]))
    np.testing.assert_allclose(x, [[.4, 0, .5, 1, 1, 0], [.2, 0, 0, 0, 0, 0]])
    assert x.dtype == np.float64


def test_empty_early_isolate_and_external_neighbor():
    g = graph()
    x = build_candidate_features(g, (1,), np.array([.7]),
                                 np.array([1, 0, 0, 0]), np.zeros(4))
    assert x[0, 2] == .5  # Node 0 need not belong to the candidate pool.
    x = build_candidate_features(nx.empty_graph(2), (0,), np.array([.7]),
                                 np.zeros(2), np.zeros(2))
    np.testing.assert_equal(x, [[.7, 0, 0, 0, 0, 0]])


def test_permutation_alignment():
    args = (np.array([0, 1, 0, 0]), np.array([1, 1, 0, 0]))
    a = build_candidate_features(graph(), (0, 1), np.array([.1, .9]), *args)
    b = build_candidate_features(graph(), (1, 0), np.array([.9, .1]), *args)
    np.testing.assert_equal(a[::-1], b)


@pytest.mark.parametrize('ids,p,early,final', [
    ((1, 1), [.1, .2], [0]*4, [0]*4),
    ((4,), [.1], [0]*4, [0]*4),
    ((True,), [.1], [0]*4, [0]*4),
    ((1,), [np.nan], [0]*4, [0]*4),
    ((1,), [1.1], [0]*4, [0]*4),
    ((1,), [.1, .2], [0]*4, [0]*4),
    ((1,), [.1], [0]*3, [0]*4),
    ((1,), [.1], [0, 2, 0, 0], [0]*4),
    ((1,), [.1], [0]*4, [0, np.nan, 0, 0]),
])
def test_invalid_masks_ids_probabilities(ids, p, early, final):
    with pytest.raises(ValueError):
        build_candidate_features(graph(), ids, np.array(p), np.array(early), np.array(final))


@pytest.mark.parametrize('changes', [
    dict(scale=(0.,)*6), dict(mean=(np.nan,)*6), dict(coefficients=(np.inf,)*6),
    dict(intercept=np.inf), dict(C=0), dict(schema_version=2),
    dict(feature_names=('wrong',)*6), dict(mean=(0.,)*5),
])
def test_linear_head_schema_nonfinite_and_zero_scale(changes):
    with pytest.raises(ValueError):
        head(**changes)


def test_head_json_roundtrip():
    h = head(coefficients=(.1, .2, .3, .4, .5, .6), intercept=.7)
    restored = LinearHead.from_dict(json.loads(json.dumps(h.to_dict())))
    np.testing.assert_allclose(restored.decision_function(np.ones((3, 6))),
                               h.decision_function(np.ones((3, 6))), atol=1e-12)
    with pytest.raises(ValueError):
        LinearHead.from_dict({**h.to_dict(), 'extra': 1})
    with pytest.raises(ValueError):
        h.decision_function(np.ones((3, 5)))


def test_rank_ties_and_cardinality():
    ids = (2, 0, 1)
    p = np.array([.8, .8, .2])
    x = np.zeros((3, 6))
    assert rank_candidates(ids, p, x, 1, head()) == frozenset({0})
    assert rank_candidates(ids, p, x, 2, head()) == frozenset({0, 2})
    assert len(rank_candidates((0,), np.array([.3]), np.zeros((1, 6)), 3, head())) == 1
    for k in (0, 4, True, 1.5):
        with pytest.raises(ValueError):
            rank_candidates(ids, p, x, k, head())


def test_target_free_signatures():
    for fn in (build_candidate_features, rank_candidates):
        assert not {'labels', 'true_k', 'source_labels', 'infection_times', 'seed'} & set(inspect.signature(fn).parameters)


def test_feature_names_follow_approved_schema():
    assert FEATURE_NAMES[-1] == 'gcn_probability_x_early'
