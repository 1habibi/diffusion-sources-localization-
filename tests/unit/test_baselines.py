from __future__ import annotations

import numpy as np

from diffusion_sources.baselines import degree_candidates, multi_jordan, uniform_candidates


def test_degree_candidates_prefers_highest_degree(path_graph):
    prediction = degree_candidates(path_graph, {0, 1, 2, 4}, 2)

    assert prediction == frozenset({1, 2})


def test_uniform_candidates_is_seed_reproducible(path_graph):
    first = uniform_candidates(path_graph.nodes(), 2, np.random.default_rng(11))
    second = uniform_candidates(path_graph.nodes(), 2, np.random.default_rng(11))

    assert first == second
    assert len(first) == 2


def test_multi_jordan_selects_separated_centers_on_path(path_graph):
    prediction = multi_jordan(
        path_graph, path_graph.nodes(), path_graph.nodes(), source_count=2
    )

    assert prediction == frozenset({0, 2})
