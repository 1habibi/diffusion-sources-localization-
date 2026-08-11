from __future__ import annotations

import numpy as np

from diffusion_sources.diffusion import simulate_ic
from diffusion_sources.features import node_features
from diffusion_sources.observations import observe_cascade


def test_primary_observation_preserves_sources_and_adds_noise(path_graph):
    cascade = simulate_ic(path_graph, {1, 3}, 1.0, 3, np.random.default_rng(2))
    observation = observe_cascade(
        path_graph, cascade, 0.5, 0, np.random.default_rng(3)
    )

    assert cascade.sources.issubset(observation.observed_infected)
    assert observation.candidate_nodes == observation.observed_infected
    assert not observation.hidden_sources


def test_hidden_source_switches_to_all_graph_candidates(path_graph):
    cascade = simulate_ic(path_graph, {1, 3}, 1.0, 3, np.random.default_rng(2))
    observation = observe_cascade(
        path_graph, cascade, 1.0, 0, np.random.default_rng(3), hide_source_count=1
    )

    assert len(observation.hidden_sources) == 1
    assert observation.candidate_nodes == frozenset(path_graph.nodes())
    assert observation.hidden_sources.isdisjoint(observation.observed_infected)


def test_features_encode_observation_and_normalized_degree(path_graph):
    cascade = simulate_ic(path_graph, {2}, 0.0, 3, np.random.default_rng(2))
    observation = observe_cascade(
        path_graph, cascade, 1.0, 0, np.random.default_rng(3)
    )

    features = node_features(path_graph, observation)

    assert features.shape == (5, 2)
    assert features[2, 0] == 1.0
    assert features[0, 0] == 0.0
    assert features[2, 1] == 1.0
    assert 0.0 < features[0, 1] < 1.0
