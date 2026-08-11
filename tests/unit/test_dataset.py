from __future__ import annotations

import numpy as np

from diffusion_sources.dataset import build_example
from diffusion_sources.diffusion import simulate_ic
from diffusion_sources.observations import observe_cascade


def test_build_example_uses_shared_graph_and_multi_label_targets(path_graph):
    cascade = simulate_ic(path_graph, {1, 3}, 1.0, 3, np.random.default_rng(4))
    observation = observe_cascade(
        path_graph, cascade, 1.0, 0, np.random.default_rng(5)
    )

    example = build_example(
        "path", path_graph, cascade, observation, simulation_seed=4, observation_seed=5
    )

    assert example.graph_id == "path"
    assert example.features.shape == (5, 2)
    assert example.source_count == 2
    assert example.source_labels.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert example.infected_mask.tolist() == [True] * 5
    assert example.candidate_mask.tolist() == [True] * 5
