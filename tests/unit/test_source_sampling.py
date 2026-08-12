from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from diffusion_sources.diffusion import SourceSampler, sample_sources


def test_sample_sources_respects_pairwise_distance():
    graph = nx.path_graph(8)
    sources = sample_sources(
        graph, 3, np.random.default_rng(7), min_distance=2, max_distance=6
    )

    assert len(sources) == 3
    for source, target in itertools.combinations(sources, 2):
        distance = nx.shortest_path_length(graph, source, target)
        assert 2 <= distance <= 6


def test_sample_sources_is_reproducible(path_graph):
    first = sample_sources(path_graph, 2, np.random.default_rng(5), min_distance=2)
    second = sample_sources(path_graph, 2, np.random.default_rng(5), min_distance=2)
    assert first == second


def test_sample_sources_rejects_impossible_constraints(path_graph):
    with pytest.raises(RuntimeError, match="Could not sample"):
        sample_sources(
            path_graph,
            3,
            np.random.default_rng(1),
            min_distance=4,
            max_attempts=10,
        )


def test_source_sampler_reuses_bounded_compatibility_cache(path_graph):
    sampler = SourceSampler(path_graph, cache_size=2)
    sampler.sample(path_graph.number_of_nodes() > 1 and 2 or 1, np.random.default_rng(2))
    sampler._compatible_nodes(0, 1, 3)
    sampler._compatible_nodes(1, 1, 3)
    sampler._compatible_nodes(2, 1, 3)

    assert len(sampler._compatible_cache) == 2
