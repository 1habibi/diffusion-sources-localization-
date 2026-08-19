from __future__ import annotations

import numpy as np
import pytest

from diffusion_sources.baselines import multi_jordan
from diffusion_sources.diffusion import simulate_ic
from diffusion_sources.features import SnapshotFeatureBuilder, node_features
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


def test_snapshot_feature_builder_uses_only_observed_mask(path_graph):
    builder = SnapshotFeatureBuilder(path_graph)
    observed = np.asarray([False, True, False, False, False])

    features = builder.build(
        observed,
        [
            "observed_neighbor_count_normalized",
            "observed_neighbor_fraction",
            "unobserved_neighbor_fraction",
            "closed_neighborhood_observed_fraction",
        ],
    )

    assert features.shape == (5, 4)
    assert features[0, 1] == 1.0
    assert features[1, 1] == 0.0
    assert features[2, 1] == 0.5
    assert features[0, 2] == 0.0
    assert features[1, 3] == pytest.approx(1 / 3)


def test_snapshot_distance_features_follow_observed_geometry(path_graph, tmp_path):
    cache_path = tmp_path / "distances.npz"
    builder = SnapshotFeatureBuilder(
        path_graph, distance_cache_path=cache_path, distance_cap=4
    )
    observed = np.asarray([False, True, True, True, False])

    features = builder.build(
        observed,
        [
            "distance_to_observation_boundary_normalized",
            "observation_boundary_missing",
            "mean_distance_to_observed_normalized",
            "max_distance_to_observed_normalized",
            "induced_observed_eccentricity_normalized",
        ],
    )

    assert cache_path.exists()
    assert features.shape == (5, 5)
    assert features[:, 0].tolist() == pytest.approx([0.25, 0.0, 0.25, 0.0, 0.25])
    assert features[:, 1].tolist() == [0.0] * 5
    assert features[2, 2] == pytest.approx((1 + 0 + 1) / 3 / 4)
    assert features[2, 3] == pytest.approx(1 / 4)
    assert features[1:4, 4].tolist() == pytest.approx([0.5, 0.25, 0.5])

    cached_builder = SnapshotFeatureBuilder(
        path_graph, distance_cache_path=cache_path, distance_cap=4
    )
    cached = cached_builder.build(
        observed, ["mean_distance_to_observed_normalized"]
    )
    assert np.array_equal(cached[:, 0], features[:, 2])


def test_normalized_multi_jordan_rank_matches_greedy_baseline(path_graph, tmp_path):
    builder = SnapshotFeatureBuilder(
        path_graph,
        distance_cache_path=tmp_path / "distances.npz",
        distance_cap=4,
    )
    observed = np.ones(5, dtype=bool)

    candidates = np.ones(5, dtype=bool)
    scores = builder.build(
        observed,
        ["multi_jordan_rank_normalized"],
        candidate_mask=candidates,
    )[:, 0]

    assert scores.shape == (5,)
    assert np.all((0.0 <= scores) & (scores <= 1.0))
    for source_count in (1, 2, 3):
        ranked = frozenset(
            np.argsort(-scores, kind="stable")[:source_count].tolist()
        )
        expected = multi_jordan(
            path_graph,
            path_graph.nodes(),
            path_graph.nodes(),
            source_count,
        )
        assert ranked == expected


def test_multi_jordan_rank_requires_candidate_mask(path_graph):
    with pytest.raises(ValueError, match="candidate_mask"):
        SnapshotFeatureBuilder(path_graph).build(
            np.ones(5, dtype=bool), ["multi_jordan_rank_normalized"]
        )


def test_snapshot_distance_cache_rejects_another_topology(path_graph, tmp_path):
    cache_path = tmp_path / "distances.npz"
    SnapshotFeatureBuilder(path_graph, distance_cache_path=cache_path).build(
        np.asarray([True, False, False, False, False]),
        ["mean_distance_to_observed_normalized"],
    )
    other_graph = path_graph.copy()
    other_graph.add_edge(0, 4)

    with pytest.raises(ValueError, match="fingerprint"):
        SnapshotFeatureBuilder(
            other_graph, distance_cache_path=cache_path
        ).build(
            np.asarray([True, False, False, False, False]),
            ["mean_distance_to_observed_normalized"],
        )


def test_snapshot_global_features_describe_observation_and_candidates(path_graph):
    builder = SnapshotFeatureBuilder(path_graph)
    observed = np.asarray([False, True, True, True, False])
    candidates = np.asarray([False, True, True, True, True])

    features = builder.build(
        observed,
        [
            "observed_count_normalized",
            "candidate_count_normalized",
            "observed_candidate_fraction",
            "observed_subgraph_density",
            "observed_component_count_normalized",
            "observed_largest_component_fraction",
        ],
        candidate_mask=candidates,
    )

    assert features.shape == (5, 6)
    assert np.all(features == features[0])
    assert features[0, 0] == pytest.approx(np.log1p(3) / np.log1p(5))
    assert features[0, 1] == pytest.approx(np.log1p(4) / np.log1p(5))
    assert features[0, 2] == pytest.approx(0.75)
    assert features[0, 3] == pytest.approx(2 / 3)
    assert features[0, 4] == pytest.approx(np.log1p(1) / np.log1p(3))
    assert features[0, 5] == 1.0


def test_snapshot_global_features_require_candidate_mask(path_graph):
    builder = SnapshotFeatureBuilder(path_graph)

    with pytest.raises(ValueError, match="candidate_mask"):
        builder.build(
            np.asarray([True, False, False, False, False]),
            ["candidate_count_normalized"],
        )
