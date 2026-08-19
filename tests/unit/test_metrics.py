from __future__ import annotations

import pytest

from diffusion_sources.metrics import set_metrics, source_radius_hits, source_set_distances


def test_set_metrics_for_partial_overlap():
    metrics = set_metrics({1, 2}, {2, 3})

    assert metrics == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "exact_set_accuracy": 0.0,
        "count_accuracy": 1.0,
        "count_mae": 0.0,
    }


def test_set_metrics_for_exact_match():
    metrics = set_metrics({1, 2}, {2, 1})

    assert metrics["f1"] == 1.0
    assert metrics["exact_set_accuracy"] == 1.0


def test_source_set_distances(path_graph):
    distances = source_set_distances(path_graph, {0, 4}, {1, 3})

    assert distances == {
        "source_to_set_distance": 1.0,
        "set_to_source_distance": 1.0,
        "symmetric_set_distance": 1.0,
    }


def test_source_set_distances_reject_empty_predictions(path_graph):
    with pytest.raises(ValueError, match="non-empty"):
        source_set_distances(path_graph, {0}, set())


def test_source_radius_hits_cover_true_sources(path_graph):
    hits = source_radius_hits(path_graph, {0, 4}, {1})

    assert hits == {"hit_at_1_hop": 0.5, "hit_at_2_hop": 0.5}


def test_source_radius_hits_support_exact_radius(path_graph):
    hits = source_radius_hits(path_graph, {0, 4}, {0, 3}, radii=(0, 1, 2))

    assert hits == {
        "hit_at_0_hop": 0.5,
        "hit_at_1_hop": 1.0,
        "hit_at_2_hop": 1.0,
    }


def test_source_radius_hits_reject_empty_predictions(path_graph):
    with pytest.raises(ValueError, match="predicted_sources"):
        source_radius_hits(path_graph, {0}, set())
