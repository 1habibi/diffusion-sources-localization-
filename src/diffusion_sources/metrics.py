"""Evaluation metrics for sets of multiple predicted diffusion sources."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np


def set_metrics(
    true_sources: Iterable[int], predicted_sources: Iterable[int]
) -> dict[str, float]:
    """Calculate set precision, recall, F1, exact match and count errors."""
    true_set = set(true_sources)
    predicted_set = set(predicted_sources)
    if not true_set:
        raise ValueError("true_sources must not be empty.")

    overlap = len(true_set & predicted_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(true_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_accuracy": float(true_set == predicted_set),
        "count_accuracy": float(len(true_set) == len(predicted_set)),
        "count_mae": float(abs(len(true_set) - len(predicted_set))),
    }


def source_set_distances(
    graph: nx.Graph,
    true_sources: Iterable[int],
    predicted_sources: Iterable[int],
) -> dict[str, float]:
    """Calculate directed nearest-source distances between two non-empty sets."""
    true_set = set(true_sources)
    predicted_set = set(predicted_sources)
    if not true_set or not predicted_set:
        raise ValueError("Both source sets must be non-empty.")

    def average_nearest(origins: set[int], targets: set[int]) -> float:
        return float(
            np.mean(
                [min(nx.shortest_path_length(graph, source, target) for target in targets)
                 for source in origins]
            )
        )

    source_to_set = average_nearest(true_set, predicted_set)
    set_to_source = average_nearest(predicted_set, true_set)
    return {
        "source_to_set_distance": source_to_set,
        "set_to_source_distance": set_to_source,
        "symmetric_set_distance": (source_to_set + set_to_source) / 2,
    }


def source_radius_hits(
    graph: nx.Graph,
    true_sources: Iterable[int],
    predicted_sources: Iterable[int],
    *,
    radii: Iterable[int] = (1, 2),
) -> dict[str, float]:
    """Return true-source coverage within each graph-distance radius.

    For every true source, the nearest predicted source is considered a hit when
    its shortest-path distance is at most the requested radius. Results are
    averaged over true sources, matching the per-cascade Hit@r-hop definition.
    """
    true_set = set(true_sources)
    predicted_set = set(predicted_sources)
    requested_radii = tuple(int(radius) for radius in radii)
    if not true_set:
        raise ValueError("true_sources must not be empty.")
    if not predicted_set:
        raise ValueError("predicted_sources must not be empty.")
    if not requested_radii or any(radius < 0 for radius in requested_radii):
        raise ValueError("radii must contain non-negative integers.")
    if len(set(requested_radii)) != len(requested_radii):
        raise ValueError("radii must not contain duplicates.")

    max_radius = max(requested_radii)
    hit_counts = {radius: 0 for radius in requested_radii}
    for source in true_set:
        distances = nx.single_source_shortest_path_length(
            graph, source, cutoff=max_radius
        )
        nearest = min(
            (distances[node] for node in predicted_set if node in distances),
            default=max_radius + 1,
        )
        for radius in requested_radii:
            hit_counts[radius] += int(nearest <= radius)

    return {
        f"hit_at_{radius}_hop": hit_counts[radius] / len(true_set)
        for radius in requested_radii
    }
