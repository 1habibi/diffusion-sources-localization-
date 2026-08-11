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
