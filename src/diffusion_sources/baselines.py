"""Classical multi-source localization baselines for oracle-k experiments."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np


def uniform_candidates(
    candidates: Iterable[int], source_count: int, rng: np.random.Generator
) -> frozenset[int]:
    """Select source_count candidates uniformly without replacement."""
    candidate_list = sorted(set(candidates))
    _validate_candidate_count(candidate_list, source_count)
    return frozenset(rng.choice(candidate_list, size=source_count, replace=False).tolist())


def degree_candidates(
    graph: nx.Graph, candidates: Iterable[int], source_count: int
) -> frozenset[int]:
    """Select candidates with the largest degree, breaking ties by node ID."""
    candidate_list = sorted(set(candidates))
    _validate_candidate_count(candidate_list, source_count)
    ranked = sorted(candidate_list, key=lambda node: (-graph.degree(node), node))
    return frozenset(ranked[:source_count])


def multi_jordan(
    graph: nx.Graph,
    observed_infected: Iterable[int],
    candidates: Iterable[int],
    source_count: int,
) -> frozenset[int]:
    """Greedily select k centers minimizing nearest-center eccentricity.

    At each step candidates are ordered by the maximum distance from an
    observed infected node to its nearest selected center. The total distance
    and node ID provide deterministic tie breakers.
    """
    observed = sorted(set(observed_infected))
    candidate_list = sorted(set(candidates))
    _validate_candidate_count(candidate_list, source_count)
    if not observed:
        raise ValueError("observed_infected must not be empty.")
    if not set(observed).issubset(graph) or not set(candidate_list).issubset(graph):
        raise ValueError("Observed and candidate nodes must belong to the graph.")

    distances = {
        candidate: nx.single_source_shortest_path_length(graph, candidate)
        for candidate in candidate_list
    }
    selected: list[int] = []
    while len(selected) < source_count:
        best_candidate = min(
            (candidate for candidate in candidate_list if candidate not in selected),
            key=lambda candidate: _center_objective(
                observed, selected + [candidate], distances
            ),
        )
        selected.append(best_candidate)
    return frozenset(selected)


def _center_objective(
    observed: list[int],
    centers: list[int],
    distances: dict[int, dict[int, int]],
) -> tuple[int, int, int]:
    nearest = [min(distances[center][node] for center in centers) for node in observed]
    return max(nearest), sum(nearest), centers[-1]


def _validate_candidate_count(candidates: list[int], source_count: int) -> None:
    if source_count < 1:
        raise ValueError("source_count must be positive.")
    if source_count > len(candidates):
        raise ValueError("source_count cannot exceed available candidates.")
