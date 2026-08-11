"""Transform complete cascades into noisy observations available to a model."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .diffusion import Cascade


@dataclass(frozen=True)
class Observation:
    """Observed infection state and the candidate mask for one cascade."""

    observed_infected: frozenset[int]
    false_positive: frozenset[int]
    candidate_nodes: frozenset[int]
    hidden_sources: frozenset[int]
    observation_fraction: float
    false_positive_count: int
    search_all_nodes: bool


def observe_cascade(
    graph: nx.Graph,
    cascade: Cascade,
    observation_fraction: float,
    false_positive_count: int,
    rng: np.random.Generator,
    *,
    hide_source_count: int = 0,
) -> Observation:
    """Sample an incomplete observation without exposing source labels.

    In the primary setup all sources remain observed and candidates are the
    observed infected nodes. Hiding a source switches candidates to all graph
    nodes because the source may no longer be visible in the snapshot.
    """
    if not 0.0 < observation_fraction <= 1.0:
        raise ValueError("observation_fraction must be in (0, 1].")
    if false_positive_count < 0:
        raise ValueError("false_positive_count must be non-negative.")
    if not 0 <= hide_source_count <= len(cascade.sources):
        raise ValueError("hide_source_count must be between 0 and source count.")

    source_nodes = sorted(cascade.sources)
    hidden_sources = frozenset(
        rng.choice(source_nodes, size=hide_source_count, replace=False).tolist()
        if hide_source_count
        else []
    )
    required_sources = cascade.sources - hidden_sources
    non_source_infected = sorted(cascade.infected - cascade.sources)
    target_count = round(observation_fraction * len(cascade.infected))
    sampled_count = max(0, target_count - len(required_sources))
    sampled_count = min(sampled_count, len(non_source_infected))

    sampled = (
        rng.choice(non_source_infected, size=sampled_count, replace=False).tolist()
        if sampled_count
        else []
    )
    observed_infected = frozenset(required_sources | set(sampled))

    available_false_positives = sorted(set(graph) - cascade.infected)
    if false_positive_count > len(available_false_positives):
        raise ValueError("Not enough susceptible nodes for requested false positives.")
    false_positive = frozenset(
        rng.choice(
            available_false_positives, size=false_positive_count, replace=False
        ).tolist()
        if false_positive_count
        else []
    )

    observed = observed_infected | false_positive
    search_all_nodes = bool(hidden_sources)
    return Observation(
        observed_infected=observed,
        false_positive=false_positive,
        candidate_nodes=frozenset(graph) if search_all_nodes else observed,
        hidden_sources=hidden_sources,
        observation_fraction=observation_fraction,
        false_positive_count=false_positive_count,
        search_all_nodes=search_all_nodes,
    )
