"""Independent Cascade diffusion with one or more simultaneous sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Cascade:
    """The full, unobserved result of one IC diffusion simulation."""

    sources: frozenset[int]
    infection_times: dict[int, int]
    history: tuple[frozenset[int], ...]
    transmission_probability: float
    max_steps: int

    @property
    def infected(self) -> frozenset[int]:
        return frozenset(self.infection_times)


def sample_sources(
    graph: nx.Graph,
    source_count: int,
    rng: np.random.Generator,
    *,
    min_distance: int = 0,
    max_distance: int | None = None,
    max_attempts: int = 1_000,
) -> frozenset[int]:
    """Sample distinct sources satisfying pairwise shortest-path constraints."""
    if not 1 <= source_count <= 3:
        raise ValueError("source_count must be between 1 and 3.")
    if source_count > graph.number_of_nodes():
        raise ValueError("source_count cannot exceed the graph node count.")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")
    if max_distance is not None and max_distance < min_distance:
        raise ValueError("max_distance must be greater than or equal to min_distance.")

    nodes = np.asarray(sorted(graph.nodes()), dtype=int)
    for _ in range(max_attempts):
        selected = frozenset(
            rng.choice(nodes, size=source_count, replace=False).tolist()
        )
        if source_count == 1 or _distances_fit(
            graph, selected, min_distance, max_distance
        ):
            return selected
    raise RuntimeError("Could not sample sources satisfying distance constraints.")


def _distances_fit(
    graph: nx.Graph,
    sources: frozenset[int],
    min_distance: int,
    max_distance: int | None,
) -> bool:
    ordered = sorted(sources)
    for index, source in enumerate(ordered):
        distances = nx.single_source_shortest_path_length(graph, source)
        for target in ordered[index + 1 :]:
            distance = distances[target]
            if distance < min_distance:
                return False
            if max_distance is not None and distance > max_distance:
                return False
    return True


def simulate_ic(
    graph: nx.Graph,
    sources: Iterable[int],
    probability: float,
    max_steps: int,
    rng: np.random.Generator,
) -> Cascade:
    """Simulate IC diffusion from simultaneously activated source nodes.

    A newly infected node gets exactly one chance to infect each susceptible
    neighbor on the next simulation step.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")

    source_set = frozenset(sources)
    if not source_set:
        raise ValueError("At least one source is required.")
    if not source_set.issubset(graph):
        raise ValueError("All sources must be graph nodes.")

    infection_times = {node: 0 for node in source_set}
    history: list[frozenset[int]] = [source_set]
    frontier = source_set

    for step in range(1, max_steps + 1):
        newly_infected: set[int] = set()
        for node in sorted(frontier):
            for neighbor in sorted(graph.neighbors(node)):
                if neighbor in infection_times or neighbor in newly_infected:
                    continue
                if rng.random() < probability:
                    newly_infected.add(neighbor)

        if not newly_infected:
            break

        for node in newly_infected:
            infection_times[node] = step
        frontier = frozenset(newly_infected)
        history.append(frontier)

    return Cascade(
        sources=source_set,
        infection_times=infection_times,
        history=tuple(history),
        transmission_probability=probability,
        max_steps=max_steps,
    )


def simulate_si(
    graph: nx.Graph,
    sources: Iterable[int],
    probability: float,
    max_steps: int,
    rng: np.random.Generator,
) -> Cascade:
    """Simulate SI where every infected node retries susceptible neighbors."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")
    source_set = frozenset(sources)
    if not source_set:
        raise ValueError("At least one source is required.")
    if not source_set.issubset(graph):
        raise ValueError("All sources must be graph nodes.")

    infection_times = {node: 0 for node in source_set}
    history: list[frozenset[int]] = [source_set]
    for step in range(1, max_steps + 1):
        newly_infected: set[int] = set()
        for node in sorted(infection_times):
            for neighbor in sorted(graph.neighbors(node)):
                if neighbor in infection_times or neighbor in newly_infected:
                    continue
                if rng.random() < probability:
                    newly_infected.add(neighbor)
        for node in newly_infected:
            infection_times[node] = step
        history.append(frozenset(newly_infected))

    return Cascade(
        sources=source_set,
        infection_times=infection_times,
        history=tuple(history),
        transmission_probability=probability,
        max_steps=max_steps,
    )
