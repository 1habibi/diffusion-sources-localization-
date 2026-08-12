"""Independent Cascade diffusion with one or more simultaneous sources."""

from __future__ import annotations

from collections import OrderedDict
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


class SourceSampler:
    """Efficiently sample source sets under pairwise distance constraints."""

    def __init__(self, graph: nx.Graph, cache_size: int = 128) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be positive.")
        self.graph = graph
        self.nodes = np.asarray(sorted(graph.nodes()), dtype=int)
        self.cache_size = cache_size
        self._compatible_cache: OrderedDict[
            tuple[int, int, int | None], frozenset[int]
        ] = OrderedDict()

    def sample(
        self,
        source_count: int,
        rng: np.random.Generator,
        *,
        min_distance: int = 0,
        max_distance: int | None = None,
        max_attempts: int = 100,
    ) -> frozenset[int]:
        """Build a source set sequentially from compatible candidate pools."""
        _validate_source_constraints(
            self.graph, source_count, min_distance, max_distance
        )
        for _ in range(max_attempts):
            selected = [int(rng.choice(self.nodes))]
            while len(selected) < source_count:
                compatible = set(self.nodes.tolist())
                for source in selected:
                    compatible.intersection_update(
                        self._compatible_nodes(source, min_distance, max_distance)
                    )
                compatible.difference_update(selected)
                if not compatible:
                    break
                selected.append(int(rng.choice(np.asarray(sorted(compatible)))))
            if len(selected) == source_count:
                return frozenset(selected)
        raise RuntimeError("Could not sample sources satisfying distance constraints.")

    def _compatible_nodes(
        self, source: int, min_distance: int, max_distance: int | None
    ) -> frozenset[int]:
        key = (source, min_distance, max_distance)
        cached = self._compatible_cache.get(key)
        if cached is not None:
            self._compatible_cache.move_to_end(key)
            return cached

        cutoff = max_distance
        distances = nx.single_source_shortest_path_length(
            self.graph, source, cutoff=cutoff
        )
        compatible = frozenset(
            node
            for node, distance in distances.items()
            if distance >= min_distance
            and (max_distance is None or distance <= max_distance)
            and node != source
        )
        self._compatible_cache[key] = compatible
        if len(self._compatible_cache) > self.cache_size:
            self._compatible_cache.popitem(last=False)
        return compatible


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
    return SourceSampler(graph).sample(
        source_count,
        rng,
        min_distance=min_distance,
        max_distance=max_distance,
        max_attempts=max_attempts,
    )


def _validate_source_constraints(
    graph: nx.Graph,
    source_count: int,
    min_distance: int,
    max_distance: int | None,
) -> None:
    if not 1 <= source_count <= 3:
        raise ValueError("source_count must be between 1 and 3.")
    if source_count > graph.number_of_nodes():
        raise ValueError("source_count cannot exceed the graph node count.")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")
    if max_distance is not None and max_distance < min_distance:
        raise ValueError("max_distance must be greater than or equal to min_distance.")


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
