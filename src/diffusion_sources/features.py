"""Node features shared by all source-localization models."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

from .observations import Observation


BASE_FEATURE_NAMES = ("observed_infected", "log_degree_normalized")
LOCAL_STRUCTURAL_FEATURE_NAMES = (
    "observed_neighbor_count_normalized",
    "observed_neighbor_fraction",
    "unobserved_neighbor_count_normalized",
    "unobserved_neighbor_fraction",
    "closed_neighborhood_observed_fraction",
)
DISTANCE_POSITION_FEATURE_NAMES = (
    "distance_to_observation_boundary_normalized",
    "observation_boundary_missing",
    "mean_distance_to_observed_normalized",
    "max_distance_to_observed_normalized",
    "induced_observed_eccentricity_normalized",
)
GLOBAL_SCALAR_FEATURE_NAMES = (
    "observed_count_normalized",
    "candidate_count_normalized",
    "observed_candidate_fraction",
    "observed_subgraph_density",
    "observed_component_count_normalized",
    "observed_largest_component_fraction",
)
JORDAN_FEATURE_NAMES = ("multi_jordan_rank_normalized",)
SNAPSHOT_FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + LOCAL_STRUCTURAL_FEATURE_NAMES
    + DISTANCE_POSITION_FEATURE_NAMES
    + GLOBAL_SCALAR_FEATURE_NAMES
    + JORDAN_FEATURE_NAMES
)


def node_features(graph: nx.Graph, observation: Observation) -> np.ndarray:
    """Build [observed_infected, normalized_log_degree] features by node ID."""
    nodes = sorted(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError("Graph nodes must be contiguous integers starting at zero.")

    max_degree = max((graph.degree(node) for node in nodes), default=0)
    denominator = math.log1p(max_degree) or 1.0
    features = np.zeros((len(nodes), 2), dtype=np.float32)
    for node in nodes:
        features[node, 0] = float(node in observation.observed_infected)
        features[node, 1] = math.log1p(graph.degree(node)) / denominator
    return features


class SnapshotFeatureBuilder:
    """Build named leakage-safe node features from one observed snapshot."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        distance_cache_path: str | Path | None = None,
        distance_cap: int = 10,
    ) -> None:
        nodes = sorted(graph.nodes())
        if nodes != list(range(len(nodes))):
            raise ValueError("Graph nodes must be contiguous integers starting at zero.")
        self.node_count = len(nodes)
        if distance_cap < 1:
            raise ValueError("distance_cap must be positive.")
        self.graph = graph
        self.distance_cap = int(distance_cap)
        self.distance_cache_path = (
            Path(distance_cache_path) if distance_cache_path is not None else None
        )
        self._distance_matrix: np.ndarray | None = None
        self.degrees = np.asarray([graph.degree(node) for node in nodes], dtype=np.float32)
        self.max_degree = float(self.degrees.max()) if len(self.degrees) else 0.0
        edges = np.asarray(sorted(graph.edges()), dtype=np.int64).reshape(-1, 2)
        if len(edges):
            self.edge_sources = np.concatenate([edges[:, 0], edges[:, 1]])
            self.edge_targets = np.concatenate([edges[:, 1], edges[:, 0]])
        else:
            self.edge_sources = np.empty(0, dtype=np.int64)
            self.edge_targets = np.empty(0, dtype=np.int64)
        self.graph_fingerprint = hashlib.sha256(edges.tobytes()).hexdigest()

    def build(
        self,
        observed_mask: np.ndarray,
        feature_names: list[str] | tuple[str, ...],
        *,
        base_features: np.ndarray | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return requested columns in a stable, explicit order."""
        names = tuple(feature_names)
        if not names:
            raise ValueError("feature_names must not be empty.")
        unknown = sorted(set(names) - set(SNAPSHOT_FEATURE_NAMES))
        if unknown:
            raise ValueError(f"Unknown snapshot features: {unknown}")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must not contain duplicates.")

        observed = np.asarray(observed_mask, dtype=bool)
        if observed.shape != (self.node_count,):
            raise ValueError("observed_mask does not match graph node count.")
        if base_features is not None and base_features.shape[:1] != (self.node_count,):
            raise ValueError("base_features does not match graph node count.")
        candidates = (
            np.asarray(candidate_mask, dtype=bool)
            if candidate_mask is not None
            else None
        )
        if candidates is not None and candidates.shape != (self.node_count,):
            raise ValueError("candidate_mask does not match graph node count.")

        observed_neighbor_count = np.zeros(self.node_count, dtype=np.float32)
        if len(self.edge_sources):
            observed_neighbor_count = np.bincount(
                self.edge_sources,
                weights=observed[self.edge_targets].astype(np.float32),
                minlength=self.node_count,
            ).astype(np.float32, copy=False)
        unobserved_neighbor_count = self.degrees - observed_neighbor_count
        degree_denominator = np.where(self.degrees > 0, self.degrees, 1.0)
        log_denominator = math.log1p(self.max_degree) or 1.0

        values = {
            "observed_infected": observed.astype(np.float32),
            "log_degree_normalized": (
                np.asarray(base_features[:, 1], dtype=np.float32)
                if base_features is not None and base_features.shape[1] >= 2
                else np.log1p(self.degrees) / log_denominator
            ),
            "observed_neighbor_count_normalized": np.log1p(observed_neighbor_count)
            / log_denominator,
            "observed_neighbor_fraction": observed_neighbor_count / degree_denominator,
            "unobserved_neighbor_count_normalized": np.log1p(unobserved_neighbor_count)
            / log_denominator,
            "unobserved_neighbor_fraction": unobserved_neighbor_count
            / degree_denominator,
            "closed_neighborhood_observed_fraction": (
                observed_neighbor_count + observed.astype(np.float32)
            )
            / (self.degrees + 1.0),
        }
        if set(names) & set(DISTANCE_POSITION_FEATURE_NAMES):
            values.update(
                self._build_distance_position_features(
                    observed, observed_neighbor_count
                )
            )
        if set(names) & set(GLOBAL_SCALAR_FEATURE_NAMES):
            if candidates is None:
                raise ValueError("Global scalar features require candidate_mask.")
            values.update(self._build_global_scalar_features(observed, candidates))
        if set(names) & set(JORDAN_FEATURE_NAMES):
            if candidates is None:
                raise ValueError("Multi-Jordan features require candidate_mask.")
            values.update(self._build_multi_jordan_features(observed, candidates))
        return np.column_stack([values[name] for name in names]).astype(
            np.float32, copy=False
        )

    def _build_global_scalar_features(
        self, observed: np.ndarray, candidates: np.ndarray
    ) -> dict[str, np.ndarray]:
        observed_nodes = np.flatnonzero(observed)
        candidate_count = int(candidates.sum())
        observed_count = len(observed_nodes)
        if observed_count < 1 or candidate_count < 1:
            raise ValueError("Global features require observed nodes and candidates.")

        observed_subgraph = self.graph.subgraph(observed_nodes)
        component_sizes = [
            len(component) for component in nx.connected_components(observed_subgraph)
        ]
        component_count = len(component_sizes)
        possible_edges = observed_count * (observed_count - 1) / 2
        density = (
            observed_subgraph.number_of_edges() / possible_edges
            if possible_edges
            else 0.0
        )
        node_count_denominator = math.log1p(self.node_count) or 1.0
        component_denominator = math.log1p(observed_count) or 1.0
        scalars = {
            "observed_count_normalized": math.log1p(observed_count)
            / node_count_denominator,
            "candidate_count_normalized": math.log1p(candidate_count)
            / node_count_denominator,
            "observed_candidate_fraction": observed_count / candidate_count,
            "observed_subgraph_density": density,
            "observed_component_count_normalized": math.log1p(component_count)
            / component_denominator,
            "observed_largest_component_fraction": max(component_sizes)
            / observed_count,
        }
        return {
            name: np.full(self.node_count, value, dtype=np.float32)
            for name, value in scalars.items()
        }

    def _build_distance_position_features(
        self, observed: np.ndarray, observed_neighbor_count: np.ndarray
    ) -> dict[str, np.ndarray]:
        observed_nodes = np.flatnonzero(observed)
        if not len(observed_nodes):
            raise ValueError("Distance features require at least one observed node.")
        distances = self._get_distance_matrix()
        capped_to_observed = np.minimum(
            distances[:, observed_nodes], self.distance_cap
        ).astype(np.float32)
        mean_distance = capped_to_observed.mean(axis=1) / self.distance_cap
        max_distance = capped_to_observed.max(axis=1) / self.distance_cap

        boundary_nodes = observed_nodes[
            self.degrees[observed_nodes]
            > observed_neighbor_count[observed_nodes]
        ]
        boundary_missing = float(not len(boundary_nodes))
        if len(boundary_nodes):
            boundary_distance = (
                np.minimum(distances[:, boundary_nodes].min(axis=1), self.distance_cap)
                .astype(np.float32)
                / self.distance_cap
            )
        else:
            boundary_distance = np.ones(self.node_count, dtype=np.float32)

        induced_eccentricity = np.zeros(self.node_count, dtype=np.float32)
        observed_subgraph = self.graph.subgraph(observed_nodes)
        for component_nodes in nx.connected_components(observed_subgraph):
            component = observed_subgraph.subgraph(component_nodes)
            for node, node_distances in nx.all_pairs_shortest_path_length(component):
                induced_eccentricity[node] = min(
                    max(node_distances.values(), default=0), self.distance_cap
                ) / self.distance_cap

        return {
            "distance_to_observation_boundary_normalized": boundary_distance,
            "observation_boundary_missing": np.full(
                self.node_count, boundary_missing, dtype=np.float32
            ),
            "mean_distance_to_observed_normalized": mean_distance,
            "max_distance_to_observed_normalized": max_distance,
            "induced_observed_eccentricity_normalized": induced_eccentricity,
        }

    def _build_multi_jordan_features(
        self, observed: np.ndarray, candidates: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Encode the complete greedy Multi-Jordan candidate ordering."""
        observed_nodes = np.flatnonzero(observed)
        candidate_nodes = np.flatnonzero(candidates)
        if not len(observed_nodes) or not len(candidate_nodes):
            raise ValueError(
                "Multi-Jordan features require observed nodes and candidates."
            )
        distances = self._get_distance_matrix()[
            np.ix_(candidate_nodes, observed_nodes)
        ].astype(np.int64)
        nearest = np.full(len(observed_nodes), np.iinfo(np.int64).max)
        available = np.ones(len(candidate_nodes), dtype=bool)
        scores = np.zeros(self.node_count, dtype=np.float32)
        denominator = max(len(candidate_nodes) - 1, 1)

        for rank in range(len(candidate_nodes)):
            available_indices = np.flatnonzero(available)
            covered = np.minimum(
                distances[available_indices], nearest[None, :]
            )
            eccentricities = covered.max(axis=1)
            total_distances = covered.sum(axis=1)
            ordered = np.lexsort(
                (
                    candidate_nodes[available_indices],
                    total_distances,
                    eccentricities,
                )
            )
            chosen_index = available_indices[int(ordered[0])]
            chosen_node = candidate_nodes[chosen_index]
            scores[chosen_node] = 1.0 - rank / denominator
            nearest = np.minimum(nearest, distances[chosen_index])
            available[chosen_index] = False

        return {"multi_jordan_rank_normalized": scores}

    def _get_distance_matrix(self) -> np.ndarray:
        if self._distance_matrix is not None:
            return self._distance_matrix
        if self.distance_cache_path is not None and self.distance_cache_path.exists():
            with np.load(self.distance_cache_path, allow_pickle=False) as archive:
                fingerprint = str(archive["graph_fingerprint"].item())
                matrix = np.asarray(archive["distances"], dtype=np.uint16)
            if fingerprint != self.graph_fingerprint:
                raise ValueError("Distance cache graph fingerprint does not match.")
            if matrix.shape != (self.node_count, self.node_count):
                raise ValueError("Distance cache shape does not match graph node count.")
            self._distance_matrix = matrix
            return matrix

        adjacency = nx.to_scipy_sparse_array(
            self.graph,
            nodelist=range(self.node_count),
            dtype=np.float32,
            format="csr",
        )
        unreachable = np.iinfo(np.uint16).max
        matrix = np.empty((self.node_count, self.node_count), dtype=np.uint16)
        chunk_size = 256
        for start in range(0, self.node_count, chunk_size):
            stop = min(start + chunk_size, self.node_count)
            chunk = shortest_path(
                adjacency,
                directed=False,
                unweighted=True,
                indices=np.arange(start, stop),
            )
            chunk = np.where(np.isfinite(chunk), chunk, unreachable)
            matrix[start:stop] = chunk.astype(np.uint16)
        self._distance_matrix = matrix
        if self.distance_cache_path is not None:
            self.distance_cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.distance_cache_path.with_suffix(
                self.distance_cache_path.suffix + ".tmp"
            )
            with temporary_path.open("wb") as file:
                np.savez_compressed(
                    file,
                    graph_fingerprint=np.asarray(self.graph_fingerprint),
                    distances=matrix,
                )
            temporary_path.replace(self.distance_cache_path)
        return matrix
