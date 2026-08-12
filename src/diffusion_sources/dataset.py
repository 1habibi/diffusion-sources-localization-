"""Compact dataset records for cascades that share a graph topology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from .diffusion import Cascade
from .features import node_features
from .observations import Observation


@dataclass(frozen=True)
class CascadeExample:
    """One model-ready cascade observation with multi-source targets."""

    graph_id: str
    features: np.ndarray
    candidate_mask: np.ndarray
    source_labels: np.ndarray
    infected_mask: np.ndarray
    source_count: int
    simulation: dict[str, Any]
    observation: dict[str, Any]


def build_example(
    graph_id: str,
    graph: nx.Graph,
    cascade: Cascade,
    observation: Observation,
    *,
    simulation_seed: int,
    observation_seed: int,
) -> CascadeExample:
    """Build compact arrays without duplicating the graph topology."""
    n_nodes = graph.number_of_nodes()
    labels = np.zeros(n_nodes, dtype=np.float32)
    infected_mask = np.zeros(n_nodes, dtype=bool)
    mask = np.zeros(n_nodes, dtype=bool)
    labels[list(cascade.sources)] = 1.0
    infected_mask[list(cascade.infected)] = True
    mask[list(observation.candidate_nodes)] = True

    return CascadeExample(
        graph_id=graph_id,
        features=node_features(graph, observation),
        candidate_mask=mask,
        source_labels=labels,
        infected_mask=infected_mask,
        source_count=len(cascade.sources),
        simulation={
            "probability": cascade.transmission_probability,
            "max_steps": cascade.max_steps,
            "simulation_seed": simulation_seed,
        },
        observation={
            "observation_fraction": observation.observation_fraction,
            "false_positive_count": observation.false_positive_count,
            "observation_seed": observation_seed,
            "hidden_sources": sorted(observation.hidden_sources),
        },
    )


def graph_to_edge_index(graph: nx.Graph) -> torch.Tensor:
    """Convert an undirected NetworkX graph to a bidirectional edge index."""
    edges = sorted(graph.edges())
    directed_edges = edges + [(target, source) for source, target in edges]
    if not directed_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()


def example_to_pyg(graph: nx.Graph, example: CascadeExample) -> Data:
    """Convert one compact example to a PyG Data object."""
    if graph.number_of_nodes() != len(example.features):
        raise ValueError("Graph and example node counts do not match.")
    return Data(
        x=torch.from_numpy(example.features),
        edge_index=graph_to_edge_index(graph),
        candidate_mask=torch.from_numpy(example.candidate_mask),
        source_labels=torch.from_numpy(example.source_labels),
        source_count=torch.tensor(example.source_count, dtype=torch.long),
        observed_mask=torch.from_numpy(example.features[:, 0].astype(bool)),
    )


def load_graph_archive(path: str | Path) -> tuple[str, nx.Graph]:
    """Load the topology archive written by the generation CLI."""
    archive = np.load(Path(path), allow_pickle=False)
    graph_id = str(archive["graph_id"].item())
    node_count = int(archive["node_count"].item())
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(archive["edges"].tolist())
    return graph_id, graph


def load_pyg_split(
    path: str | Path,
    graph: nx.Graph,
    feature_indices: list[int] | None = None,
    limit: int | None = None,
) -> list[Data]:
    """Load a generated split archive into independent PyG examples."""
    archive = np.load(Path(path), allow_pickle=False)
    features = archive["features"]
    candidate_masks = archive["candidate_masks"]
    source_labels = archive["source_labels"]
    source_counts = archive["source_counts"]
    lengths = {len(features), len(candidate_masks), len(source_labels), len(source_counts)}
    if len(lengths) != 1:
        raise ValueError("Generated split arrays have inconsistent lengths.")
    if features.shape[1] != graph.number_of_nodes():
        raise ValueError("Generated split does not match graph node count.")
    if feature_indices is None:
        feature_indices = list(range(features.shape[2]))
    if not feature_indices or min(feature_indices) < 0 or max(feature_indices) >= features.shape[2]:
        raise ValueError("feature_indices must select available feature columns.")

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided.")
    example_count = min(len(features), limit) if limit is not None else len(features)
    edge_index = graph_to_edge_index(graph)
    examples: list[Data] = []
    for index in range(example_count):
        examples.append(
            Data(
                x=torch.from_numpy(features[index][:, feature_indices]).float(),
                edge_index=edge_index,
                candidate_mask=torch.from_numpy(candidate_masks[index]).bool(),
                source_labels=torch.from_numpy(source_labels[index]).float(),
                source_count=torch.tensor(int(source_counts[index]), dtype=torch.long),
                observed_mask=torch.from_numpy(features[index, :, 0].astype(bool)),
            )
        )
    return examples
