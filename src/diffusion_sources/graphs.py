"""Graph loading, preprocessing, and small synthetic graph helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import networkx as nx


def prepare_graph(graph: nx.Graph) -> tuple[nx.Graph, dict[Hashable, int]]:
    """Return the largest connected simple component with contiguous integer IDs."""
    prepared = nx.Graph(graph)
    prepared.remove_edges_from(nx.selfloop_edges(prepared))

    if prepared.number_of_nodes() == 0:
        raise ValueError("Graph must contain at least one node.")

    largest_component = max(nx.connected_components(prepared), key=len)
    prepared = prepared.subgraph(largest_component).copy()
    ordered_nodes = sorted(prepared.nodes(), key=str)
    mapping = {node: index for index, node in enumerate(ordered_nodes)}
    return nx.relabel_nodes(prepared, mapping, copy=True), mapping


def load_edge_list(path: str | Path) -> tuple[nx.Graph, dict[Hashable, int]]:
    """Load an undirected whitespace-delimited edge list and preprocess it."""
    graph = nx.read_edgelist(Path(path), nodetype=str, data=False)
    return prepare_graph(graph)


def karate_graph() -> nx.Graph:
    """Return the preprocessed Zachary Karate Club graph."""
    graph, _ = prepare_graph(nx.karate_club_graph())
    return graph


def erdos_renyi_graph(
    n: int, probability: float, seed: int | None = None
) -> nx.Graph:
    """Generate and preprocess an Erdos-Renyi graph."""
    graph, _ = prepare_graph(nx.erdos_renyi_graph(n, probability, seed=seed))
    return graph


def barabasi_albert_graph(
    n: int, attachments: int, seed: int | None = None
) -> nx.Graph:
    """Generate and preprocess a Barabasi-Albert graph."""
    graph, _ = prepare_graph(nx.barabasi_albert_graph(n, attachments, seed=seed))
    return graph
