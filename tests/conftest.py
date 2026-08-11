from __future__ import annotations

import networkx as nx
import pytest


@pytest.fixture
def path_graph() -> nx.Graph:
    return nx.path_graph(5)


@pytest.fixture
def cycle_graph() -> nx.Graph:
    return nx.cycle_graph(5)


@pytest.fixture
def disconnected_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([(10, 11), (11, 12), (100, 101)])
    graph.add_edge(12, 12)
    return graph
