from __future__ import annotations

import networkx as nx

from diffusion_sources.graphs import prepare_graph


def test_prepare_graph_keeps_largest_component_and_removes_loops(disconnected_graph):
    graph, mapping = prepare_graph(disconnected_graph)

    assert set(graph.nodes()) == {0, 1, 2}
    assert set(mapping) == {10, 11, 12}
    assert graph.number_of_edges() == 2
    assert not list(nx.selfloop_edges(graph))
