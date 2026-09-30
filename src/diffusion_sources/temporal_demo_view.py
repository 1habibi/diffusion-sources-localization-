"""Small, stable drawing of a cascade; inference always uses the full graph."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import plotly.graph_objects as go

from .temporal_demo_inference import TemporalDemoResult


@dataclass(frozen=True)
class DemoGraphView:
    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    positions: dict[int, tuple[float, float]]
    total_count: int
    selected_node: int | None

    @property
    def displayed_count(self) -> int:
        return len(self.node_ids)


def build_demo_graph_view(
    result: TemporalDemoResult, selected_node: int | None, max_nodes: int = 300
) -> DemoGraphView:
    """Prefer cascade context, but never clip truth/predictions/selection."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    graph = result.graph
    if selected_node is not None and selected_node not in graph:
        raise ValueError("selected_node is not in the graph")
    mandatory = (
        set(result.scenario.cascade.sources)
        | set(result.snapshot.sources)
        | set(result.temporal_sources)
    )
    if selected_node is not None:
        mandatory.add(selected_node)
    chosen = set(mandatory)
    for node in sorted(result.scenario.final.observed_infected):
        if len(chosen) >= max_nodes:
            break
        chosen.add(node)
    for node in sorted(mandatory):
        for neighbor in sorted(graph.neighbors(node)):
            if len(chosen) >= max_nodes:
                break
            chosen.add(neighbor)
        if len(chosen) >= max_nodes:
            break
    for node in sorted(graph.nodes):
        if len(chosen) >= max_nodes:
            break
        chosen.add(node)
    ids = tuple(sorted(chosen))
    subgraph = graph.subgraph(ids)
    positions = nx.spring_layout(subgraph, seed=17)
    return DemoGraphView(
        ids,
        tuple(sorted((min(a, b), max(a, b)) for a, b in subgraph.edges)),
        {node: (float(xy[0]), float(xy[1])) for node, xy in positions.items()},
        graph.number_of_nodes(),
        selected_node,
    )


def plot_demo_graph(
    view: DemoGraphView,
    result: TemporalDemoResult,
    method: str,
    frame: str,
    show_truth: bool,
) -> go.Figure:
    """Render two traces: edges, then nodes with original IDs in customdata."""
    if method not in {"snapshot", "temporal"}:
        raise ValueError("method must be snapshot or temporal")
    if frame not in {"early", "final", "scores"}:
        raise ValueError("frame must be early, final or scores")
    predicted = result.snapshot.sources if method == "snapshot" else result.temporal_sources
    truth = result.scenario.cascade.sources
    observed = result.scenario.early_nodes if frame == "early" else result.scenario.final.observed_infected
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for a, b in view.edges:
        edge_x.extend((view.positions[a][0], view.positions[b][0], None))
        edge_y.extend((view.positions[a][1], view.positions[b][1], None))
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="skip", showlegend=False,
        line={"color": "#64748b", "width": 0.8}, opacity=0.35,
    )
    raw_scores = result.snapshot.scores
    score_by_node = {node: float(raw_scores[node]) for node in view.node_ids}
    low = min(score_by_node.values(), default=0.0)
    high = max(score_by_node.values(), default=1.0)
    def color(node: int) -> str:
        if frame == "scores":
            intensity = (score_by_node[node] - low) / (high - low) if high > low else 0.5
            return f"rgb({int(219 - 178 * intensity)},{int(234 - 120 * intensity)},{int(254 - 75 * intensity)})"
        return "#2563eb" if node in observed else "#cbd5e1"
    node_trace = go.Scatter(
        x=[view.positions[node][0] for node in view.node_ids],
        y=[view.positions[node][1] for node in view.node_ids],
        mode="markers", showlegend=False,
        customdata=list(view.node_ids),
        text=[f"Узел {node}<br>Сырой балл GCN: {score_by_node[node]:.3f}" for node in view.node_ids],
        hovertemplate="%{text}<extra></extra>",
        marker={
            "color": [color(node) for node in view.node_ids],
            "symbol": ["diamond" if node in predicted else "circle" for node in view.node_ids],
            "size": [16 if node == view.selected_node else (11 if node in predicted else 7) for node in view.node_ids],
            "line": {
                "color": ["#16a34a" if show_truth and node in truth else "#475569" for node in view.node_ids],
                "width": [3 if show_truth and node in truth else 1 for node in view.node_ids],
            },
        },
    )
    figure = go.Figure((edge_trace, node_trace))
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        dragmode="select", height=590,
    )
    return figure
