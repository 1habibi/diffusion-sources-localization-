"""Matplotlib visualizations for cascades and source scores."""

from __future__ import annotations

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .demo import DemoResult


def plot_demo_result(
    result: DemoResult,
    *,
    layout_seed: int = 17,
) -> plt.Figure:
    """Plot observed state, source scores, true sources, and predictions."""
    graph = result.graph
    positions = nx.spring_layout(graph, seed=layout_seed)
    nodes = sorted(graph.nodes())
    scores = result.prediction.scores.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    observed_colors = [
        "#E07A5F"
        if node in result.observation.observed_infected
        else "#D9E2E8"
        for node in nodes
    ]
    nx.draw_networkx_edges(graph, positions, ax=axes[0], alpha=0.25, width=0.8)
    nx.draw_networkx_nodes(
        graph, positions, nodelist=nodes, node_color=observed_colors,
        node_size=170, edgecolors="#243642", linewidths=0.5, ax=axes[0]
    )
    _draw_source_markers(axes[0], positions, result.cascade.sources, "#1B4332", "o")
    axes[0].set_title("Observed cascade; green rings are true sources")

    nx.draw_networkx_edges(graph, positions, ax=axes[1], alpha=0.2, width=0.8)
    collection = nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        node_color=[scores[node] for node in nodes],
        cmap="magma",
        vmin=0.0,
        vmax=max(1.0, float(scores.max())),
        node_size=190,
        edgecolors="#243642",
        linewidths=0.5,
        ax=axes[1],
    )
    _draw_source_markers(
        axes[1], positions, result.prediction.sources, "#00B4D8", "s"
    )
    figure.colorbar(collection, ax=axes[1], label="source score")
    axes[1].set_title(
        f"Predicted sources; cyan squares, k_hat={result.prediction.source_count}"
    )
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()
    return figure


def _draw_source_markers(axis, positions, nodes, color, marker) -> None:
    if not nodes:
        return
    coordinates = np.asarray([positions[node] for node in sorted(nodes)])
    axis.scatter(
        coordinates[:, 0], coordinates[:, 1], s=300, facecolors="none",
        edgecolors=color, linewidths=2.2, marker=marker, zorder=4
    )
