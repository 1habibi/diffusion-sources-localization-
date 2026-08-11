"""Node features shared by all source-localization models."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from .observations import Observation


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
