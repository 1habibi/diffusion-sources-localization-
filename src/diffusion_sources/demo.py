"""Reusable end-to-end inference service used by the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from .dataset import build_example, example_to_pyg, load_graph_archive
from .diffusion import Cascade, simulate_ic
from .inference import SourcePrediction, predict_joint
from .metrics import set_metrics, source_radius_hits, source_set_distances
from .models import JointSourceCountGCN
from .observations import Observation, observe_cascade


@dataclass(frozen=True)
class DemoResult:
    graph: nx.Graph
    cascade: Cascade
    observation: Observation
    prediction: SourcePrediction
    metrics: dict[str, float]


def load_demo_model(
    run_dir: str | Path,
) -> tuple[JointSourceCountGCN, list[int]]:
    """Restore the selected checkpoint and its feature configuration."""
    run_path = Path(run_dir)
    config = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    metrics = __import__("json").loads(
        (run_path / "metrics.json").read_text(encoding="utf-8")
    )
    feature_indices = [int(value) for value in metrics["feature_indices"]]
    model = JointSourceCountGCN(
        input_dim=len(feature_indices),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(torch.load(run_path / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, feature_indices


def run_demo(
    graph: nx.Graph,
    model: JointSourceCountGCN,
    feature_indices: list[int],
    sources: set[int] | frozenset[int],
    *,
    probability: float,
    max_steps: int,
    observation_fraction: float,
    false_positive_count: int = 0,
    seed: int = 2026,
) -> DemoResult:
    """Simulate, observe, infer, and evaluate one user-controlled cascade."""
    cascade = simulate_ic(
        graph, sources, probability, max_steps, np.random.default_rng(seed)
    )
    observation = observe_cascade(
        graph,
        cascade,
        observation_fraction,
        false_positive_count,
        np.random.default_rng(seed + 1),
    )
    example = build_example(
        "demo",
        graph,
        cascade,
        observation,
        simulation_seed=seed,
        observation_seed=seed + 1,
    )
    data = example_to_pyg(graph, example)
    data.x = data.x[:, feature_indices]
    with torch.no_grad():
        source_logits, count_logits = model(data)
    prediction = predict_joint(source_logits, count_logits, data.candidate_mask)
    metrics = {
        **set_metrics(cascade.sources, prediction.sources),
        **source_set_distances(graph, cascade.sources, prediction.sources),
        **source_radius_hits(graph, cascade.sources, prediction.sources),
    }
    return DemoResult(graph, cascade, observation, prediction, metrics)


def load_demo_graph(data_dir: str | Path) -> nx.Graph:
    """Load the topology used by a generated dataset."""
    _, graph = load_graph_archive(Path(data_dir) / "graph.npz")
    return graph
