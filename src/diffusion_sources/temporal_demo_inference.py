"""Paired Snapshot and frozen Temporal-v3 inference for one demo cascade."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from .dataset import graph_to_edge_index
from .inference import SourcePrediction, predict_joint
from .metrics import set_metrics, source_set_distances
from .temporal_demo_assets import DemoResources
from .temporal_demo_scenario import DemoScenario
from .temporal_scoring import CandidateScores, correct_sources


@dataclass(frozen=True)
class TemporalDemoResult:
    graph: nx.Graph
    scenario: DemoScenario
    snapshot: SourcePrediction
    temporal_sources: frozenset[int]
    candidate_scores: CandidateScores
    snapshot_metrics: dict[str, float]
    temporal_metrics: dict[str, float]


def infer_demo(resources: DemoResources, scenario: DemoScenario) -> TemporalDemoResult:
    """Run the frozen model once; never include simulator targets in model data."""
    graph = resources.graph
    n = graph.number_of_nodes()
    final = scenario.final
    observed = np.zeros(n, dtype=bool)
    candidates = np.zeros(n, dtype=bool)
    observed[list(final.observed_infected)] = True
    candidates[list(final.candidate_nodes)] = True
    names = resources.model_config["data"]["feature_names"]
    features = resources.builder.build(observed, names, candidate_mask=candidates)
    data = Data(
        x=torch.from_numpy(features),
        edge_index=graph_to_edge_index(graph),
        candidate_mask=torch.from_numpy(candidates),
        observed_mask=torch.from_numpy(observed),
    )
    with torch.inference_mode():
        source_logits, count_logits = resources.model(data)
        snapshot = predict_joint(source_logits, count_logits, data.candidate_mask)
    ids = tuple(np.flatnonzero(candidates).tolist())
    candidate_scores = CandidateScores(
        ids,
        tuple(float(snapshot.scores[node]) for node in ids),
        tuple(node in scenario.early_nodes for node in ids),
        snapshot.sources,
        snapshot.source_count,
    )
    temporal = correct_sources(candidate_scores, .5)
    truth = scenario.cascade.sources
    snapshot_metrics = {**set_metrics(truth, snapshot.sources),
                        **source_set_distances(graph, truth, snapshot.sources)}
    temporal_metrics = {**set_metrics(truth, temporal),
                        **source_set_distances(graph, truth, temporal)}
    return TemporalDemoResult(graph, scenario, snapshot, temporal, candidate_scores,
                              snapshot_metrics, temporal_metrics)
