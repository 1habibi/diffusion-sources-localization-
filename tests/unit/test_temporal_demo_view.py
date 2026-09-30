"""A bounded graph view must not hide the sources or change inference input."""

from __future__ import annotations

import networkx as nx
import torch

from diffusion_sources.diffusion import Cascade
from diffusion_sources.inference import SourcePrediction
from diffusion_sources.observations import Observation
from diffusion_sources.temporal_demo_inference import TemporalDemoResult
from diffusion_sources.temporal_demo_scenario import DemoScenario
from diffusion_sources.temporal_demo_view import build_demo_graph_view, plot_demo_graph
from diffusion_sources.temporal_scoring import CandidateScores


def _result() -> TemporalDemoResult:
    graph = nx.path_graph(10)
    truth = frozenset({0, 8})
    cascade = Cascade(truth, {node: 0 if node in truth else 1 for node in range(10)},
                      (truth, frozenset(set(range(10)) - truth)), .02, 3)
    observed = frozenset({0, 1, 2, 3, 8, 9})
    final = Observation(observed, frozenset(), observed, frozenset(), .75, 0, False)
    scenario = DemoScenario(cascade, final, frozenset({8}), 11, 12, 1)
    snapshot = SourcePrediction(torch.linspace(.1, .9, 10), 2, frozenset({1, 8}))
    candidates = CandidateScores(tuple(sorted(observed)), tuple(float(snapshot.scores[i]) for i in sorted(observed)),
                                 tuple(i == 8 for i in sorted(observed)), snapshot.sources, 2)
    return TemporalDemoResult(graph, scenario, snapshot, frozenset({0, 9}),
                              candidates, {"f1": .5}, {"f1": .5})


def test_full_graph_not_modified_by_view() -> None:
    result = _result()
    before = (result.graph.number_of_nodes(), result.graph.number_of_edges())
    view = build_demo_graph_view(result, selected_node=6, max_nodes=3)
    assert (result.graph.number_of_nodes(), result.graph.number_of_edges()) == before == (10, 9)
    assert view.total_count == 10
    assert view.displayed_count < view.total_count


def test_truth_predictions_selection_survive_cap() -> None:
    result = _result()
    view = build_demo_graph_view(result, selected_node=6, max_nodes=3)
    assert {0, 1, 6, 8, 9} <= set(view.node_ids)
    assert view.displayed_count >= 5


def test_positions_identical_across_methods_and_frames() -> None:
    result = _result()
    view = build_demo_graph_view(result, selected_node=None, max_nodes=8)
    early = plot_demo_graph(view, result, "snapshot", "early", True)
    final = plot_demo_graph(view, result, "temporal", "final", True)
    assert list(early.data[1].x) == list(final.data[1].x)
    assert list(early.data[1].y) == list(final.data[1].y)


def test_plot_marks_match_t1_t3_truth_and_prediction() -> None:
    result = _result()
    view = build_demo_graph_view(result, selected_node=None, max_nodes=10)
    early = plot_demo_graph(view, result, "snapshot", "early", True)
    final = plot_demo_graph(view, result, "snapshot", "final", True)
    changed = view.node_ids.index(2)
    assert early.data[1].marker.color[changed] != final.data[1].marker.color[changed]
    snapshot_symbol = early.data[1].marker.symbol[view.node_ids.index(1)]
    temporal = plot_demo_graph(view, result, "temporal", "early", True)
    assert snapshot_symbol != temporal.data[1].marker.symbol[view.node_ids.index(1)]
    hidden = plot_demo_graph(view, result, "temporal", "final", False)
    assert hidden.data[1].marker.line.color[view.node_ids.index(8)] != final.data[1].marker.line.color[view.node_ids.index(8)]


def test_plot_selected_point_maps_to_original_id() -> None:
    result = _result()
    view = build_demo_graph_view(result, selected_node=None, max_nodes=8)
    figure = plot_demo_graph(view, result, "temporal", "scores", True)
    index = view.node_ids.index(9)
    assert figure.data[1].customdata[index] == 9
    assert len(figure.data[1].customdata) == view.displayed_count
