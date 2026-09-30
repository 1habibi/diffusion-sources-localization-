"""Frozen demo inference compares two rankings without exposing targets."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest
import torch

from diffusion_sources.diffusion import Cascade
from diffusion_sources.features import SnapshotFeatureBuilder
from diffusion_sources.observations import Observation
from diffusion_sources.temporal_demo_assets import DemoResources, FEATURE_NAMES
from diffusion_sources.temporal_demo_inference import infer_demo
from diffusion_sources.temporal_demo_scenario import DemoScenario
from diffusion_sources.temporal_scoring import correct_sources


class FrozenStub:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, data):
        assert not hasattr(data, "source_labels")
        assert not hasattr(data, "source_count")
        assert data.x.shape == (5, 5)
        assert int(data.candidate_mask.sum()) == 5
        self.calls += 1
        return torch.tensor([.9, .8, .2, .1, .3]), torch.tensor([[0., 10., 0.]])


def _case(early: frozenset[int] = frozenset({4})):
    graph = nx.path_graph(5)
    cascade = Cascade(frozenset({0, 4}), {0: 0, 4: 0, 1: 1, 3: 1, 2: 2},
                      (frozenset({0, 4}), frozenset({1, 3}), frozenset({2})), .02, 3)
    final = Observation(frozenset(range(5)), frozenset(), frozenset(range(5)),
                        frozenset(), 1.0, 0, False)
    scenario = DemoScenario(cascade, final, early, 20, 21, 1)
    model = FrozenStub()
    resources = DemoResources(graph, model, SnapshotFeatureBuilder(graph), {},
                              {"data": {"feature_names": list(FEATURE_NAMES)}})
    return resources, scenario, model


def test_targets_removed_before_model_call() -> None:
    resources, scenario, model = _case()
    result = infer_demo(resources, scenario)
    assert model.calls == 1
    assert result.snapshot.source_count == 2


def test_beta_zero_equals_snapshot() -> None:
    resources, scenario, _ = _case()
    result = infer_demo(resources, scenario)
    assert correct_sources(result.candidate_scores, 0) == result.snapshot.sources


def test_temporal_keeps_estimated_count_and_candidates() -> None:
    resources, scenario, _ = _case()
    result = infer_demo(resources, scenario)
    assert len(result.temporal_sources) == result.snapshot.source_count == 2
    assert result.temporal_sources <= set(result.candidate_scores.candidate_ids)
    assert result.temporal_sources == frozenset({0, 4})


def test_empty_early_equals_snapshot() -> None:
    resources, scenario, _ = _case(frozenset())
    result = infer_demo(resources, replace(scenario, early_nodes=frozenset()))
    assert result.temporal_sources == result.snapshot.sources


def test_metrics_match_source_sets() -> None:
    resources, scenario, _ = _case()
    result = infer_demo(resources, scenario)
    assert result.snapshot.sources == frozenset({0, 1})
    assert result.snapshot_metrics["f1"] == pytest.approx(.5)
    assert result.snapshot_metrics["exact_set_accuracy"] == 0
    assert result.snapshot_metrics["symmetric_set_distance"] == pytest.approx(1.0)
    assert result.temporal_metrics["f1"] == pytest.approx(1.0)
    assert result.temporal_metrics["exact_set_accuracy"] == 1
    assert result.temporal_metrics["symmetric_set_distance"] == 0
