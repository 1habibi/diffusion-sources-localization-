"""The Streamlit workbench keeps one honest, successful demo result."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest
import torch
from streamlit.testing.v1 import AppTest

import app
from diffusion_sources.diffusion import Cascade
from diffusion_sources.inference import SourcePrediction
from diffusion_sources.observations import Observation
from diffusion_sources.temporal_demo_inference import TemporalDemoResult
from diffusion_sources.temporal_demo_scenario import DemoScenario
from diffusion_sources.temporal_scoring import CandidateScores


def _result(true_k: int = 2, predicted_k: int = 1) -> TemporalDemoResult:
    graph = nx.path_graph(10)
    truth = frozenset(range(true_k))
    cascade = Cascade(truth, {node: 0 if node in truth else 1 for node in graph},
                      (truth, frozenset(set(graph) - truth)), .02, 3)
    observed = frozenset({0, 1, 2, 3, 4, 5})
    final = Observation(observed, frozenset(), observed, frozenset(), .75, 0, False)
    scenario = DemoScenario(cascade, final, frozenset({1}), 123, 456, 2)
    scores = torch.linspace(.1, .9, 10)
    snapshot = SourcePrediction(scores, predicted_k, frozenset({4}))
    candidates = CandidateScores(tuple(sorted(observed)), tuple(float(scores[i]) for i in sorted(observed)),
                                 tuple(i == 1 for i in sorted(observed)), snapshot.sources, predicted_k)
    return TemporalDemoResult(graph, scenario, snapshot, frozenset({1}), candidates,
                              {"f1": .25, "exact_set_accuracy": 0., "symmetric_set_distance": 1.5},
                              {"f1": .5, "exact_set_accuracy": 0., "symmetric_set_distance": 1.0})


@pytest.fixture
def fake_services(monkeypatch: pytest.MonkeyPatch):
    calls = {"generation": 0, "inference": 0}
    resources = SimpleNamespace(graph=nx.path_graph(10))
    monkeypatch.setattr(app, "load_resources", lambda: resources)

    def generate(_resources, settings):
        calls["generation"] += 1
        return _result(settings.true_k).scenario

    def infer(_resources, scenario):
        calls["inference"] += 1
        return _result(len(scenario.cascade.sources))

    monkeypatch.setattr(app, "generate_demo_scenario", generate, raising=False)
    monkeypatch.setattr(app, "infer_demo", infer, raising=False)
    return calls


def _page() -> AppTest:
    return AppTest.from_string("from app import main\nmain()", default_timeout=30).run()


def test_initial_screen_has_no_fake_metrics(fake_services) -> None:
    page = _page()
    assert not page.exception
    assert len(page.metric) == 0
    assert fake_services == {"generation": 0, "inference": 0}


def test_success_persists_on_switch(fake_services) -> None:
    page = _page()
    page.button(key="run_demo").click().run()
    assert not page.exception
    assert fake_services == {"generation": 1, "inference": 1}
    original = page.session_state["demo_result"]
    page.radio(key="view_method").set_value("Temporal-v3").run()
    page.radio(key="view_frame").set_value("t = 1").run()
    assert page.session_state["demo_result"] is original
    assert fake_services == {"generation": 1, "inference": 1}


def test_failed_new_run_keeps_previous_result(fake_services, monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page()
    page.button(key="run_demo").click().run()
    original = page.session_state["demo_result"]

    def broken(*_args):
        raise RuntimeError("No valid demo cascade after 100 attempts")

    monkeypatch.setattr(app, "generate_demo_scenario", broken)
    page.button(key="run_demo").click().run()
    assert page.session_state["demo_result"] is original
    assert any("100" in error.value for error in page.error)
    assert any(metric.label == "F1" for metric in page.metric)


def test_missing_backup_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "load_resources", lambda: (_ for _ in ()).throw(FileNotFoundError("best_model.pt")))
    page = _page()
    page.button(key="run_demo").click().run()
    assert not page.exception
    assert any("best_model.pt" in error.value for error in page.error)
    assert len(page.metric) == 0


def test_predicted_k_is_not_control_value(fake_services) -> None:
    page = _page()
    page.button(key="run_demo").click().run()
    values = {metric.label: metric.value for metric in page.metric}
    assert values["Предсказанное k"] == "1"
    assert values["Истинное k"] == "2"


def test_graph_selection_does_not_override_later_selectbox_choice(fake_services, monkeypatch: pytest.MonkeyPatch) -> None:
    retained_event = SimpleNamespace(selection=SimpleNamespace(points=[{"customdata": 1}]))
    monkeypatch.setattr(app.st, "plotly_chart", lambda *args, **kwargs: retained_event)
    page = _page()
    page.button(key="run_demo").click().run()
    assert page.session_state["selected_node"] == 1
    page.selectbox(key="selected_node").set_value(2).run()
    assert not page.exception
    assert page.session_state["selected_node"] == 2
