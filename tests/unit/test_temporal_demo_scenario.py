"""New demo cascades are deterministic and obey the frozen generation regime."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from diffusion_sources.diffusion import Cascade
from diffusion_sources.temporal_demo_scenario import DemoSettings, generate_demo_scenario


def _resources():
    return SimpleNamespace(
        graph=nx.path_graph(8),
        generation_config={
            "simulation": {"max_steps": 3, "distance_ranges": [{"min": 1, "max": 2}, {"min": 3, "max": 5}]},
            "observation": {"false_positive_count": 0, "hide_source_count": 0},
            "dataset": {"min_candidates": 5, "max_infected_fraction": 0.75, "distance_cache_size": 16},
        },
    )


def _cascade(sources: frozenset[int], infected: int) -> Cascade:
    nodes = set(sources)
    for node in range(8):
        if len(nodes) >= infected:
            break
        nodes.add(node)
    return Cascade(sources, {node: 0 if node in sources else 1 for node in nodes},
                   (sources, frozenset(nodes - sources)), .01, 3)


def test_same_settings_replay_same_sources_masks(monkeypatch) -> None:
    import diffusion_sources.temporal_demo_scenario as module

    monkeypatch.setattr(module, "simulate_ic", lambda graph, sources, probability, steps, rng: _cascade(sources, 6))
    settings = DemoSettings(true_k=1, probability=.01, observation_fraction=1.0, seed=2026)
    first = generate_demo_scenario(_resources(), settings)
    second = generate_demo_scenario(_resources(), settings)
    assert first.cascade.sources == second.cascade.sources
    assert first.final.observed_infected == second.final.observed_infected
    assert first.early_nodes == second.early_nodes
    assert (first.simulation_seed, first.observation_seed, first.attempt) == (
        second.simulation_seed, second.observation_seed, second.attempt)


@pytest.mark.parametrize("settings", [
    DemoSettings(0, .01, .5, 1), DemoSettings(4, .01, .5, 1),
    DemoSettings(1, .04, .5, 1), DemoSettings(1, .01, .6, 1),
    DemoSettings(1, .01, .5, -1), DemoSettings(1, .01, .5, 1.5),
])
def test_reject_invalid_controls(settings: DemoSettings) -> None:
    with pytest.raises(ValueError):
        generate_demo_scenario(_resources(), settings)


def test_reject_then_accept_is_deterministic(monkeypatch) -> None:
    import diffusion_sources.temporal_demo_scenario as module

    calls = 0

    def fake_simulation(graph, sources, probability, steps, rng):
        nonlocal calls
        calls += 1
        return _cascade(sources, 1 if calls == 1 else 6)

    monkeypatch.setattr(module, "simulate_ic", fake_simulation)
    result = generate_demo_scenario(_resources(), DemoSettings(1, .01, 1.0, 10))
    assert result.attempt == 2
    assert calls == 2
    assert len(result.final.candidate_nodes) >= 5


def test_exhaustion_is_explicit(monkeypatch) -> None:
    import diffusion_sources.temporal_demo_scenario as module

    calls = 0

    def only_source(graph, sources, probability, steps, rng):
        nonlocal calls
        calls += 1
        return _cascade(sources, 1)

    monkeypatch.setattr(module, "simulate_ic", only_source)
    with pytest.raises(RuntimeError, match="100"):
        generate_demo_scenario(_resources(), DemoSettings(1, .01, 1.0, 10))
    assert calls == 100


def test_early_sampling_does_not_force_sources(monkeypatch) -> None:
    import diffusion_sources.temporal_demo_scenario as module

    monkeypatch.setattr(module, "simulate_ic", lambda graph, sources, probability, steps, rng: _cascade(sources, 6))
    monkeypatch.setattr(module, "sample_early_nodes", lambda infected, fraction, seed: frozenset())
    result = generate_demo_scenario(_resources(), DemoSettings(1, .01, 1.0, 20))
    assert result.early_nodes == frozenset()
    assert result.cascade.sources


def test_k3_five_candidates_are_rejected(monkeypatch) -> None:
    import diffusion_sources.temporal_demo_scenario as module

    monkeypatch.setattr(module, "simulate_ic", lambda graph, sources, probability, steps, rng: _cascade(sources, 7))
    resources = _resources()
    resources.generation_config["dataset"]["max_infected_fraction"] = 1.0
    with pytest.raises(RuntimeError, match="100"):
        generate_demo_scenario(resources, DemoSettings(3, .01, .75, 30))
