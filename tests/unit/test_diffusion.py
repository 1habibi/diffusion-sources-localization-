from __future__ import annotations

import numpy as np
import pytest

from diffusion_sources.diffusion import simulate_ic, simulate_si


def test_probability_zero_keeps_only_sources(path_graph):
    cascade = simulate_ic(path_graph, {1, 3}, 0.0, 5, np.random.default_rng(7))

    assert cascade.infected == frozenset({1, 3})
    assert cascade.history == (frozenset({1, 3}),)
    assert cascade.infection_times == {1: 0, 3: 0}


def test_probability_one_infects_connected_graph(path_graph):
    cascade = simulate_ic(path_graph, {2}, 1.0, 4, np.random.default_rng(7))

    assert cascade.infected == frozenset(path_graph.nodes())
    assert cascade.infection_times[2] == 0
    assert cascade.infection_times[0] == 2
    assert cascade.infection_times[4] == 2


def test_simulation_is_reproducible_and_source_order_independent(cycle_graph):
    first = simulate_ic(cycle_graph, [1, 3], 0.4, 4, np.random.default_rng(17))
    second = simulate_ic(cycle_graph, [3, 1], 0.4, 4, np.random.default_rng(17))

    assert first == second


def test_simulation_rejects_invalid_arguments(path_graph):
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="At least one source"):
        simulate_ic(path_graph, [], 0.5, 2, rng)
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate_ic(path_graph, {0}, 1.1, 2, rng)
    with pytest.raises(ValueError, match="graph nodes"):
        simulate_ic(path_graph, {9}, 0.5, 2, rng)


def test_si_probability_one_infects_connected_graph(path_graph):
    cascade = simulate_si(path_graph, {0, 4}, 1.0, 3, np.random.default_rng(1))
    assert cascade.infected == frozenset(path_graph.nodes())


def test_si_is_reproducible(cycle_graph):
    first = simulate_si(cycle_graph, {1, 3}, 0.4, 4, np.random.default_rng(17))
    second = simulate_si(cycle_graph, {3, 1}, 0.4, 4, np.random.default_rng(17))
    assert first == second


def test_si_keeps_retrying_after_an_empty_step(path_graph):
    class RetryRng:
        values = iter([0.9, 0.1, 0.1, 0.1])

        def random(self):
            return next(self.values)

    cascade = simulate_si(path_graph, {0}, 0.5, 2, RetryRng())

    assert cascade.history[1] == frozenset()
    assert 1 in cascade.infected
