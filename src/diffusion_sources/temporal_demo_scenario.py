"""Deterministic, reference-compatible IC cascades for a local demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion import Cascade, SourceSampler, simulate_ic
from .observations import Observation, observe_cascade
from .temporal_demo_assets import DemoResources
from .temporal_replay import sample_early_nodes


MAX_ATTEMPTS = 100


@dataclass(frozen=True)
class DemoSettings:
    true_k: int
    probability: float
    observation_fraction: float
    seed: int


@dataclass(frozen=True)
class DemoScenario:
    cascade: Cascade
    final: Observation
    early_nodes: frozenset[int]
    simulation_seed: int
    observation_seed: int
    attempt: int


def _validate_settings(settings: DemoSettings) -> None:
    if type(settings.true_k) is not int or settings.true_k not in (1, 2, 3):
        raise ValueError("Number of simulated sources must be 1, 2 or 3")
    if settings.probability not in (.01, .02, .03):
        raise ValueError("Transmission probability must be 0.01, 0.02 or 0.03")
    if settings.observation_fraction not in (.5, .75, 1.0):
        raise ValueError("Observation fraction must be 0.50, 0.75 or 1.00")
    if type(settings.seed) is not int or settings.seed < 0:
        raise ValueError("Seed must be a nonnegative integer")


def _attempt_seed(user_seed: int, attempt: int, role: int) -> int:
    return int(np.random.SeedSequence([0xD3A0, user_seed, attempt, role])
               .generate_state(1, dtype=np.uint32)[0])


def generate_demo_scenario(resources: DemoResources, settings: DemoSettings) -> DemoScenario:
    """Return the first accepted cascade, or fail explicitly after 100 attempts."""
    _validate_settings(settings)
    config = resources.generation_config
    simulation = config["simulation"]
    dataset = config["dataset"]
    if (simulation["max_steps"] != 3 or config["observation"].get("false_positive_count", 0) != 0
            or config["observation"].get("hide_source_count", 0) != 0):
        raise ValueError("Unsupported frozen simulation protocol")
    ranges = simulation["distance_ranges"]
    if ranges != [{"min": 1, "max": 2}, {"min": 3, "max": 5}]:
        raise ValueError("Frozen source distance ranges mismatch")

    sampler = SourceSampler(resources.graph, cache_size=int(dataset.get("distance_cache_size", 128)))
    min_candidates = max(int(dataset["min_candidates"]), 2 * settings.true_k)
    max_infected = float(dataset["max_infected_fraction"]) * resources.graph.number_of_nodes()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        simulation_seed = _attempt_seed(settings.seed, attempt, 0)
        observation_seed = _attempt_seed(settings.seed, attempt, 1)
        simulation_rng = np.random.default_rng(simulation_seed)
        distance_range = ranges[(settings.seed + attempt) % 2]
        try:
            sources = sampler.sample(settings.true_k, simulation_rng,
                                     min_distance=distance_range["min"],
                                     max_distance=distance_range["max"])
        except RuntimeError:
            continue
        cascade = simulate_ic(resources.graph, sources, settings.probability, 3, simulation_rng)
        if len(cascade.infected) > max_infected:
            continue
        final = observe_cascade(resources.graph, cascade, settings.observation_fraction,
                                0, np.random.default_rng(observation_seed))
        if len(final.candidate_nodes) < min_candidates:
            continue
        early = sample_early_nodes(
            (node for node, time in cascade.infection_times.items() if time <= 1),
            settings.observation_fraction, observation_seed)
        return DemoScenario(cascade, final, early, simulation_seed, observation_seed, attempt)
    raise RuntimeError(f"No valid demo cascade after {MAX_ATTEMPTS} attempts; change seed or settings")
