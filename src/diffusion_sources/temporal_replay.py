"""Checked reconstruction of IC histories; only source-blind masks leave replay."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

import networkx as nx
import numpy as np

from .diffusion import SourceSampler, simulate_ic
from .observations import observe_cascade

POLICY_VERSION = 'independent-bernoulli-t1-seedsequence-v1'
FIELDS = ('features', 'candidate_masks', 'source_labels', 'infected_masks',
          'source_counts', 'simulation_seeds', 'observation_seeds',
          'probabilities', 'observation_fractions')


def sample_early_nodes(infected_by_t1: Iterable[int], fraction: float,
                       observation_seed: int, *, t1: int = 1) -> frozenset[int]:
    """Sample without seeing source identity; an empty observation is valid."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1 or t1 != 1 or observation_seed < 0:
        raise ValueError('Invalid early sampling policy (fraction, seed or t1).')
    nodes = np.asarray(sorted(set(infected_by_t1)), dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([5000000, observation_seed, t1]))
    return frozenset(nodes[rng.random(len(nodes)) < fraction].tolist())


def replay_early_mask(graph: nx.Graph, config: dict,
                      archive: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    """Replay one accepted example, stopping on any mismatch instead of filtering."""
    def fail(field: str) -> None:
        raise ValueError(f'example {index}: inconsistent {field}')

    if sorted(graph.nodes()) != list(range(graph.number_of_nodes())):
        fail('graph node IDs')
    for field in FIELDS:
        if field not in archive:
            fail(f'missing {field}')
    size = len(archive['source_counts'])
    if index < 0 or index >= size or any(len(archive[f]) != size for f in FIELDS):
        fail('archive lengths/index')
    simulation, observation = config['simulation'], config['observation']
    if (simulation['max_steps'] != 3 or observation.get('hide_source_count', 0) != 0
            or observation.get('false_positive_count', 0) != 0):
        fail('unsupported observation/simulation protocol')
    conditions = list(itertools.product(
        simulation.get('distance_ranges', [{'min': 0, 'max': None}]),
        simulation['probabilities'], observation['fractions'], simulation.get('source_counts', [1, 2, 3])))
    if not conditions:
        fail('empty conditions')
    dr, probability, fraction, k = conditions[index % len(conditions)]
    for field, value in [('source_counts', k), ('probabilities', probability),
                          ('observation_fractions', fraction)]:
        if not np.isclose(archive[field][index], value, rtol=0, atol=1e-8):
            fail(field)
    n = graph.number_of_nodes()
    for field in ('source_labels', 'infected_masks', 'candidate_masks'):
        row = archive[field][index]
        if row.shape != (n,) or not np.isin(row, [0, 1]).all():
            fail(field)
    features = archive['features'][index]
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] < 1:
        fail('features')
    simulation_seed, observation_seed = (int(archive[f][index]) for f in ('simulation_seeds', 'observation_seeds'))
    if (simulation_seed < 0 or observation_seed < 0
            or simulation_seed != archive['simulation_seeds'][index]
            or observation_seed != archive['observation_seeds'][index]):
        fail('seeds')
    rng = np.random.default_rng(simulation_seed)
    sources = SourceSampler(graph, cache_size=int(config.get('dataset', {}).get('distance_cache_size', 128))).sample(
        int(k), rng, min_distance=int(dr.get('min', 0)), max_distance=dr.get('max'))
    if sources != frozenset(np.flatnonzero(archive['source_labels'][index]).tolist()):
        fail('source_labels')
    cascade = simulate_ic(graph, sources, float(probability), 3, rng)
    if cascade.infected != frozenset(np.flatnonzero(archive['infected_masks'][index]).tolist()):
        fail('infected_masks')
    final = observe_cascade(graph, cascade, float(fraction), 0, np.random.default_rng(observation_seed))
    if final.candidate_nodes != frozenset(np.flatnonzero(archive['candidate_masks'][index]).tolist()):
        fail('candidate_masks')
    expected_observed = np.asarray([node in final.observed_infected for node in range(n)])
    if not np.array_equal(features[:, 0], expected_observed):
        fail('features observed mask')
    early = sample_early_nodes(
        (node for node, time in cascade.infection_times.items() if time <= 1),
        float(fraction), observation_seed)
    result = np.zeros(n, dtype=bool)
    result[list(early)] = True
    return result
