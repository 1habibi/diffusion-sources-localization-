from __future__ import annotations

import networkx as nx
import pytest


@pytest.fixture
def temporal_dataset(tmp_path):
    from diffusion_sources.generation import generate_dataset
    config = {
        'graph': {'id': 'karate', 'kind': 'karate'},
        'simulation': {'source_counts': [1, 2, 3], 'probabilities': [.4],
                       'max_steps': 3, 'distance_ranges': [{'min': 1, 'max': 5}]},
        'observation': {'fractions': [1.], 'false_positive_count': 0},
        'dataset': {'seed': 13, 'splits': {'train': 6, 'validation': 3},
                    'min_candidates': 3, 'max_infected_fraction': .99,
                    'max_attempt_factor': 100},
    }
    data = tmp_path / 'data'
    generate_dataset(config, data)
    return data, config


@pytest.fixture
def path_graph() -> nx.Graph:
    return nx.path_graph(5)


@pytest.fixture
def cycle_graph() -> nx.Graph:
    return nx.cycle_graph(5)


@pytest.fixture
def disconnected_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([(10, 11), (11, 12), (100, 101)])
    graph.add_edge(12, 12)
    return graph
