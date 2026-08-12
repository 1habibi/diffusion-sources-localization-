from __future__ import annotations

import networkx as nx
import pytest

from diffusion_sources.dataset import load_graph_archive, load_pyg_split
from diffusion_sources.generation import generate_dataset


def tiny_config() -> dict:
    return {
        "graph": {"id": "karate", "kind": "karate"},
        "simulation": {
            "source_counts": [1, 2, 3],
            "probabilities": [0.4],
            "max_steps": 2,
            "distance_ranges": [{"min": 1, "max": 5}],
        },
        "observation": {"fractions": [1.0], "false_positive_count": 0},
        "dataset": {
            "seed": 13,
            "splits": {"train": 6, "validation": 3, "test": 3},
            "min_candidates": 3,
            "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }


def test_load_generated_graph_and_split(tmp_path):
    generate_dataset(tiny_config(), tmp_path)
    graph_id, graph = load_graph_archive(tmp_path / "graph.npz")
    examples = load_pyg_split(tmp_path / "train.npz", graph)

    assert graph_id == "karate"
    assert graph.number_of_nodes() == 34
    assert len(examples) == 6
    assert examples[0].x.shape == (34, 2)
    assert [int(item.source_count) for item in examples] == [1, 2, 3, 1, 2, 3]

    infected_only = load_pyg_split(
        tmp_path / "train.npz", graph, feature_indices=[0]
    )
    assert infected_only[0].x.shape == (34, 1)

    limited = load_pyg_split(tmp_path / "train.npz", graph, limit=2)
    assert len(limited) == 2


def test_load_split_rejects_graph_with_wrong_node_count(tmp_path):
    generate_dataset(tiny_config(), tmp_path)
    wrong_graph = nx.path_graph(5)

    with pytest.raises(ValueError, match="node count"):
        load_pyg_split(tmp_path / "train.npz", wrong_graph)
