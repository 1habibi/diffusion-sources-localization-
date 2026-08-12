from __future__ import annotations

import numpy as np
import pytest

from diffusion_sources.generation import generate_dataset, graph_from_config


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
            "show_progress": False,
        },
    }


def test_generate_dataset_saves_topology_once_and_balanced_targets(tmp_path):
    summary = generate_dataset(tiny_config(), tmp_path)

    assert summary.accepted == {"train": 6, "validation": 3, "test": 3}
    assert (tmp_path / "graph.npz").exists()
    assert (tmp_path / "generation_summary.json").exists()
    train = np.load(tmp_path / "train.npz")
    assert train["features"].shape == (6, 34, 2)
    assert train["source_counts"].tolist() == [1, 2, 3, 1, 2, 3]
    assert set(summary.rejections["train"]) == {
        "source_distance",
        "cascade_too_large",
        "empty_observation",
        "too_few_candidates",
    }
    assert summary.duration_seconds["train"] >= 0.0


def test_graph_from_config_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported graph kind"):
        graph_from_config({"kind": "unknown"})


def test_graph_from_config_loads_edge_list(tmp_path):
    edge_list = tmp_path / "graph.txt"
    edge_list.write_text("10 11\n11 12\n100 101\n", encoding="utf-8")

    graph_id, graph = graph_from_config(
        {"id": "external", "kind": "edge_list", "path": str(edge_list)}
    )

    assert graph_id == "external"
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_hidden_source_generation_uses_all_nodes_as_candidates(tmp_path):
    config = tiny_config()
    config["observation"]["hide_source_count"] = 1
    generate_dataset(config, tmp_path)
    train = np.load(tmp_path / "train.npz")

    assert train["candidate_masks"].all()
    assert train["hidden_source_masks"].sum(axis=1).tolist() == [1] * 6
    assert np.all(train["features"][:, :, 0].sum(axis=1) >= 1)
