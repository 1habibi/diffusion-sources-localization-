from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import yaml

from diffusion_sources.generation import generate_dataset
from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.robustness import build_condition_data, evaluate_robustness


def test_build_condition_data_preserves_sources_and_adds_candidates():
    graph = nx.path_graph(6)
    infected = np.asarray([True, True, True, True, False, False])
    sources = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    data = build_condition_data(
        graph,
        infected,
        sources,
        fraction=0.5,
        false_positive_fraction=0.5,
        rng=np.random.default_rng(4),
        feature_indices=[0, 1],
    )

    source_nodes = set(np.flatnonzero(sources))
    candidates = set(np.flatnonzero(data.candidate_mask.numpy()))
    assert source_nodes.issubset(candidates)
    assert data.x.shape == (6, 2)
    assert len(candidates) >= len(source_nodes)


def test_evaluate_robustness_builds_full_condition_grid(tmp_path):
    generation_config = {
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
            "splits": {"train": 3, "validation": 3, "test": 3},
            "min_candidates": 3,
            "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "robustness"
    run_dir.mkdir()
    generate_dataset(generation_config, data_dir)
    model = JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    (run_dir / "metrics.json").write_text(
        '{"feature_indices": [0, 1]}', encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": {"hidden_dim": 8, "dropout": 0.0}}),
        encoding="utf-8",
    )

    summary = evaluate_robustness(
        data_dir,
        run_dir,
        output_dir,
        fractions=(1.0, 0.5),
        noise_levels=(0.0, 0.1),
        seed=5,
    )

    assert len(summary["conditions"]) == 4
    assert (output_dir / "robustness_table.csv").exists()
    assert (output_dir / "robustness_heatmaps.png").exists()
