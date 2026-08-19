from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from diffusion_sources.generation import generate_dataset
from diffusion_sources.models import JointSourceCountGCN, NodeOnlyGCN
from diffusion_sources.v1_hop_diagnostics import evaluate_v1_hops


def _write_run(path: Path, model, *, threshold: float | None = None) -> None:
    path.mkdir(parents=True)
    torch.save(model.state_dict(), path / "best_model.pt")
    (path / "config.yaml").write_text(
        yaml.safe_dump({"model": {"hidden_dim": 8, "dropout": 0.0}}),
        encoding="utf-8",
    )
    metrics = {"feature_indices": [0, 1]}
    if threshold is not None:
        metrics["threshold"] = threshold
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_evaluate_v1_hops_preserves_runs_and_writes_new_diagnostics(tmp_path):
    config = {
        "graph": {"id": "karate", "kind": "karate"},
        "simulation": {
            "source_counts": [1],
            "probabilities": [0.4],
            "max_steps": 2,
            "distance_ranges": [{"min": 1, "max": 5}],
        },
        "observation": {"fractions": [1.0], "false_positive_count": 0},
        "dataset": {
            "seed": 13,
            "splits": {"test": 2},
            "min_candidates": 2,
            "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }
    data_dir = tmp_path / "data"
    generate_dataset(config, data_dir)
    joint_root, node_root = tmp_path / "joint", tmp_path / "node"
    _write_run(
        joint_root / "seed_1", JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.0)
    )
    _write_run(
        node_root / "seed_1",
        NodeOnlyGCN(input_dim=2, hidden_dim=8, dropout=0.0),
        threshold=0.5,
    )

    summary = evaluate_v1_hops(
        data_dir, joint_root, node_root, tmp_path / "output", seeds=(1,), batch_size=2
    )

    assert summary["test_examples_per_seed"] == 2
    assert set(summary["metrics"]) == {
        "joint_estimated_k",
        "joint_oracle_k",
        "node_thresholded",
        "node_oracle_k",
    }
    assert (tmp_path / "output" / "v1_hop_metrics.json").exists()
    assert not (joint_root / "seed_1" / "v1_hop_metrics.json").exists()
