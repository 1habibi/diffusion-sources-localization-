from __future__ import annotations

import json

import torch
import yaml

from diffusion_sources.generation import generate_dataset
from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.transfer import evaluate_transfer


def test_evaluate_transfer_creates_external_report(tmp_path):
    data_config = {
        "graph": {"id": "external", "kind": "karate"},
        "simulation": {
            "source_counts": [1, 2, 3],
            "probabilities": [0.4],
            "max_steps": 2,
            "distance_ranges": [{"min": 1, "max": 5}],
        },
        "observation": {"fractions": [1.0], "false_positive_count": 0},
        "dataset": {
            "seed": 13,
            "splits": {"test": 3},
            "min_candidates": 3,
            "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "transfer"
    generate_dataset(data_config, data_dir)
    run_dir.mkdir()
    model = JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": {"hidden_dim": 8, "dropout": 0.0}}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "graph_id": "train_graph",
                "feature_indices": [0, 1],
                "prediction_metrics": {
                    "joint_estimated_k": {
                        "all": {
                            "f1": 0.5,
                            "count_accuracy": 0.5,
                            "count_mae": 0.5,
                            "symmetric_set_distance": 1.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_transfer(data_dir, run_dir, output_dir)

    assert summary["external_example_count"] == 3
    assert summary["external_graph"] == "external"
    assert (output_dir / "transfer_table.csv").exists()
    assert (output_dir / "transfer_comparison.png").exists()
