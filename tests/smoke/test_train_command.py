from __future__ import annotations

import subprocess
import sys

import yaml

from diffusion_sources.generation import generate_dataset


def test_train_model_cli_saves_report_artifacts(tmp_path):
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
            "seed": 19,
            "splits": {"train": 6, "validation": 3, "test": 3},
            "min_candidates": 3,
            "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "run"
    generate_dataset(generation_config, data_dir)
    training_config = {
        "data": {"directory": str(data_dir)},
        "model": {"input_dim": 2, "hidden_dim": 8, "dropout": 0.0},
        "training": {
            "seed": 7,
            "device": "cpu",
            "learning_rate": 0.01,
            "max_epochs": 2,
            "patience": 2,
        },
        "loss": {
            "lambda_count": 1.0,
            "lambda_consistency": 0.1,
            "use_pos_weight": True,
        },
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_model.py",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "history.json").exists()
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "test_predictions.csv").exists()
