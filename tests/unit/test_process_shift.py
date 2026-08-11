from __future__ import annotations

import csv
import json

import torch
import yaml

from diffusion_sources.generation import generate_dataset
from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.process_shift import (
    aggregate_process_rows,
    evaluate_process_shift,
    load_ic_rows,
)


def test_load_ic_rows_keeps_only_estimated_joint(tmp_path):
    path = tmp_path / "predictions.csv"
    fieldnames = [
        "example", "k", "method", "precision", "recall", "f1",
        "exact_set_accuracy", "count_accuracy", "count_mae",
        "source_to_set_distance", "set_to_source_distance", "symmetric_set_distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method in ("joint_estimated_k", "joint_oracle_k"):
            writer.writerow(
                {
                    "example": 0, "k": 1, "method": method, "precision": 0.5,
                    "recall": 0.5, "f1": 0.5, "exact_set_accuracy": 0.0,
                    "count_accuracy": 1.0, "count_mae": 0.0,
                    "source_to_set_distance": 1.0, "set_to_source_distance": 1.0,
                    "symmetric_set_distance": 1.0,
                }
            )
    rows = load_ic_rows(path)
    assert len(rows) == 1
    assert rows[0]["process"] == "IC"


def test_aggregate_process_rows_compares_both_processes():
    base = {"f1": 0.5, "count_accuracy": 1.0, "count_mae": 0.0,
            "symmetric_set_distance": 1.0}
    result = aggregate_process_rows(
        [{"process": "IC", **base}, {"process": "SI", **base}]
    )
    assert result["IC"]["f1"] == 0.5
    assert result["SI"]["symmetric_set_distance"] == 1.0


def test_evaluate_process_shift_creates_comparison_artifacts(tmp_path):
    config = {
        "graph": {"id": "karate", "kind": "karate"},
        "simulation": {
            "source_counts": [1, 2, 3], "probabilities": [0.4], "max_steps": 2,
            "distance_ranges": [{"min": 1, "max": 5}],
        },
        "observation": {"fractions": [1.0], "false_positive_count": 0},
        "dataset": {
            "seed": 13, "splits": {"train": 3, "validation": 3, "test": 3},
            "min_candidates": 3, "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }
    data_dir, run_dir, output_dir = tmp_path / "data", tmp_path / "run", tmp_path / "out"
    generate_dataset(config, data_dir)
    run_dir.mkdir()
    model = JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.0)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    (run_dir / "metrics.json").write_text(
        json.dumps({"feature_indices": [0, 1]}), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": {"hidden_dim": 8, "dropout": 0.0}}),
        encoding="utf-8",
    )
    fieldnames = [
        "example", "k", "method", "precision", "recall", "f1",
        "exact_set_accuracy", "count_accuracy", "count_mae",
        "source_to_set_distance", "set_to_source_distance", "symmetric_set_distance",
    ]
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, k in enumerate((1, 2, 3)):
            writer.writerow(
                {
                    "example": index, "k": k, "method": "joint_estimated_k",
                    "precision": 0.5, "recall": 0.5, "f1": 0.5,
                    "exact_set_accuracy": 0.0, "count_accuracy": 0.5,
                    "count_mae": 0.5, "source_to_set_distance": 1.0,
                    "set_to_source_distance": 1.0, "symmetric_set_distance": 1.0,
                }
            )

    summary = evaluate_process_shift(data_dir, run_dir, output_dir, seed=5)

    assert set(summary["processes"]) == {"IC", "SI"}
    assert (output_dir / "process_shift_table.csv").exists()
    assert (output_dir / "process_shift_comparison.png").exists()
