from __future__ import annotations

from diffusion_sources.evaluate_cli import evaluate_baselines
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


def test_evaluate_baselines_saves_tables(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "evaluation"
    generate_dataset(tiny_config(), data_dir)

    summary = evaluate_baselines(
        data_dir, output_dir, uniform_repeats=3, seed=4
    )

    assert summary["example_count"] == 3
    assert set(summary["methods"]) == {"uniform", "degree", "multi_jordan"}
    assert (output_dir / "baseline_metrics.json").exists()
    assert (output_dir / "baseline_table.csv").exists()
