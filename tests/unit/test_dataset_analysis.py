from __future__ import annotations

from diffusion_sources.dataset_analysis import analyze_dataset
from diffusion_sources.generation import generate_dataset


def test_analyze_dataset_saves_statistics(tmp_path):
    config = {
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
            "splits": {"train": 6},
            "min_candidates": 3,
            "max_infected_fraction": 0.9,
            "max_attempt_factor": 100,
        },
    }
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "report"
    generate_dataset(config, data_dir)

    summary = analyze_dataset(data_dir, report_dir)

    assert summary["graph"]["nodes"] == 34
    assert summary["total_examples"] == 6
    assert summary["splits"]["train"]["k_counts"] == {"1": 2, "2": 2, "3": 2}
    assert summary["splits"]["train"]["probability_counts"] == {"0.4": 6}
    assert (report_dir / "dataset_analysis.json").exists()
    assert (report_dir / "cascade_distributions.png").exists()
