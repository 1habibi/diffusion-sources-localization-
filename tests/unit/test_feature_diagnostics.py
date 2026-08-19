from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from diffusion_sources.feature_diagnostics import diagnose_snapshot_features
from diffusion_sources.generation import generate_dataset


def _config(data_dir: Path) -> dict:
    return {
        "model_version": "snapshot_v2",
        "experiment": "diagnostic_test",
        "data": {
            "directory": str(data_dir),
            "feature_names": [
                "observed_infected",
                "mean_distance_to_observed_normalized",
                "observed_count_normalized",
                "candidate_count_normalized",
            ],
            "distance_cache": str(data_dir / "distance_cache.npz"),
            "distance_cap": 5,
        },
        "evaluation": {"evaluate_test": False},
    }


def _generate(data_dir: Path) -> None:
    generate_dataset(
        {
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
                "splits": {"train": 2, "validation": 2, "test": 2},
                "min_candidates": 2,
                "max_infected_fraction": 0.9,
                "max_attempt_factor": 100,
            },
        },
        data_dir,
    )


def test_diagnose_snapshot_features_uses_only_development_splits(tmp_path):
    data_dir = tmp_path / "data"
    _generate(data_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(data_dir)), encoding="utf-8")

    summary = diagnose_snapshot_features(
        config_path, tmp_path / "feature_diagnostics.json"
    )

    assert set(summary["splits"]) == {"train", "validation"}
    assert summary["scope"] == "candidate_nodes_train_validation_only"
    assert summary["splits"]["train"]["features"]["observed_infected"]["mean"] == 1.0
    assert [
        "observed_count_normalized", "candidate_count_normalized"
    ] in summary["splits"]["train"]["exact_duplicate_feature_pairs"]
    assert (tmp_path / "feature_diagnostics.json").exists()


def test_diagnose_snapshot_features_rejects_unlocked_test(tmp_path):
    data_dir = tmp_path / "data"
    _generate(data_dir)
    config = _config(data_dir)
    config["evaluation"]["evaluate_test"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluate_test=false"):
        diagnose_snapshot_features(config_path, tmp_path / "output.json")
