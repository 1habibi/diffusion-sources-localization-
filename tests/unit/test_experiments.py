from __future__ import annotations

import json

import pytest
import yaml

from diffusion_sources.experiments import (
    AGGREGATED_METRICS,
    aggregate_seed_metrics,
    run_experiment_series,
)


def test_aggregate_seed_metrics_calculates_mean_and_interval():
    records = []
    for seed, value in ((1, 0.2), (2, 0.4), (3, 0.6)):
        record = {"experiment": "joint", "seed": seed}
        record.update({metric: value for metric in AGGREGATED_METRICS})
        records.append(record)

    aggregates = aggregate_seed_metrics(records)

    assert aggregates["joint"]["f1"]["mean"] == pytest.approx(0.4)
    assert aggregates["joint"]["f1"]["std"] > 0
    assert 0.0 <= aggregates["joint"]["f1"]["ci95_low"] < 0.4
    assert aggregates["joint"]["f1"]["ci95_high"] > 0.4


def test_run_experiment_series_aggregates_mocked_runs(tmp_path, monkeypatch):
    train_config = {
        "data": {},
        "model": {},
        "training": {"seed": 0},
        "loss": {},
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train_config), encoding="utf-8")
    series_path = tmp_path / "series.yaml"
    series_path.write_text(
        yaml.safe_dump(
            {
                "seeds": [1, 2, 3],
                "experiments": {
                    "joint": {
                        "kind": "joint",
                        "config": str(train_path),
                        "prediction": "joint_estimated_k",
                    },
                    "node": {
                        "kind": "node",
                        "config": str(train_path),
                        "prediction": "node_thresholded",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_result(config, output_dir):
        seed = config["training"]["seed"]
        prediction = (
            "joint_estimated_k"
            if config["experiment"] == "joint"
            else "node_thresholded"
        )
        values = {
            metric: seed / 10 for metric in AGGREGATED_METRICS
        }
        result = {
            "seed": seed,
            "prediction_metrics": {prediction: {"all": values}},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    monkeypatch.setattr("diffusion_sources.experiments.run_training", fake_result)
    monkeypatch.setattr("diffusion_sources.experiments.run_node_training", fake_result)

    result = run_experiment_series(series_path, tmp_path / "output")

    assert len(result["runs"]) == 6
    assert result["aggregates"]["joint"]["f1"]["mean"] == pytest.approx(0.2)
    assert (tmp_path / "output" / "series_summary.csv").exists()
    assert (tmp_path / "output" / "series_comparison.png").exists()
