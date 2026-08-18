from __future__ import annotations

import json

import pytest
import yaml

from diffusion_sources.experiments import AGGREGATED_METRICS
from diffusion_sources.saved_results import REQUIRED_ARTIFACTS, aggregate_saved_results


def test_aggregate_saved_results_validates_and_summarizes_runs(tmp_path):
    runs_root = tmp_path / "runs"
    for experiment, prediction, oracle_prediction in (
        ("joint", "joint_estimated_k", "joint_oracle_k"),
        ("node", "node_thresholded", "node_oracle_k"),
    ):
        for seed in (1, 2, 3):
            run_dir = runs_root / experiment / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            for artifact in REQUIRED_ARTIFACTS:
                (run_dir / artifact).write_bytes(b"test")
            values = {metric: seed / 10 for metric in AGGREGATED_METRICS}
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "prediction_metrics": {
                            prediction: {"all": values, "1": values},
                            oracle_prediction: {"all": values, "1": values},
                        },
                    }
                ),
                encoding="utf-8",
            )

    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(json.dumps({"methods": {}}), encoding="utf-8")
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "seeds": [1, 2, 3],
                "runs_root": "runs",
                "baselines": "baselines.json",
                "experiments": {
                    "joint": {
                        "directory": "joint",
                        "estimated_prediction": "joint_estimated_k",
                        "oracle_prediction": "joint_oracle_k",
                    },
                    "node": {
                        "directory": "node",
                        "estimated_prediction": "node_thresholded",
                        "oracle_prediction": "node_oracle_k",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_saved_results(config_path, tmp_path / "output")

    assert result["estimated_aggregates"]["all"]["joint"]["f1"]["mean"] == pytest.approx(0.2)
    assert all(record["complete"] for record in result["artifact_checks"])
    assert (tmp_path / "output" / "estimated_k_comparison.png").exists()
    assert (tmp_path / "output" / "oracle_k_methods.png").exists()
