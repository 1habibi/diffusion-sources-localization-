from __future__ import annotations

import json

from diffusion_sources.ablations import build_ablation_report


def _write_metrics(path, experiment, prediction_key, f1, count_accuracy, distance):
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "seed": 7,
                "prediction_metrics": {
                    prediction_key: {
                        "all": {
                            "f1": f1,
                            "precision": f1,
                            "recall": f1,
                            "exact_set_accuracy": 0.0,
                            "count_accuracy": count_accuracy,
                            "count_mae": 1.0 - count_accuracy,
                            "symmetric_set_distance": distance,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_ablation_report_creates_three_artifacts(tmp_path):
    node = tmp_path / "node"
    no_consistency = tmp_path / "no_consistency"
    joint = tmp_path / "joint"
    _write_metrics(node, "node_only", "node_thresholded", 0.3, 0.4, 1.2)
    _write_metrics(
        no_consistency,
        "joint_without_consistency",
        "joint_estimated_k",
        0.4,
        0.5,
        1.0,
    )
    _write_metrics(joint, "joint_full", "joint_estimated_k", 0.5, 0.6, 0.8)

    outputs = build_ablation_report(node, no_consistency, joint, tmp_path / "report")

    assert len(outputs) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
