from __future__ import annotations

import json

from diffusion_sources.feature_ablation import build_feature_ablation


def _write_run(path, feature_indices, f1):
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "seed": 7,
                "feature_indices": feature_indices,
                "prediction_metrics": {
                    "joint_estimated_k": {
                        "all": {
                            "f1": f1,
                            "precision": f1,
                            "recall": f1,
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


def test_build_feature_ablation_creates_artifacts(tmp_path):
    infected = tmp_path / "infected"
    full = tmp_path / "full"
    _write_run(infected, [0], 0.3)
    _write_run(full, [0, 1], 0.4)

    outputs = build_feature_ablation(infected, full, tmp_path / "report")

    assert len(outputs) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
