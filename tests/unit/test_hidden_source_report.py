from __future__ import annotations

import json

from diffusion_sources.hidden_source_report import build_hidden_source_report


def _write_run(path, f1, distance):
    path.mkdir()
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "seed": 7,
                "prediction_metrics": {
                    "joint_estimated_k": {
                        "all": {
                            "f1": f1,
                            "precision": f1,
                            "recall": f1,
                            "exact_set_accuracy": 0.0,
                            "count_accuracy": 0.5,
                            "count_mae": 0.5,
                            "symmetric_set_distance": distance,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_hidden_source_report_creates_separate_mode_artifacts(tmp_path):
    primary, hidden = tmp_path / "primary", tmp_path / "hidden"
    _write_run(primary, 0.5, 1.0)
    _write_run(hidden, 0.2, 2.0)

    outputs = build_hidden_source_report(primary, hidden, tmp_path / "report")

    assert len(outputs) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
