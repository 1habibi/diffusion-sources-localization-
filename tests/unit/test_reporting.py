from __future__ import annotations

import csv
import json

from diffusion_sources.reporting import build_report


def test_build_report_creates_figures_and_table(tmp_path):
    run_dir = tmp_path / "run"
    baseline_dir = tmp_path / "baselines"
    output_dir = tmp_path / "report"
    run_dir.mkdir()
    baseline_dir.mkdir()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split", "epoch", "loss", "source_loss", "count_loss",
                "consistency_loss", "macro_f1", "pr_auc", "count_accuracy",
            ],
        )
        writer.writeheader()
        for split in ("train", "validation"):
            writer.writerow(
                {
                    "split": split, "epoch": 1, "loss": 1.0,
                    "source_loss": 0.5, "count_loss": 0.4,
                    "consistency_loss": 0.1, "macro_f1": 0.5,
                    "pr_auc": 0.6, "count_accuracy": 0.7,
                }
            )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {"train": {"loss": 1.0}, "test": {"loss": 1.2}},
                "prediction_metrics": {
                    "joint_estimated_k": {
                        "all": {"f1": 0.6, "symmetric_set_distance": 0.8}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "methods": {
                    "degree": {
                        "all": {"f1": 0.4, "symmetric_set_distance": 1.2}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = build_report(run_dir, output_dir, baseline_dir)

    assert len(outputs) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
