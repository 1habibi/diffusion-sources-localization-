from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from diffusion_sources.v1_diagnostics import diagnose_v1


def _write_prediction_file(path: Path, method: str) -> None:
    fields = [
        "example", "k", "predicted_k", "method", "precision", "recall",
        "f1", "exact_set_accuracy", "count_accuracy", "count_mae",
        "source_to_set_distance", "set_to_source_distance", "symmetric_set_distance",
    ]
    rows = []
    for example, true_k in enumerate((1, 2, 3)):
        rows.append(
            {
                "example": example,
                "k": true_k,
                "predicted_k": true_k,
                "method": method,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "exact_set_accuracy": 1.0,
                "count_accuracy": 1.0,
                "count_mae": 0.0,
                "source_to_set_distance": 0.0,
                "set_to_source_distance": 0.0,
                "symmetric_set_distance": 0.0,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_diagnose_v1_builds_all_outputs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    candidate_masks = np.zeros((3, 25), dtype=bool)
    candidate_masks[0, :5] = True
    candidate_masks[1, :12] = True
    candidate_masks[2, :25] = True
    np.savez(data_dir / "test.npz", candidate_masks=candidate_masks)
    joint_root = tmp_path / "joint"
    node_root = tmp_path / "node"
    for root, method in (
        (joint_root, "joint_estimated_k"),
        (node_root, "node_thresholded"),
    ):
        for seed in (7026, 7027, 7028):
            run_dir = root / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            _write_prediction_file(run_dir / "test_predictions.csv", method)

    summary = diagnose_v1(
        data_dir,
        joint_root,
        node_root,
        tmp_path / "output",
        bootstrap_repeats=100,
    )

    assert summary["test_examples"] == 3
    assert summary["confusion"]["joint_no_consistency"]["matrix"][0][0] == 3
    assert (tmp_path / "output" / "v1_diagnostics.json").exists()
    assert (tmp_path / "output" / "count_confusion.png").exists()
    assert (tmp_path / "output" / "candidate_size_analysis.csv").exists()
