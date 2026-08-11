"""Train and evaluate Joint Source-Count GCN from generated NPZ splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .dataset import load_graph_archive, load_pyg_split
from .inference import predict_joint, predict_oracle_k
from .metrics import set_metrics, source_set_distances
from .models import JointSourceCountGCN
from .training import evaluate_epoch, fit_joint_model, save_training_result


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a training YAML configuration."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    for section in ("data", "model", "training", "loss"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing training configuration section: {section}.")
    return config


def set_seed(seed: int) -> None:
    """Set seeds used by the current CPU training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_pos_weight(examples: list) -> torch.Tensor:
    """Calculate negative/positive ratio inside train candidate masks."""
    positives = 0.0
    candidates = 0
    for data in examples:
        mask = data.candidate_mask.bool()
        positives += float(data.source_labels[mask].sum().item())
        candidates += int(mask.sum().item())
    if positives <= 0:
        raise ValueError("Training split contains no positive source labels.")
    return torch.tensor((candidates - positives) / positives, dtype=torch.float32)


def run_training(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Train, select by validation, evaluate test once, and save artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config["data"]["directory"])
    graph_id, graph = load_graph_archive(data_dir / "graph.npz")
    feature_indices = [int(value) for value in config["data"].get("feature_indices", [0, 1])]
    splits = {
        split: load_pyg_split(
            data_dir / f"{split}.npz", graph, feature_indices=feature_indices
        )
        for split in ("train", "validation", "test")
    }

    seed = int(config["training"].get("seed", 0))
    set_seed(seed)
    device = torch.device(config["training"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    for examples in splits.values():
        for data in examples:
            data.to(device)

    model = JointSourceCountGCN(
        input_dim=int(config["model"].get("input_dim", 2)),
        hidden_dim=int(config["model"].get("hidden_dim", 64)),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"].get("learning_rate", 1e-3))
    )
    pos_weight = (
        calculate_pos_weight(splits["train"]).to(device)
        if config["loss"].get("use_pos_weight", True)
        else None
    )
    lambda_count = float(config["loss"].get("lambda_count", 1.0))
    lambda_consistency = float(config["loss"].get("lambda_consistency", 0.1))

    started_at = time.perf_counter()
    result = fit_joint_model(
        model,
        splits["train"],
        splits["validation"],
        optimizer,
        max_epochs=int(config["training"].get("max_epochs", 100)),
        patience=int(config["training"].get("patience", 10)),
        lambda_count=lambda_count,
        lambda_consistency=lambda_consistency,
        pos_weight=pos_weight,
    )
    training_seconds = time.perf_counter() - started_at
    save_training_result(result, output_path)

    learning_rate = float(optimizer.param_groups[0]["lr"])
    metrics = {
        split: asdict(
            evaluate_epoch(
                model,
                examples,
                learning_rate=learning_rate,
                lambda_count=lambda_count,
                lambda_consistency=lambda_consistency,
                pos_weight=pos_weight,
            )
        )
        for split, examples in splits.items()
    }
    prediction_rows, prediction_metrics = evaluate_test_predictions(
        model, splits["test"], graph
    )
    _write_prediction_rows(output_path / "test_predictions.csv", prediction_rows)
    summary = {
        "graph_id": graph_id,
        "feature_indices": feature_indices,
        "model": "joint_source_count",
        "experiment": str(config.get("experiment", "joint_full")),
        "seed": seed,
        "device": str(device),
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "stop_reason": result.stop_reason,
        "training_seconds": training_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
        "lambda_count": lambda_count,
        "lambda_consistency": lambda_consistency,
        "metrics": metrics,
        "prediction_metrics": prediction_metrics,
        "split_sizes": {split: len(examples) for split, examples in splits.items()},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
    }
    (output_path / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return summary


@torch.no_grad()
def evaluate_test_predictions(model, examples: list, graph) -> tuple[list[dict], dict]:
    """Evaluate selected checkpoint in estimated-k and oracle-k modes."""
    model.eval()
    rows: list[dict] = []
    for index, data in enumerate(examples):
        source_logits, count_logits = model(data)
        true_sources = frozenset(
            torch.nonzero(data.source_labels, as_tuple=False).flatten().cpu().tolist()
        )
        true_count = int(data.source_count.item())
        predictions = {
            "joint_estimated_k": predict_joint(
                source_logits, count_logits, data.candidate_mask
            ),
            "joint_oracle_k": predict_oracle_k(
                source_logits, data.candidate_mask, true_count
            ),
        }
        for method, prediction in predictions.items():
            rows.append(
                {
                    "example": index,
                    "k": true_count,
                    "predicted_k": prediction.source_count,
                    "method": method,
                    **set_metrics(true_sources, prediction.sources),
                    **source_set_distances(graph, true_sources, prediction.sources),
                }
            )
    return rows, _aggregate_prediction_rows(rows)


def _aggregate_prediction_rows(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        result[method] = {}
        for group in ("all", 1, 2, 3):
            grouped = (
                method_rows if group == "all" else [row for row in method_rows if row["k"] == group]
            )
            if not grouped:
                continue
            result[method][str(group)] = {
                metric: float(np.mean([row[metric] for row in grouped]))
                for metric in (
                    "precision",
                    "recall",
                    "f1",
                    "exact_set_accuracy",
                    "count_accuracy",
                    "count_mae",
                    "symmetric_set_distance",
                )
            }
    return result


def _write_prediction_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_training(load_train_config(args.config), args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
