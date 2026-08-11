"""Train Node-only GCN and tune its estimated-k threshold on validation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import load_graph_archive, load_pyg_split
from .inference import predict_oracle_k, predict_thresholded
from .metrics import set_metrics, source_set_distances
from .models import NodeOnlyGCN
from .train_cli import calculate_pos_weight, load_train_config, set_seed
from .training import evaluate_node_epoch, fit_node_model, save_training_result


@torch.no_grad()
def select_threshold(model, examples: list, thresholds: list[float]) -> tuple[float, float]:
    """Select the validation threshold with maximum per-cascade macro-F1."""
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    model.eval()
    best_threshold, best_f1 = thresholds[0], -1.0
    for threshold in thresholds:
        scores = []
        for data in examples:
            prediction = predict_thresholded(model(data), data.candidate_mask, threshold)
            true_sources = torch.nonzero(
                data.source_labels, as_tuple=False
            ).flatten().cpu().tolist()
            scores.append(set_metrics(true_sources, prediction.sources)["f1"])
        mean_f1 = float(np.mean(scores))
        if mean_f1 > best_f1:
            best_threshold, best_f1 = threshold, mean_f1
    return float(best_threshold), best_f1


@torch.no_grad()
def evaluate_node_predictions(model, examples: list, graph, threshold: float):
    """Evaluate Node-only GCN in estimated-k and oracle-k modes."""
    model.eval()
    rows = []
    for index, data in enumerate(examples):
        logits = model(data)
        true_sources = frozenset(
            torch.nonzero(data.source_labels, as_tuple=False).flatten().cpu().tolist()
        )
        true_count = int(data.source_count.item())
        predictions = {
            "node_thresholded": predict_thresholded(
                logits, data.candidate_mask, threshold
            ),
            "node_oracle_k": predict_oracle_k(
                logits, data.candidate_mask, true_count
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
    return rows, _aggregate(rows)


def _aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        result[method] = {}
        for group in ("all", 1, 2, 3):
            selected = method_rows if group == "all" else [r for r in method_rows if r["k"] == group]
            if selected:
                result[method][str(group)] = {
                    metric: float(np.mean([row[metric] for row in selected]))
                    for metric in (
                        "precision", "recall", "f1", "exact_set_accuracy",
                        "count_accuracy", "count_mae", "symmetric_set_distance",
                    )
                }
    return result


def run_node_training(config: dict, output_dir: str | Path) -> dict:
    """Train Node-only GCN, tune threshold, and evaluate test exactly once."""
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
    for examples in splits.values():
        for data in examples:
            data.to(device)
    model = NodeOnlyGCN(
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
    started = time.perf_counter()
    result = fit_node_model(
        model,
        splits["train"],
        splits["validation"],
        optimizer,
        max_epochs=int(config["training"].get("max_epochs", 100)),
        patience=int(config["training"].get("patience", 10)),
        pos_weight=pos_weight,
    )
    save_training_result(result, output_path)
    thresholds = [float(value) for value in config["inference"]["thresholds"]]
    threshold, validation_threshold_f1 = select_threshold(
        model, splits["validation"], thresholds
    )
    rows, prediction_metrics = evaluate_node_predictions(
        model, splits["test"], graph, threshold
    )
    with (output_path / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    metrics = {
        split: asdict(
            evaluate_node_epoch(
                model, examples, learning_rate=learning_rate, pos_weight=pos_weight
            )
        )
        for split, examples in splits.items()
    }
    summary = {
        "graph_id": graph_id,
        "feature_indices": feature_indices,
        "model": "node_only",
        "experiment": str(config.get("experiment", "node_only")),
        "seed": seed,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "stop_reason": result.stop_reason,
        "training_seconds": time.perf_counter() - started,
        "threshold": threshold,
        "validation_threshold_f1": validation_threshold_f1,
        "metrics": metrics,
        "prediction_metrics": prediction_metrics,
    }
    (output_path / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_node_training(load_train_config(args.config), args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
