"""Train and evaluate Joint Source-Count GCN from generated NPZ splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .dataset import load_graph_archive, load_pyg_split
from .features import GLOBAL_SCALAR_FEATURE_NAMES, SnapshotFeatureBuilder
from .inference import predict_joint, predict_oracle_k
from .metrics import set_metrics, source_radius_hits, source_set_distances
from .models import JointSourceCountGCN
from .runtime import peak_memory_bytes, reset_peak_memory, runtime_metadata
from .shortlist import evaluate_shortlist_grid
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
    """Set seeds used by CPU and CUDA training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    print(f"[stage] Loading graph: {data_dir / 'graph.npz'}", flush=True)
    graph_id, graph = load_graph_archive(data_dir / "graph.npz")
    configured_feature_names = config["data"].get("feature_names")
    feature_names = (
        [str(value) for value in configured_feature_names]
        if configured_feature_names is not None
        else None
    )
    feature_indices = (
        None
        if feature_names is not None
        else [int(value) for value in config["data"].get("feature_indices", [0, 1])]
    )
    split_limits = config["data"].get("split_limits", {})
    feature_builder = (
        SnapshotFeatureBuilder(
            graph,
            distance_cache_path=config["data"].get("distance_cache"),
            distance_cap=int(config["data"].get("distance_cap", 10)),
        )
        if feature_names is not None
        else None
    )
    evaluate_test = bool(config.get("evaluation", {}).get("evaluate_test", True))
    split_names = ("train", "validation", "test") if evaluate_test else ("train", "validation")
    splits = {}
    for split in split_names:
        print(f"[stage] Loading {split} split...", flush=True)
        splits[split] = load_pyg_split(
            data_dir / f"{split}.npz",
            graph,
            feature_indices=feature_indices,
            feature_names=feature_names,
            feature_builder=feature_builder,
            limit=(int(split_limits[split]) if split in split_limits else None),
        )
        print(
            f"[stage] Loaded {split}: {len(splits[split]):,} examples",
            flush=True,
        )

    seed = int(config["training"].get("seed", 0))
    set_seed(seed)
    device = torch.device(config["training"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    input_dim = len(feature_names) if feature_names is not None else len(feature_indices)
    configured_input_dim = int(config["model"].get("input_dim", input_dim))
    if configured_input_dim != input_dim:
        raise ValueError("model.input_dim must match the selected data features.")
    source_head_mode = str(config["model"].get("source_head_mode", "local"))
    backbone_mode = str(config["model"].get("backbone_mode", "plain_2"))
    source_head_strategy = str(
        config["model"].get("source_head_strategy", "shared")
    )
    shortlist_mode = str(config["model"].get("shortlist_mode", "disabled"))
    selected_global_features = [
        name for name in (feature_names or []) if name in GLOBAL_SCALAR_FEATURE_NAMES
    ]
    configured_global_feature_dim = int(
        config["model"].get("global_feature_dim", 0)
    )
    expected_global_feature_dim = (
        len(selected_global_features) if source_head_mode == "global_context" else 0
    )
    if configured_global_feature_dim != expected_global_feature_dim:
        raise ValueError(
            "model.global_feature_dim must match selected global scalar features."
        )
    model = JointSourceCountGCN(
        input_dim=input_dim,
        hidden_dim=int(config["model"].get("hidden_dim", 64)),
        dropout=float(config["model"].get("dropout", 0.2)),
        source_head_mode=source_head_mode,
        global_feature_dim=configured_global_feature_dim,
        backbone_mode=backbone_mode,
        source_head_strategy=source_head_strategy,
        shortlist_mode=shortlist_mode,
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
    lambda_rank = float(config["loss"].get("lambda_rank", 0.0))
    rank_negatives_per_positive = int(
        config["loss"].get("rank_negatives_per_positive", 8)
    )
    rank_hard_negative_fraction = float(
        config["loss"].get("rank_hard_negative_fraction", 0.5)
    )
    lambda_preliminary = float(
        config["loss"].get("lambda_preliminary", 0.0)
    )
    shortlist_config = config.get("shortlist", {})
    shortlist_enabled = bool(shortlist_config.get("enabled", False))
    if shortlist_enabled and shortlist_mode != "preliminary":
        raise ValueError(
            "shortlist.enabled requires model.shortlist_mode=preliminary."
        )
    batch_size = int(config["training"].get("batch_size", 1))
    checkpoint_path = output_path / "last_checkpoint.pt"
    resume_from = (
        checkpoint_path
        if bool(config["training"].get("resume", False)) and checkpoint_path.exists()
        else None
    )

    resume_note = f"; resume={resume_from}" if resume_from is not None else ""
    print(
        f"[stage] Starting training on {device}: max_epochs={config['training'].get('max_epochs', 100)}, "
        f"patience={config['training'].get('patience', 10)}{resume_note}",
        flush=True,
    )
    started_at = time.perf_counter()
    reset_peak_memory(device)
    result = fit_joint_model(
        model,
        splits["train"],
        splits["validation"],
        optimizer,
        max_epochs=int(config["training"].get("max_epochs", 100)),
        patience=int(config["training"].get("patience", 10)),
        lambda_count=lambda_count,
        lambda_consistency=lambda_consistency,
        lambda_rank=lambda_rank,
        rank_negatives_per_positive=rank_negatives_per_positive,
        rank_hard_negative_fraction=rank_hard_negative_fraction,
        lambda_preliminary=lambda_preliminary,
        pos_weight=pos_weight,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )
    training_seconds = time.perf_counter() - started_at
    save_training_result(result, output_path)
    print(
        f"[stage] Training complete: best_epoch={result.best_epoch}, "
        f"stopped_epoch={result.stopped_epoch}, elapsed={training_seconds / 60:.1f} min",
        flush=True,
    )

    learning_rate = float(optimizer.param_groups[0]["lr"])
    print("[stage] Evaluating aggregate train/validation metrics...", flush=True)
    metrics = {
        split: asdict(
            evaluate_epoch(
                model,
                examples,
                learning_rate=learning_rate,
                lambda_count=lambda_count,
                lambda_consistency=lambda_consistency,
                lambda_rank=lambda_rank,
                rank_negatives_per_positive=rank_negatives_per_positive,
                rank_hard_negative_fraction=rank_hard_negative_fraction,
                lambda_preliminary=lambda_preliminary,
                pos_weight=pos_weight,
                batch_size=batch_size,
            )
        )
        for split, examples in splits.items()
    }
    print("[stage] Evaluating detailed validation predictions...", flush=True)
    validation_rows, validation_prediction_metrics = evaluate_test_predictions(
        model, splits["validation"], graph
    )
    _write_prediction_rows(
        output_path / "validation_predictions.csv", validation_rows
    )
    prediction_metrics = None
    if evaluate_test:
        prediction_rows, prediction_metrics = evaluate_test_predictions(
            model, splits["test"], graph
        )
        _write_prediction_rows(output_path / "test_predictions.csv", prediction_rows)
    shortlist_validation = None
    if shortlist_enabled:
        print("[stage] Evaluating validation shortlist grid...", flush=True)
        shortlist_validation = evaluate_shortlist_grid(
            model,
            splits["validation"],
            shortlist_config.get("sizes", [8, 12, 16, 24, 32]),
            bootstrap_repeats=int(
                shortlist_config.get("bootstrap_repeats", 2000)
            ),
            bootstrap_seed=int(shortlist_config.get("bootstrap_seed", 17026)),
            micro_recall_min=float(
                shortlist_config.get("micro_candidate_recall_min", 0.95)
            ),
            per_k_recall_min=float(
                shortlist_config.get("per_k_candidate_recall_min", 0.95)
            ),
            bootstrap_ci_low_min=float(
                shortlist_config.get("bootstrap_ci_low_min", 0.93)
            ),
            require_f1_or_latency_improvement=bool(
                shortlist_config.get(
                    "require_f1_or_latency_improvement", True
                )
            ),
        )
        (output_path / "shortlist_validation.json").write_text(
            json.dumps(shortlist_validation, indent=2), encoding="utf-8"
        )
    summary = {
        "graph_id": graph_id,
        "feature_indices": feature_indices,
        "feature_names": feature_names,
        "model": "joint_source_count",
        "source_head_mode": source_head_mode,
        "source_head_strategy": source_head_strategy,
        "shortlist_mode": shortlist_mode,
        "backbone_mode": backbone_mode,
        "global_feature_names": selected_global_features,
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
        "lambda_rank": lambda_rank,
        "rank_negatives_per_positive": rank_negatives_per_positive,
        "rank_hard_negative_fraction": rank_hard_negative_fraction,
        "lambda_preliminary": lambda_preliminary,
        "batch_size": batch_size,
        "resumed": resume_from is not None,
        "peak_memory_bytes": peak_memory_bytes(device),
        "metrics": metrics,
        "validation_prediction_metrics": validation_prediction_metrics,
        "prediction_metrics": prediction_metrics,
        "shortlist_validation": shortlist_validation,
        "test_evaluated": evaluate_test,
        "split_sizes": {split: len(examples) for split, examples in splits.items()},
        "runtime": runtime_metadata(device),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(f"[stage] Artifacts saved: {output_path}", flush=True)
    return summary


@torch.no_grad()
def evaluate_test_predictions(model, examples: list, graph) -> tuple[list[dict], dict]:
    """Evaluate selected checkpoint in estimated-k and oracle-k modes."""
    model.eval()
    rows: list[dict] = []
    for index, data in enumerate(examples):
        data = data.to(next(model.parameters()).device)
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
                    **source_radius_hits(graph, true_sources, prediction.sources),
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
                    "hit_at_1_hop",
                    "hit_at_2_hop",
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
