"""Minimal reproducible training loop with report-ready epoch metrics."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .inference import predict_joint, predict_oracle_k
from .losses import joint_source_count_loss, masked_source_loss
from .metrics import set_metrics


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    source_loss: float
    count_loss: float
    consistency_loss: float
    macro_f1: float
    pr_auc: float
    count_accuracy: float
    count_mae: float
    duration_seconds: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    stopped_epoch: int
    stop_reason: str
    train_history: tuple[EpochMetrics, ...]
    validation_history: tuple[EpochMetrics, ...]
    best_state_dict: dict[str, torch.Tensor]


def train_one_epoch(
    model: torch.nn.Module,
    examples: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    lambda_count: float = 1.0,
    lambda_consistency: float = 0.1,
    pos_weight: torch.Tensor | None = None,
) -> None:
    """Perform one optimization pass over single-graph PyG examples."""
    model.train()
    for data in examples:
        optimizer.zero_grad()
        source_logits, count_logits = model(data)
        loss = joint_source_count_loss(
            source_logits,
            count_logits,
            data.source_labels,
            data.source_count,
            data.candidate_mask,
            getattr(data, "batch", None),
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
        )
        loss.total.backward()
        optimizer.step()


def train_node_one_epoch(
    model: torch.nn.Module,
    examples: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    pos_weight: torch.Tensor | None = None,
) -> None:
    """Perform one optimization pass for the node-only baseline."""
    model.train()
    for data in examples:
        optimizer.zero_grad()
        logits = model(data)
        loss = masked_source_loss(
            logits, data.source_labels, data.candidate_mask, pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    examples: Iterable,
    *,
    learning_rate: float = 0.0,
    lambda_count: float = 1.0,
    lambda_consistency: float = 0.1,
    pos_weight: torch.Tensor | None = None,
) -> EpochMetrics:
    """Evaluate losses and localization/count metrics with dropout disabled."""
    started_at = time.perf_counter()
    model.eval()
    losses: list[tuple[float, float, float, float]] = []
    f1_scores: list[float] = []
    count_matches: list[float] = []
    count_errors: list[float] = []
    labels_for_ap: list[float] = []
    scores_for_ap: list[float] = []

    for data in examples:
        source_logits, count_logits = model(data)
        loss = joint_source_count_loss(
            source_logits,
            count_logits,
            data.source_labels,
            data.source_count,
            data.candidate_mask,
            getattr(data, "batch", None),
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
        )
        losses.append(
            (
                loss.total.item(),
                loss.source.item(),
                loss.count.item(),
                loss.consistency.item(),
            )
        )
        prediction = predict_joint(source_logits, count_logits, data.candidate_mask)
        true_sources = set(torch.nonzero(data.source_labels, as_tuple=False).flatten().tolist())
        f1_scores.append(set_metrics(true_sources, prediction.sources)["f1"])
        true_count = int(data.source_count.item())
        count_matches.append(float(prediction.source_count == true_count))
        count_errors.append(float(abs(prediction.source_count - true_count)))
        mask = data.candidate_mask.bool()
        labels_for_ap.extend(data.source_labels[mask].cpu().tolist())
        scores_for_ap.extend(prediction.scores[mask].cpu().tolist())

    if not losses:
        raise ValueError("At least one example is required for evaluation.")
    loss_array = np.asarray(losses)
    pr_auc = (
        float(average_precision_score(labels_for_ap, scores_for_ap))
        if any(labels_for_ap)
        else 0.0
    )
    return EpochMetrics(
        loss=float(loss_array[:, 0].mean()),
        source_loss=float(loss_array[:, 1].mean()),
        count_loss=float(loss_array[:, 2].mean()),
        consistency_loss=float(loss_array[:, 3].mean()),
        macro_f1=float(np.mean(f1_scores)),
        pr_auc=pr_auc,
        count_accuracy=float(np.mean(count_matches)),
        count_mae=float(np.mean(count_errors)),
        duration_seconds=time.perf_counter() - started_at,
        learning_rate=learning_rate,
    )


@torch.no_grad()
def evaluate_node_epoch(
    model: torch.nn.Module,
    examples: Iterable,
    *,
    learning_rate: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> EpochMetrics:
    """Evaluate Node-only GCN in oracle-k mode for checkpoint selection."""
    started_at = time.perf_counter()
    model.eval()
    losses: list[float] = []
    f1_scores: list[float] = []
    labels_for_ap: list[float] = []
    scores_for_ap: list[float] = []
    for data in examples:
        logits = model(data)
        loss = masked_source_loss(
            logits, data.source_labels, data.candidate_mask, pos_weight=pos_weight
        )
        losses.append(float(loss.item()))
        true_count = int(data.source_count.item())
        prediction = predict_oracle_k(logits, data.candidate_mask, true_count)
        true_sources = set(torch.nonzero(data.source_labels, as_tuple=False).flatten().tolist())
        f1_scores.append(set_metrics(true_sources, prediction.sources)["f1"])
        mask = data.candidate_mask.bool()
        labels_for_ap.extend(data.source_labels[mask].cpu().tolist())
        scores_for_ap.extend(prediction.scores[mask].cpu().tolist())
    if not losses:
        raise ValueError("At least one example is required for evaluation.")
    return EpochMetrics(
        loss=float(np.mean(losses)),
        source_loss=float(np.mean(losses)),
        count_loss=0.0,
        consistency_loss=0.0,
        macro_f1=float(np.mean(f1_scores)),
        pr_auc=float(average_precision_score(labels_for_ap, scores_for_ap)),
        count_accuracy=1.0,
        count_mae=0.0,
        duration_seconds=time.perf_counter() - started_at,
        learning_rate=learning_rate,
    )


def fit_joint_model(
    model: torch.nn.Module,
    train_examples: Iterable,
    validation_examples: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int = 100,
    patience: int = 10,
    lambda_count: float = 1.0,
    lambda_consistency: float = 0.1,
    pos_weight: torch.Tensor | None = None,
) -> TrainingResult:
    """Train with early stopping on validation macro-F1."""
    train_data = tuple(train_examples)
    validation_data = tuple(validation_examples)
    if not train_data or not validation_data:
        raise ValueError("Train and validation sets must not be empty.")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    train_history: list[EpochMetrics] = []
    validation_history: list[EpochMetrics] = []
    best_epoch = 0
    best_score = -1.0
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    for epoch in range(1, max_epochs + 1):
        train_one_epoch(
            model,
            train_data,
            optimizer,
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = evaluate_epoch(
            model,
            train_data,
            learning_rate=learning_rate,
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
        )
        validation_metrics = evaluate_epoch(
            model,
            validation_data,
            learning_rate=learning_rate,
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)

        if validation_metrics.macro_f1 > best_score:
            best_score = validation_metrics.macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stop_reason = "early_stopping"
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        stopped_epoch=len(train_history),
        stop_reason=stop_reason,
        train_history=tuple(train_history),
        validation_history=tuple(validation_history),
        best_state_dict=best_state,
    )


def fit_node_model(
    model: torch.nn.Module,
    train_examples: Iterable,
    validation_examples: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int = 100,
    patience: int = 10,
    pos_weight: torch.Tensor | None = None,
) -> TrainingResult:
    """Train Node-only GCN with early stopping on validation oracle-k F1."""
    train_data = tuple(train_examples)
    validation_data = tuple(validation_examples)
    if not train_data or not validation_data:
        raise ValueError("Train and validation sets must not be empty.")
    train_history: list[EpochMetrics] = []
    validation_history: list[EpochMetrics] = []
    best_epoch, best_score, stale_epochs = 0, -1.0, 0
    best_state: dict[str, torch.Tensor] = {}
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        train_node_one_epoch(model, train_data, optimizer, pos_weight=pos_weight)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = evaluate_node_epoch(
            model, train_data, learning_rate=learning_rate, pos_weight=pos_weight
        )
        validation_metrics = evaluate_node_epoch(
            model, validation_data, learning_rate=learning_rate, pos_weight=pos_weight
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)
        if validation_metrics.macro_f1 > best_score:
            best_epoch, best_score, stale_epochs = epoch, validation_metrics.macro_f1, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stop_reason = "early_stopping"
                break
    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        stopped_epoch=len(train_history),
        stop_reason=stop_reason,
        train_history=tuple(train_history),
        validation_history=tuple(validation_history),
        best_state_dict=best_state,
    )


def save_training_result(result: TrainingResult, output_dir: str | Path) -> None:
    """Save histories as JSON/CSV and the selected checkpoint for reporting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "stop_reason": result.stop_reason,
        "train": [asdict(item) for item in result.train_history],
        "validation": [asdict(item) for item in result.validation_history],
    }
    (output_path / "history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_path / "history.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["split", "epoch", *asdict(result.train_history[0]).keys()]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, history in (
            ("train", result.train_history),
            ("validation", result.validation_history),
        ):
            for epoch, metrics in enumerate(history, start=1):
                writer.writerow({"split": split, "epoch": epoch, **asdict(metrics)})
    torch.save(result.best_state_dict, output_path / "best_model.pt")
