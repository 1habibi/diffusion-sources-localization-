"""Minimal reproducible training loop with report-ready epoch metrics."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

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


def save_last_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_epoch: int,
    best_score: float,
    best_state_dict: dict[str, torch.Tensor],
    stale_epochs: int,
    train_history: list[EpochMetrics],
    validation_history: list[EpochMetrics],
) -> None:
    """Atomically save all state needed to resume at the next epoch."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_state_dict": best_state_dict,
            "stale_epochs": stale_epochs,
            "train_history": [asdict(item) for item in train_history],
            "validation_history": [asdict(item) for item in validation_history],
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def load_last_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> dict[str, Any]:
    """Restore model and optimizer and return loop bookkeeping state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and checkpoint["cuda_random_states"] is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_random_states"]]
        )
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_score": float(checkpoint["best_score"]),
        "best_state": checkpoint["best_state_dict"],
        "stale_epochs": int(checkpoint["stale_epochs"]),
        "train_history": [EpochMetrics(**item) for item in checkpoint["train_history"]],
        "validation_history": [
            EpochMetrics(**item) for item in checkpoint["validation_history"]
        ],
    }


def train_one_epoch(
    model: torch.nn.Module,
    examples: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    lambda_count: float = 1.0,
    lambda_consistency: float = 0.1,
    pos_weight: torch.Tensor | None = None,
    batch_size: int = 1,
    progress_description: str | None = None,
) -> None:
    """Perform one optimization pass over single-graph PyG examples."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model.train()
    loader = DataLoader(tuple(examples), batch_size=batch_size, shuffle=True)
    for data in tqdm(loader, desc=progress_description, leave=False, disable=progress_description is None):
        data = data.to(next(model.parameters()).device)
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
    batch_size: int = 1,
    progress_description: str | None = None,
) -> None:
    """Perform one optimization pass for the node-only baseline."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model.train()
    loader = DataLoader(tuple(examples), batch_size=batch_size, shuffle=True)
    for data in tqdm(loader, desc=progress_description, leave=False, disable=progress_description is None):
        data = data.to(next(model.parameters()).device)
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
    batch_size: int = 1,
    progress_description: str | None = None,
) -> EpochMetrics:
    """Evaluate losses and localization/count metrics with dropout disabled."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    started_at = time.perf_counter()
    model.eval()
    losses: list[tuple[float, float, float, float]] = []
    f1_scores: list[float] = []
    count_matches: list[float] = []
    count_errors: list[float] = []
    labels_for_ap: list[float] = []
    scores_for_ap: list[float] = []

    loader = DataLoader(tuple(examples), batch_size=batch_size, shuffle=False)
    for data in tqdm(loader, desc=progress_description, leave=False, disable=progress_description is None):
        data = data.to(next(model.parameters()).device)
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
        ptr = data.ptr if hasattr(data, "ptr") else torch.tensor([0, data.num_nodes])
        for graph_index in range(count_logits.size(0)):
            start = int(ptr[graph_index].item())
            end = int(ptr[graph_index + 1].item())
            prediction = predict_joint(
                source_logits[start:end],
                count_logits[graph_index : graph_index + 1],
                data.candidate_mask[start:end],
            )
            true_sources = set(
                torch.nonzero(
                    data.source_labels[start:end], as_tuple=False
                ).flatten().tolist()
            )
            f1_scores.append(set_metrics(true_sources, prediction.sources)["f1"])
            true_count = int(data.source_count[graph_index].item())
            count_matches.append(float(prediction.source_count == true_count))
            count_errors.append(float(abs(prediction.source_count - true_count)))
        mask = data.candidate_mask.bool()
        labels_for_ap.extend(data.source_labels[mask].cpu().tolist())
        scores_for_ap.extend(torch.sigmoid(source_logits[mask]).cpu().tolist())

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
    batch_size: int = 1,
    progress_description: str | None = None,
) -> EpochMetrics:
    """Evaluate Node-only GCN in oracle-k mode for checkpoint selection."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    started_at = time.perf_counter()
    model.eval()
    losses: list[float] = []
    f1_scores: list[float] = []
    labels_for_ap: list[float] = []
    scores_for_ap: list[float] = []
    loader = DataLoader(tuple(examples), batch_size=batch_size, shuffle=False)
    for data in tqdm(loader, desc=progress_description, leave=False, disable=progress_description is None):
        data = data.to(next(model.parameters()).device)
        logits = model(data)
        loss = masked_source_loss(
            logits, data.source_labels, data.candidate_mask, pos_weight=pos_weight
        )
        losses.append(float(loss.item()))
        ptr = data.ptr if hasattr(data, "ptr") else torch.tensor([0, data.num_nodes])
        source_counts = data.source_count.reshape(-1)
        for graph_index in range(len(source_counts)):
            start = int(ptr[graph_index].item())
            end = int(ptr[graph_index + 1].item())
            true_count = int(source_counts[graph_index].item())
            prediction = predict_oracle_k(
                logits[start:end], data.candidate_mask[start:end], true_count
            )
            true_sources = set(
                torch.nonzero(
                    data.source_labels[start:end], as_tuple=False
                ).flatten().tolist()
            )
            f1_scores.append(set_metrics(true_sources, prediction.sources)["f1"])
        mask = data.candidate_mask.bool()
        labels_for_ap.extend(data.source_labels[mask].cpu().tolist())
        scores_for_ap.extend(torch.sigmoid(logits[mask]).cpu().tolist())
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
    batch_size: int = 1,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train with early stopping on validation macro-F1."""
    train_data = tuple(train_examples)
    validation_data = tuple(validation_examples)
    if not train_data or not validation_data:
        raise ValueError("Train and validation sets must not be empty.")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    state = _initial_training_state(model)
    if resume_from is not None:
        state = load_last_checkpoint(
            resume_from, model, optimizer, next(model.parameters()).device
        )
    train_history = state["train_history"]
    validation_history = state["validation_history"]
    best_epoch = state["best_epoch"]
    best_score = state["best_score"]
    best_state = state["best_state"]
    epochs_without_improvement = state["stale_epochs"]
    stop_reason = "max_epochs"

    for epoch in range(state["start_epoch"], max_epochs + 1):
        train_one_epoch(
            model,
            train_data,
            optimizer,
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
            batch_size=batch_size,
            progress_description=f"Epoch {epoch}/{max_epochs} train",
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = evaluate_epoch(
            model,
            train_data,
            learning_rate=learning_rate,
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
            batch_size=batch_size,
            progress_description=f"Epoch {epoch}/{max_epochs} train eval",
        )
        validation_metrics = evaluate_epoch(
            model,
            validation_data,
            learning_rate=learning_rate,
            lambda_count=lambda_count,
            lambda_consistency=lambda_consistency,
            pos_weight=pos_weight,
            batch_size=batch_size,
            progress_description=f"Epoch {epoch}/{max_epochs} validation",
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)

        improved = validation_metrics.macro_f1 > best_score
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
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val F1 {validation_metrics.macro_f1:.4f} | "
            f"best F1 {best_score:.4f} | "
            f"{'improved' if improved else f'patience {epochs_without_improvement}/{patience}'} | "
            f"{train_metrics.duration_seconds + validation_metrics.duration_seconds:.1f}s"
        )
        if checkpoint_path is not None:
            save_last_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_epoch=best_epoch,
                best_score=best_score,
                best_state_dict=best_state,
                stale_epochs=epochs_without_improvement,
                train_history=train_history,
                validation_history=validation_history,
            )
        if stop_reason == "early_stopping":
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
    batch_size: int = 1,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train Node-only GCN with early stopping on validation oracle-k F1."""
    train_data = tuple(train_examples)
    validation_data = tuple(validation_examples)
    if not train_data or not validation_data:
        raise ValueError("Train and validation sets must not be empty.")
    state = _initial_training_state(model)
    if resume_from is not None:
        state = load_last_checkpoint(
            resume_from, model, optimizer, next(model.parameters()).device
        )
    train_history = state["train_history"]
    validation_history = state["validation_history"]
    best_epoch = state["best_epoch"]
    best_score = state["best_score"]
    best_state = state["best_state"]
    stale_epochs = state["stale_epochs"]
    stop_reason = "max_epochs"
    for epoch in range(state["start_epoch"], max_epochs + 1):
        train_node_one_epoch(
            model,
            train_data,
            optimizer,
            pos_weight=pos_weight,
            batch_size=batch_size,
            progress_description=f"Epoch {epoch}/{max_epochs} train",
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = evaluate_node_epoch(
            model,
            train_data,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            batch_size=batch_size,
            progress_description=f"Epoch {epoch}/{max_epochs} train eval",
        )
        validation_metrics = evaluate_node_epoch(
            model,
            validation_data,
            learning_rate=learning_rate,
            pos_weight=pos_weight,
            batch_size=batch_size,
            progress_description=f"Epoch {epoch}/{max_epochs} validation",
        )
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)
        improved = validation_metrics.macro_f1 > best_score
        if validation_metrics.macro_f1 > best_score:
            best_epoch, best_score, stale_epochs = epoch, validation_metrics.macro_f1, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stop_reason = "early_stopping"
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val F1 {validation_metrics.macro_f1:.4f} | "
            f"best F1 {best_score:.4f} | "
            f"{'improved' if improved else f'patience {stale_epochs}/{patience}'} | "
            f"{train_metrics.duration_seconds + validation_metrics.duration_seconds:.1f}s"
        )
        if checkpoint_path is not None:
            save_last_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_epoch=best_epoch,
                best_score=best_score,
                best_state_dict=best_state,
                stale_epochs=stale_epochs,
                train_history=train_history,
                validation_history=validation_history,
            )
        if stop_reason == "early_stopping":
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


def _initial_training_state(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "start_epoch": 1,
        "best_epoch": 0,
        "best_score": -1.0,
        "best_state": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "stale_epochs": 0,
        "train_history": [],
        "validation_history": [],
    }


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
