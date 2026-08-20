"""Validation-only safety evaluation for optional two-stage shortlists."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .inference import predict_joint
from .metrics import set_metrics


@torch.no_grad()
def evaluate_shortlist_grid(
    model: torch.nn.Module,
    examples: Iterable,
    shortlist_sizes: Iterable[int],
    *,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 17026,
    micro_recall_min: float = 0.95,
    per_k_recall_min: float = 0.95,
    bootstrap_ci_low_min: float = 0.93,
    require_f1_or_latency_improvement: bool = True,
    progress_description: str | None = None,
) -> dict[str, Any]:
    """Compare full scoring with safe top-M inference on validation examples."""
    sizes = tuple(dict.fromkeys(int(size) for size in shortlist_sizes))
    if not sizes or any(size < 3 for size in sizes):
        raise ValueError("shortlist sizes must be unique integers of at least 3.")
    if bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100.")
    if any(
        not 0.0 <= threshold <= 1.0
        for threshold in (
            micro_recall_min,
            per_k_recall_min,
            bootstrap_ci_low_min,
        )
    ):
        raise ValueError("Shortlist recall thresholds must be between 0 and 1.")
    data_items = tuple(examples)
    if not data_items:
        raise ValueError("Shortlist evaluation requires examples.")
    if not hasattr(model, "forward_shortlisted"):
        raise ValueError("Model does not support shortlist inference.")

    model.eval()
    results = {}
    for size in tqdm(
        sizes,
        desc=progress_description,
        unit="size",
        leave=False,
        disable=progress_description is None,
    ):
        results[str(size)] = _evaluate_shortlist_size(
            model,
            data_items,
            size,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_seed=bootstrap_seed + size,
            micro_recall_min=micro_recall_min,
            per_k_recall_min=per_k_recall_min,
            bootstrap_ci_low_min=bootstrap_ci_low_min,
            require_f1_or_latency_improvement=require_f1_or_latency_improvement,
            progress_description=(
                f"Shortlist M={size}" if progress_description is not None else None
            ),
        )
    eligible = [
        (size, results[str(size)])
        for size in sizes
        if results[str(size)]["eligible"]
    ]
    selected_size = (
        max(
            eligible,
            key=lambda item: (
                item[1]["shortlist_macro_f1"] - item[1]["full_macro_f1"],
                -item[1]["shortlist_latency_seconds"],
                -item[0],
            ),
        )[0]
        if eligible
        else None
    )
    return {
        "selection_split": "validation",
        "sizes": list(sizes),
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": bootstrap_seed,
        "criteria": {
            "micro_candidate_recall_min": micro_recall_min,
            "per_k_candidate_recall_min": per_k_recall_min,
            "bootstrap_ci_low_min": bootstrap_ci_low_min,
            "requires_f1_or_latency_improvement": require_f1_or_latency_improvement,
        },
        "results": results,
        "selected_size": selected_size,
        "decision": "eligible" if selected_size is not None else "reject_shortlist",
    }


def _evaluate_shortlist_size(
    model: torch.nn.Module,
    examples: tuple,
    shortlist_size: int,
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    micro_recall_min: float,
    per_k_recall_min: float,
    bootstrap_ci_low_min: float,
    require_f1_or_latency_improvement: bool,
    progress_description: str | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    retained_counts: list[int] = []
    source_counts: list[int] = []
    full_f1: list[float] = []
    shortlist_f1: list[float] = []
    fallback_count = 0
    original_candidates = 0
    shortlisted_candidates = 0
    full_seconds = 0.0
    shortlist_seconds = 0.0

    for example in tqdm(
        examples,
        desc=progress_description,
        unit="example",
        leave=False,
        disable=progress_description is None,
    ):
        data = example.to(device)
        _synchronize(device)
        started = time.perf_counter()
        full_source_logits, full_count_logits = model(data)
        full_prediction = predict_joint(
            full_source_logits, full_count_logits, data.candidate_mask
        )
        _synchronize(device)
        full_seconds += time.perf_counter() - started

        _synchronize(device)
        started = time.perf_counter()
        (
            shortlist_source_logits,
            shortlist_count_logits,
            _,
            shortlist_mask,
            fallback_flags,
        ) = model.forward_shortlisted(data, shortlist_size)
        shortlist_prediction = predict_joint(
            shortlist_source_logits, shortlist_count_logits, shortlist_mask
        )
        _synchronize(device)
        shortlist_seconds += time.perf_counter() - started

        true_sources = frozenset(
            torch.nonzero(data.source_labels, as_tuple=False)
            .flatten()
            .cpu()
            .tolist()
        )
        true_count = int(data.source_count.item())
        retained = len(true_sources & frozenset(
            torch.nonzero(shortlist_mask, as_tuple=False)
            .flatten()
            .cpu()
            .tolist()
        ))
        retained_counts.append(retained)
        source_counts.append(true_count)
        fallback_count += int(any(fallback_flags))
        original_candidates += int(data.candidate_mask.sum().item())
        shortlisted_candidates += int(shortlist_mask.sum().item())

        full_f1.append(set_metrics(true_sources, full_prediction.sources)["f1"])
        shortlist_f1.append(
            set_metrics(true_sources, shortlist_prediction.sources)["f1"]
        )

    retained_array = np.asarray(retained_counts, dtype=np.int64)
    source_array = np.asarray(source_counts, dtype=np.int64)
    micro_recall = float(retained_array.sum() / source_array.sum())
    by_k = {
        str(k): float(
            retained_array[source_array == k].sum()
            / source_array[source_array == k].sum()
        )
        for k in (1, 2, 3)
        if np.any(source_array == k)
    }
    ci_low, ci_high = _bootstrap_micro_recall(
        retained_array,
        source_array,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    full_macro_f1 = float(np.mean(full_f1))
    shortlist_macro_f1 = float(np.mean(shortlist_f1))
    f1_improved = shortlist_macro_f1 > full_macro_f1
    latency_improved = shortlist_seconds < full_seconds
    safety_passed = (
        micro_recall >= micro_recall_min
        and all(value >= per_k_recall_min for value in by_k.values())
        and len(by_k) == 3
        and ci_low >= bootstrap_ci_low_min
    )
    benefit_passed = f1_improved or latency_improved
    return {
        "shortlist_size": shortlist_size,
        "example_count": len(examples),
        "micro_candidate_recall": micro_recall,
        "candidate_recall_by_k": by_k,
        "candidate_recall_bootstrap_ci_95": [ci_low, ci_high],
        "full_macro_f1": full_macro_f1,
        "shortlist_macro_f1": shortlist_macro_f1,
        "full_latency_seconds": full_seconds,
        "shortlist_latency_seconds": shortlist_seconds,
        "latency_ratio": shortlist_seconds / full_seconds if full_seconds else None,
        "candidate_reduction_fraction": 1.0
        - shortlisted_candidates / original_candidates,
        "fallback_rate": fallback_count / len(examples),
        "safety_passed": safety_passed,
        "f1_improved": f1_improved,
        "latency_improved": latency_improved,
        "benefit_passed": benefit_passed,
        "eligible": safety_passed
        and (benefit_passed or not require_f1_or_latency_improvement),
    }


def _bootstrap_micro_recall(
    retained: np.ndarray,
    totals: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, len(retained), size=len(retained))
        values[index] = retained[sample].sum() / totals[sample].sum()
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
