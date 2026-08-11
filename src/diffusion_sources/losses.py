"""Masked multi-label and joint source-count training objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class JointLoss:
    total: torch.Tensor
    source: torch.Tensor
    count: torch.Tensor
    consistency: torch.Tensor


def masked_source_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate BCE only over candidates allowed by the experiment mode."""
    mask = candidate_mask.bool()
    if not mask.any():
        raise ValueError("candidate_mask must contain at least one node.")
    return F.binary_cross_entropy_with_logits(
        source_logits[mask], source_labels.float()[mask], pos_weight=pos_weight
    )


def joint_source_count_loss(
    source_logits: torch.Tensor,
    count_logits: torch.Tensor,
    source_labels: torch.Tensor,
    source_count: torch.Tensor,
    candidate_mask: torch.Tensor,
    batch: torch.Tensor | None = None,
    *,
    lambda_count: float = 1.0,
    lambda_consistency: float = 0.1,
    pos_weight: torch.Tensor | None = None,
) -> JointLoss:
    """Combine source BCE, count CE and differentiable cardinality consistency."""
    source = masked_source_loss(
        source_logits, source_labels, candidate_mask, pos_weight=pos_weight
    )
    count_targets = source_count.reshape(-1).long() - 1
    count = F.cross_entropy(count_logits, count_targets)

    if batch is None:
        batch = torch.zeros(
            source_logits.size(0), dtype=torch.long, device=source_logits.device
        )
    mask = candidate_mask.bool()
    candidate_probabilities = torch.sigmoid(source_logits[mask])
    probability_sums = torch.zeros(
        count_logits.size(0), dtype=source_logits.dtype, device=source_logits.device
    )
    probability_sums.index_add_(0, batch[mask], candidate_probabilities)
    count_values = torch.arange(1, 4, device=count_logits.device, dtype=source_logits.dtype)
    expected_count = (torch.softmax(count_logits, dim=-1) * count_values).sum(dim=-1)
    consistency = F.mse_loss(probability_sums, expected_count)
    total = source + lambda_count * count + lambda_consistency * consistency
    return JointLoss(total=total, source=source, count=count, consistency=consistency)
