"""Masked multi-label and joint source-count training objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class JointLoss:
    total: torch.Tensor
    source: torch.Tensor
    count: torch.Tensor
    consistency: torch.Tensor
    ranking: torch.Tensor
    preliminary: torch.Tensor


def select_source_logits_for_counts(
    source_logits: torch.Tensor,
    source_count: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route each node to its graph's k-specific head, or keep shared logits."""
    if source_logits.ndim == 1:
        return source_logits
    if source_logits.ndim != 2 or source_logits.size(1) != 3:
        raise ValueError("source_logits must have shape [nodes] or [nodes, 3].")
    counts = source_count.reshape(-1).long()
    if not torch.all((counts >= 1) & (counts <= 3)):
        raise ValueError("source_count values must be in 1..3.")
    if batch is None:
        batch = torch.zeros(
            source_logits.size(0), dtype=torch.long, device=source_logits.device
        )
    if batch.numel() != source_logits.size(0):
        raise ValueError("batch must contain one graph index per node.")
    if batch.numel() and int(batch.max().item()) >= counts.numel():
        raise ValueError("source_count must contain one value per graph.")
    head_indices = counts[batch] - 1
    return source_logits.gather(1, head_indices.unsqueeze(1)).squeeze(1)


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


def pairwise_source_ranking_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    batch: torch.Tensor | None = None,
    *,
    negatives_per_positive: int = 8,
    hard_negative_fraction: float = 0.5,
    random_sampling: bool = True,
) -> torch.Tensor:
    """Rank true sources above sampled false candidates within each graph."""
    if source_logits.ndim != 1:
        raise ValueError("pairwise ranking expects one routed logit per node.")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive.")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be between 0 and 1.")
    if batch is None:
        batch = torch.zeros(
            source_logits.size(0), dtype=torch.long, device=source_logits.device
        )
    if batch.numel() != source_logits.size(0):
        raise ValueError("batch must contain one graph index per node.")

    candidate_mask = candidate_mask.bool()
    positive_mask = candidate_mask & source_labels.bool()
    negative_mask = candidate_mask & ~source_labels.bool()
    pair_losses: list[torch.Tensor] = []
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_index in range(graph_count):
        graph_mask = batch == graph_index
        positives = torch.nonzero(
            graph_mask & positive_mask, as_tuple=False
        ).flatten()
        negatives = torch.nonzero(
            graph_mask & negative_mask, as_tuple=False
        ).flatten()
        if positives.numel() == 0 or negatives.numel() == 0:
            continue

        sample_count = min(negatives_per_positive, int(negatives.numel()))
        hard_count = min(
            sample_count,
            int(math.ceil(sample_count * hard_negative_fraction)),
        )
        ranked_negatives = negatives[
            torch.argsort(source_logits[negatives].detach(), descending=True)
        ]
        hard_negatives = ranked_negatives[:hard_count]
        remaining = ranked_negatives[hard_count:]
        random_count = sample_count - hard_count

        for positive in positives:
            if random_count and remaining.numel():
                if random_sampling:
                    permutation = torch.randperm(
                        remaining.numel(), device=remaining.device
                    )
                    sampled_random = remaining[permutation[:random_count]]
                else:
                    positions = torch.linspace(
                        0,
                        remaining.numel() - 1,
                        steps=random_count,
                        device=remaining.device,
                    ).round().long()
                    sampled_random = remaining[positions]
                sampled_negatives = torch.cat(
                    [hard_negatives, sampled_random]
                )
            else:
                sampled_negatives = hard_negatives
            pair_losses.append(
                F.softplus(
                    source_logits[sampled_negatives] - source_logits[positive]
                )
            )

    if not pair_losses:
        return source_logits.sum() * 0.0
    return torch.cat(pair_losses).mean()


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
    lambda_rank: float = 0.0,
    rank_negatives_per_positive: int = 8,
    rank_hard_negative_fraction: float = 0.5,
    rank_random_sampling: bool = True,
    preliminary_logits: torch.Tensor | None = None,
    lambda_preliminary: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> JointLoss:
    """Combine source BCE, count CE and differentiable cardinality consistency."""
    if lambda_rank < 0.0:
        raise ValueError("lambda_rank must be non-negative.")
    if lambda_preliminary < 0.0:
        raise ValueError("lambda_preliminary must be non-negative.")
    routed_source_logits = select_source_logits_for_counts(
        source_logits, source_count, batch
    )
    source = masked_source_loss(
        routed_source_logits, source_labels, candidate_mask, pos_weight=pos_weight
    )
    count_targets = source_count.reshape(-1).long() - 1
    count = F.cross_entropy(count_logits, count_targets)

    if batch is None:
        batch = torch.zeros(
            routed_source_logits.size(0),
            dtype=torch.long,
            device=routed_source_logits.device,
        )
    mask = candidate_mask.bool()
    candidate_probabilities = torch.sigmoid(routed_source_logits[mask])
    probability_sums = torch.zeros(
        count_logits.size(0),
        dtype=routed_source_logits.dtype,
        device=routed_source_logits.device,
    )
    probability_sums.index_add_(0, batch[mask], candidate_probabilities)
    count_values = torch.arange(
        1, 4, device=count_logits.device, dtype=routed_source_logits.dtype
    )
    expected_count = (torch.softmax(count_logits, dim=-1) * count_values).sum(dim=-1)
    consistency = F.mse_loss(probability_sums, expected_count)
    ranking = (
        pairwise_source_ranking_loss(
            routed_source_logits,
            source_labels,
            candidate_mask,
            batch,
            negatives_per_positive=rank_negatives_per_positive,
            hard_negative_fraction=rank_hard_negative_fraction,
            random_sampling=rank_random_sampling,
        )
        if lambda_rank > 0.0
        else routed_source_logits.sum() * 0.0
    )
    if lambda_preliminary > 0.0:
        if preliminary_logits is None:
            raise ValueError(
                "lambda_preliminary > 0 requires preliminary_logits."
            )
        preliminary = masked_source_loss(
            preliminary_logits,
            source_labels,
            candidate_mask,
            pos_weight=pos_weight,
        )
    else:
        preliminary = routed_source_logits.sum() * 0.0
    total = (
        source
        + lambda_count * count
        + lambda_consistency * consistency
        + lambda_rank * ranking
        + lambda_preliminary * preliminary
    )
    return JointLoss(
        total=total,
        source=source,
        count=count,
        consistency=consistency,
        ranking=ranking,
        preliminary=preliminary,
    )
