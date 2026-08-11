"""Convert neural model outputs into sets of predicted source nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SourcePrediction:
    scores: torch.Tensor
    source_count: int
    sources: frozenset[int]
    count_probabilities: torch.Tensor | None = None


def predict_oracle_k(
    source_logits: torch.Tensor, candidate_mask: torch.Tensor, source_count: int
) -> SourcePrediction:
    """Select the highest-scoring candidates when true cardinality is known."""
    scores = torch.sigmoid(source_logits.detach())
    candidates = torch.nonzero(candidate_mask.bool(), as_tuple=False).flatten()
    if not 1 <= source_count <= len(candidates):
        raise ValueError("source_count must fit the candidate set.")
    selected = candidates[torch.topk(scores[candidates], source_count).indices]
    return SourcePrediction(scores, source_count, frozenset(selected.cpu().tolist()))


def predict_joint(
    source_logits: torch.Tensor,
    count_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> SourcePrediction:
    """Estimate cardinality and select top-scoring candidate source nodes."""
    if count_logits.shape != (1, 3):
        raise ValueError("predict_joint currently expects one graph with three count logits.")
    probabilities = torch.softmax(count_logits.detach(), dim=-1).squeeze(0)
    source_count = int(torch.argmax(probabilities).item()) + 1
    available_candidates = int(candidate_mask.bool().sum().item())
    if available_candidates == 0:
        raise ValueError("candidate_mask must contain at least one node.")
    selected_count = min(source_count, available_candidates)
    oracle_prediction = predict_oracle_k(source_logits, candidate_mask, selected_count)
    return SourcePrediction(
        scores=oracle_prediction.scores,
        source_count=source_count,
        sources=oracle_prediction.sources,
        count_probabilities=probabilities,
    )


def predict_thresholded(
    source_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    threshold: float,
    *,
    max_sources: int = 3,
) -> SourcePrediction:
    """Select candidates above a validation threshold, constrained to 1..max_sources."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    scores = torch.sigmoid(source_logits.detach())
    candidates = torch.nonzero(candidate_mask.bool(), as_tuple=False).flatten()
    ranked = candidates[torch.argsort(scores[candidates], descending=True)]
    selected = ranked[scores[ranked] >= threshold][:max_sources]
    if selected.numel() == 0:
        selected = ranked[:1]
    return SourcePrediction(scores, len(selected), frozenset(selected.cpu().tolist()))
