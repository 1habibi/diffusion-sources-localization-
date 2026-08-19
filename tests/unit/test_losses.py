from __future__ import annotations

import pytest
import torch

from diffusion_sources.losses import (
    joint_source_count_loss,
    masked_source_loss,
    pairwise_source_ranking_loss,
    select_source_logits_for_counts,
)


def test_masked_source_loss_ignores_nodes_outside_candidates():
    logits = torch.tensor([0.0, 0.0, 100.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 0.0])
    mask = torch.tensor([True, True, False])

    loss = masked_source_loss(logits, labels, mask)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[2] == 0.0


def test_joint_loss_is_finite_and_backpropagates():
    source_logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    count_logits = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)
    loss = joint_source_count_loss(
        source_logits,
        count_logits,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor(1),
        torch.tensor([True, True, False]),
    )
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert source_logits.grad is not None
    assert count_logits.grad is not None


def test_specialized_source_logits_are_routed_by_true_count_per_graph():
    logits = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )

    selected = select_source_logits_for_counts(
        logits,
        torch.tensor([1, 3]),
        torch.tensor([0, 0, 1, 1]),
    )

    assert torch.equal(selected, torch.tensor([1.0, 2.0, 300.0, 400.0]))


def test_joint_loss_only_backpropagates_through_true_k_specialized_head():
    source_logits = torch.zeros((3, 3), requires_grad=True)
    count_logits = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)

    loss = joint_source_count_loss(
        source_logits,
        count_logits,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor(2),
        torch.tensor([True, True, False]),
    )
    loss.total.backward()

    assert source_logits.grad is not None
    assert torch.equal(source_logits.grad[:, 0], torch.zeros(3))
    assert torch.count_nonzero(source_logits.grad[:, 1]) > 0
    assert torch.equal(source_logits.grad[:, 2], torch.zeros(3))


def test_pairwise_ranking_loss_prefers_positive_above_negative():
    labels = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([True, True])

    good = pairwise_source_ranking_loss(
        torch.tensor([3.0, -1.0]), labels, candidates
    )
    bad = pairwise_source_ranking_loss(
        torch.tensor([-1.0, 3.0]), labels, candidates
    )

    assert good < bad


def test_pairwise_ranking_loss_never_pairs_different_graphs():
    logits = torch.tensor([2.0, 0.0, -2.0, 0.0])
    loss = pairwise_source_ranking_loss(
        logits,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([True, True, True, True]),
        torch.tensor([0, 0, 1, 1]),
        negatives_per_positive=1,
        hard_negative_fraction=1.0,
    )
    expected = (
        torch.nn.functional.softplus(torch.tensor(-2.0))
        + torch.nn.functional.softplus(torch.tensor(2.0))
    ) / 2

    assert torch.allclose(loss, expected)


def test_hard_negative_sampling_selects_highest_scoring_false_candidate():
    loss = pairwise_source_ranking_loss(
        torch.tensor([0.0, -5.0, 3.0, 1.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([True, True, True, True]),
        negatives_per_positive=1,
        hard_negative_fraction=1.0,
    )

    assert torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(3.0)))


def test_zero_rank_weight_preserves_original_joint_objective():
    source_logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    count_logits = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)
    loss = joint_source_count_loss(
        source_logits,
        count_logits,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor(1),
        torch.tensor([True, True, False]),
        lambda_rank=0.0,
    )

    assert loss.ranking.item() == 0.0
    assert torch.allclose(
        loss.total, loss.source + loss.count + 0.1 * loss.consistency
    )


def test_preliminary_loss_trains_only_supplied_preliminary_logits():
    source_logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    count_logits = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)
    preliminary_logits = torch.zeros(3, requires_grad=True)

    loss = joint_source_count_loss(
        source_logits,
        count_logits,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor(1),
        torch.tensor([True, True, False]),
        preliminary_logits=preliminary_logits,
        lambda_preliminary=1.0,
    )
    loss.total.backward()

    assert loss.preliminary.item() > 0.0
    assert preliminary_logits.grad is not None
    assert preliminary_logits.grad[2] == 0.0


def test_preliminary_weight_requires_preliminary_logits():
    with pytest.raises(ValueError, match="preliminary_logits"):
        joint_source_count_loss(
            torch.tensor([0.2, -0.1, 0.4]),
            torch.tensor([[0.3, 0.1, -0.2]]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor(1),
            torch.tensor([True, True, False]),
            lambda_preliminary=1.0,
        )
