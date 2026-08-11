from __future__ import annotations

import torch

from diffusion_sources.losses import joint_source_count_loss, masked_source_loss


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
