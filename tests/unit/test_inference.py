from __future__ import annotations

import torch

from diffusion_sources.inference import predict_joint, predict_oracle_k, predict_thresholded


def test_oracle_k_only_selects_candidates():
    prediction = predict_oracle_k(
        torch.tensor([0.1, 2.0, 10.0]), torch.tensor([True, True, False]), 1
    )
    assert prediction.sources == frozenset({1})


def test_joint_uses_count_head_and_node_scores():
    prediction = predict_joint(
        torch.tensor([3.0, 2.0, 1.0]),
        torch.tensor([[0.0, 3.0, 0.0]]),
        torch.tensor([True, True, True]),
    )
    assert prediction.source_count == 2
    assert prediction.sources == frozenset({0, 1})


def test_thresholded_prediction_is_constrained_to_one_to_three_sources():
    prediction = predict_thresholded(
        torch.tensor([-10.0, -9.0, -8.0]),
        torch.tensor([True, True, True]),
        0.9,
    )
    assert prediction.source_count == 1
    assert prediction.sources == frozenset({2})


def test_joint_selects_specialized_head_from_predicted_count():
    specialized_logits = torch.tensor(
        [
            [10.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 2.0, 10.0],
        ]
    )

    prediction = predict_joint(
        specialized_logits,
        torch.tensor([[0.0, 5.0, 0.0]]),
        torch.tensor([True, True, True]),
    )

    assert prediction.source_count == 2
    assert prediction.sources == frozenset({1, 2})


def test_oracle_k_selects_matching_specialized_head():
    specialized_logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )

    prediction = predict_oracle_k(
        specialized_logits, torch.tensor([True, True, True]), 1
    )

    assert prediction.sources == frozenset({0})
