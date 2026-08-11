from __future__ import annotations

import torch
from torch_geometric.data import Data

from diffusion_sources.models import JointSourceCountGCN, NodeOnlyGCN


def model_data() -> Data:
    return Data(
        x=torch.tensor([[1.0, 0.5], [1.0, 1.0], [0.0, 0.5]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        observed_mask=torch.tensor([True, True, False]),
        candidate_mask=torch.tensor([True, True, False]),
        source_labels=torch.tensor([1.0, 0.0, 0.0]),
        source_count=torch.tensor(1),
    )


def test_node_only_model_returns_one_logit_per_node():
    logits = NodeOnlyGCN(hidden_dim=8, dropout=0.0)(model_data())
    assert logits.shape == (3,)


def test_joint_model_returns_node_and_count_logits():
    source_logits, count_logits = JointSourceCountGCN(hidden_dim=8, dropout=0.0)(
        model_data()
    )
    assert source_logits.shape == (3,)
    assert count_logits.shape == (1, 3)
