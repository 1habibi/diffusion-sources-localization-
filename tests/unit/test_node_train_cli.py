from __future__ import annotations

import torch
from torch_geometric.data import Data

from diffusion_sources.models import NodeOnlyGCN
from diffusion_sources.node_train_cli import select_threshold


def test_select_threshold_returns_value_from_validation_grid():
    data = Data(
        x=torch.tensor([[1.0, 0.5], [1.0, 1.0], [0.0, 0.5]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        observed_mask=torch.tensor([True, True, False]),
        candidate_mask=torch.tensor([True, True, False]),
        source_labels=torch.tensor([1.0, 0.0, 0.0]),
        source_count=torch.tensor(1),
    )
    model = NodeOnlyGCN(hidden_dim=8, dropout=0.0)
    threshold, score = select_threshold(model, [data], [0.2, 0.5, 0.8])

    assert threshold in {0.2, 0.5, 0.8}
    assert 0.0 <= score <= 1.0
