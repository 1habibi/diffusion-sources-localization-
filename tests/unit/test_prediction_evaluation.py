from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data

from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.train_cli import evaluate_test_predictions


def test_evaluate_test_predictions_returns_both_joint_modes():
    graph = nx.path_graph(3)
    data = Data(
        x=torch.tensor([[1.0, 0.5], [1.0, 1.0], [0.0, 0.5]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        observed_mask=torch.tensor([True, True, False]),
        candidate_mask=torch.tensor([True, True, False]),
        source_labels=torch.tensor([1.0, 0.0, 0.0]),
        source_count=torch.tensor(1),
    )
    model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)

    rows, metrics = evaluate_test_predictions(model, [data], graph)

    assert len(rows) == 2
    assert set(metrics) == {"joint_estimated_k", "joint_oracle_k"}
    assert all(0.0 <= row["f1"] <= 1.0 for row in rows)
