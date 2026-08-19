from __future__ import annotations

import torch
from torch_geometric.data import Data

from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.shortlist import evaluate_shortlist_grid


def model_data() -> Data:
    return Data(
        x=torch.tensor([[1.0, 0.5], [1.0, 1.0], [0.0, 0.5]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        observed_mask=torch.tensor([True, True, False]),
        candidate_mask=torch.tensor([True, True, False]),
        source_labels=torch.tensor([1.0, 0.0, 0.0]),
        source_count=torch.tensor(1),
    )


def test_shortlist_grid_reports_recall_and_rejects_missing_k_slices():
    model = JointSourceCountGCN(
        hidden_dim=8, dropout=0.0, shortlist_mode="preliminary"
    )

    summary = evaluate_shortlist_grid(
        model,
        [model_data(), model_data()],
        [3],
        bootstrap_repeats=100,
        bootstrap_seed=11,
    )

    result = summary["results"]["3"]
    assert result["micro_candidate_recall"] == 1.0
    assert result["candidate_recall_by_k"] == {"1": 1.0}
    assert not result["safety_passed"]
    assert summary["selected_size"] is None
    assert summary["decision"] == "reject_shortlist"
