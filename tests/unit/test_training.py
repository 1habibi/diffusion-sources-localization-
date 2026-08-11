from __future__ import annotations

import json

import torch
from torch_geometric.data import Data

from diffusion_sources.models import JointSourceCountGCN, NodeOnlyGCN
from diffusion_sources.training import (
    evaluate_epoch,
    evaluate_node_epoch,
    fit_joint_model,
    fit_node_model,
    save_training_result,
)


def model_data() -> Data:
    return Data(
        x=torch.tensor([[1.0, 0.5], [1.0, 1.0], [0.0, 0.5]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        observed_mask=torch.tensor([True, True, False]),
        candidate_mask=torch.tensor([True, True, False]),
        source_labels=torch.tensor([1.0, 0.0, 0.0]),
        source_count=torch.tensor(1),
    )


def test_evaluate_epoch_returns_report_metrics():
    model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    metrics = evaluate_epoch(model, [model_data()])

    assert metrics.loss >= 0.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert metrics.count_mae >= 0.0


def test_fit_and_save_training_result(tmp_path):
    torch.manual_seed(4)
    model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    result = fit_joint_model(
        model,
        [model_data()],
        [model_data()],
        optimizer,
        max_epochs=3,
        patience=2,
    )
    save_training_result(result, tmp_path)

    assert 1 <= result.best_epoch <= result.stopped_epoch
    assert len(result.train_history) == result.stopped_epoch
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "best_model.pt").exists()
    summary = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert summary["best_epoch"] == result.best_epoch


def test_fit_node_model_tracks_compatible_history():
    model = NodeOnlyGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    result = fit_node_model(
        model, [model_data()], [model_data()], optimizer, max_epochs=2, patience=2
    )
    metrics = evaluate_node_epoch(model, [model_data()])

    assert result.best_epoch >= 1
    assert metrics.count_loss == 0.0
    assert metrics.count_accuracy == 1.0
