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
    train_node_one_epoch,
    train_one_epoch,
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


def test_joint_training_supports_multi_graph_batch():
    model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_one_epoch(model, [model_data(), model_data()], optimizer, batch_size=2)
    metrics = evaluate_epoch(model, [model_data(), model_data()], batch_size=2)
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_node_training_supports_multi_graph_batch():
    model = NodeOnlyGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_node_one_epoch(model, [model_data(), model_data()], optimizer, batch_size=2)
    metrics = evaluate_node_epoch(model, [model_data(), model_data()], batch_size=2)
    assert 0.0 <= metrics.macro_f1 <= 1.0


def test_joint_training_resumes_from_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "last.pt"
    model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first = fit_joint_model(
        model,
        [model_data()],
        [model_data()],
        optimizer,
        max_epochs=1,
        patience=3,
        checkpoint_path=checkpoint,
    )
    resumed_model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed = fit_joint_model(
        resumed_model,
        [model_data()],
        [model_data()],
        resumed_optimizer,
        max_epochs=2,
        patience=3,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
    )

    assert first.stopped_epoch == 1
    assert resumed.stopped_epoch == 2
    assert len(resumed.train_history) == 2


def test_resume_restores_torch_random_state(tmp_path):
    checkpoint = tmp_path / "rng.pt"
    torch.manual_seed(123)
    model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fit_joint_model(
        model,
        [model_data()],
        [model_data()],
        optimizer,
        max_epochs=1,
        patience=3,
        checkpoint_path=checkpoint,
    )
    expected = torch.rand(3)
    torch.manual_seed(999)
    restored_model = JointSourceCountGCN(hidden_dim=8, dropout=0.0)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
    from diffusion_sources.training import load_last_checkpoint

    load_last_checkpoint(checkpoint, restored_model, restored_optimizer, "cpu")

    assert torch.equal(torch.rand(3), expected)


def test_node_training_resumes_from_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "node_last.pt"
    model = NodeOnlyGCN(hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fit_node_model(
        model,
        [model_data()],
        [model_data()],
        optimizer,
        max_epochs=1,
        patience=3,
        checkpoint_path=checkpoint,
    )
    resumed_model = NodeOnlyGCN(hidden_dim=8, dropout=0.0)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed = fit_node_model(
        resumed_model,
        [model_data()],
        [model_data()],
        resumed_optimizer,
        max_epochs=2,
        patience=3,
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
    )

    assert resumed.stopped_epoch == 2
    assert checkpoint.exists()
