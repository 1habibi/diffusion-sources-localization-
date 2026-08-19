from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from diffusion_sources.models import GCNBackbone, JointSourceCountGCN, NodeOnlyGCN


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


def test_joint_global_context_model_supports_batched_graphs():
    first = model_data()
    second = model_data()
    first.global_features = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    second.global_features = torch.tensor([[0.6, 0.7, 0.8, 0.9]])
    batch = next(iter(DataLoader([first, second], batch_size=2)))
    model = JointSourceCountGCN(
        hidden_dim=8,
        dropout=0.0,
        source_head_mode="global_context",
        global_feature_dim=4,
    )

    source_logits, count_logits = model(batch)

    assert source_logits.shape == (6,)
    assert count_logits.shape == (2, 3)


def test_joint_global_context_requires_matching_graph_features():
    model = JointSourceCountGCN(
        hidden_dim=8,
        dropout=0.0,
        source_head_mode="global_context",
        global_feature_dim=4,
    )

    with pytest.raises(ValueError, match="global_features"):
        model(model_data())


def test_residual_three_layer_backbone_returns_hidden_embeddings():
    data = model_data()
    backbone = GCNBackbone(
        input_dim=2, hidden_dim=8, dropout=0.0, mode="residual_3"
    )

    hidden = backbone(data.x, data.edge_index)

    assert hidden.shape == (3, 8)
    assert backbone.conv3 is not None


def test_joint_global_context_supports_residual_three_layer_backbone():
    data = model_data()
    data.global_features = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    model = JointSourceCountGCN(
        hidden_dim=8,
        dropout=0.0,
        source_head_mode="global_context",
        global_feature_dim=4,
        backbone_mode="residual_3",
    )

    source_logits, count_logits = model(data)
    (source_logits.sum() + count_logits.sum()).backward()

    assert source_logits.shape == (3,)
    assert count_logits.shape == (1, 3)
    assert model.backbone.conv3.lin.weight.grad is not None


def test_backbone_rejects_unknown_mode():
    with pytest.raises(ValueError, match="backbone mode"):
        GCNBackbone(mode="unknown")


def test_joint_specialized_heads_return_one_logit_column_per_k():
    data = model_data()
    data.global_features = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    model = JointSourceCountGCN(
        hidden_dim=8,
        dropout=0.0,
        source_head_mode="global_context",
        global_feature_dim=4,
        backbone_mode="residual_3",
        source_head_strategy="specialized_k",
    )

    source_logits, count_logits = model(data)

    assert source_logits.shape == (3, 3)
    assert count_logits.shape == (1, 3)
    assert model.source_head is None
    assert len(model.source_heads) == 3


def test_joint_model_rejects_unknown_source_head_strategy():
    with pytest.raises(ValueError, match="source_head_strategy"):
        JointSourceCountGCN(source_head_strategy="unknown")


def test_preliminary_head_is_detached_from_shared_encoder():
    data = model_data()
    data.global_features = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    model = JointSourceCountGCN(
        hidden_dim=8,
        dropout=0.0,
        source_head_mode="global_context",
        global_feature_dim=4,
        shortlist_mode="preliminary",
    )

    _, _, preliminary_logits = model.forward_all(data)
    preliminary_logits.sum().backward()

    assert preliminary_logits.shape == (3,)
    assert model.preliminary_head.weight.grad is not None
    assert model.backbone.conv1.lin.weight.grad is None


def test_safe_shortlist_uses_top_m_candidates_per_graph():
    shortlist, fallback = JointSourceCountGCN._safe_shortlist_mask(
        torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 0.0]),
        torch.tensor([True, True, True, True, True, False]),
        torch.tensor([0, 0, 0, 1, 1, 1]),
        shortlist_size=2,
    )

    assert torch.equal(
        shortlist, torch.tensor([False, True, True, True, True, False])
    )
    assert fallback == (False, False)


def test_safe_shortlist_falls_back_to_full_candidates_on_invalid_scores():
    candidates = torch.tensor([True, True, False])
    shortlist, fallback = JointSourceCountGCN._safe_shortlist_mask(
        torch.tensor([float("nan"), 1.0, 0.0]),
        candidates,
        torch.tensor([0, 0, 0]),
        shortlist_size=1,
    )

    assert torch.equal(shortlist, candidates)
    assert fallback == (True,)


def test_forward_shortlisted_scores_only_selected_candidates():
    data = model_data()
    data.global_features = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
    model = JointSourceCountGCN(
        hidden_dim=8,
        dropout=0.0,
        source_head_mode="global_context",
        global_feature_dim=4,
        source_head_strategy="specialized_k",
        shortlist_mode="preliminary",
    )

    source_logits, count_logits, _, shortlist, fallback = (
        model.forward_shortlisted(data, shortlist_size=1)
    )

    assert source_logits.shape == (3, 3)
    assert count_logits.shape == (1, 3)
    assert shortlist.sum().item() == 1
    assert torch.isneginf(source_logits[~shortlist]).all()
    assert torch.isfinite(source_logits[shortlist]).all()
    assert fallback == (False,)
