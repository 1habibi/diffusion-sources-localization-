"""GCN models for multi-source localization and source-count estimation."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCNBackbone(nn.Module):
    """Two-layer node encoder shared by source-localization models."""

    def __init__(
        self, input_dim: int = 2, hidden_dim: int = 64, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x, edge_index).relu()
        hidden = self.dropout(hidden)
        return self.conv2(hidden, edge_index).relu()


class NodeOnlyGCN(nn.Module):
    """Multi-label GCN baseline with one source logit per node."""

    def __init__(
        self, input_dim: int = 2, hidden_dim: int = 64, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.backbone = GCNBackbone(input_dim, hidden_dim, dropout)
        self.source_head = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        hidden = self.backbone(data.x, data.edge_index)
        return self.source_head(hidden).squeeze(-1)


class JointSourceCountGCN(nn.Module):
    """Joint node localization and graph-level source-count classifier."""

    def __init__(
        self, input_dim: int = 2, hidden_dim: int = 64, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.backbone = GCNBackbone(input_dim, hidden_dim, dropout)
        self.source_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(data.x, data.edge_index)
        source_logits = self.source_head(hidden).squeeze(-1)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)

        all_pool = torch.cat(
            [global_mean_pool(hidden, batch), global_max_pool(hidden, batch)], dim=-1
        )
        observed_mask = data.observed_mask.bool()
        if not observed_mask.any():
            raise ValueError("Every graph must contain an observed infected node.")
        observed_pool = torch.cat(
            [
                global_mean_pool(hidden[observed_mask], batch[observed_mask]),
                global_max_pool(hidden[observed_mask], batch[observed_mask]),
            ],
            dim=-1,
        )
        if observed_pool.size(0) != all_pool.size(0):
            raise ValueError("Every graph in a batch must contain an observed node.")
        return source_logits, self.count_head(torch.cat([all_pool, observed_pool], dim=-1))
