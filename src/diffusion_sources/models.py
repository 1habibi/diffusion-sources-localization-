"""GCN models for multi-source localization and source-count estimation."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCNBackbone(nn.Module):
    """Configurable node encoder shared by source-localization models."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        *,
        mode: str = "plain_2",
    ) -> None:
        super().__init__()
        if mode not in {"plain_2", "residual_3"}:
            raise ValueError("backbone mode must be 'plain_2' or 'residual_3'.")
        self.mode = mode
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim) if mode == "residual_3" else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x, edge_index).relu()
        hidden = self.dropout(hidden)
        if self.mode == "plain_2":
            return self.conv2(hidden, edge_index).relu()
        hidden = (self.conv2(hidden, edge_index) + hidden).relu()
        hidden = self.dropout(hidden)
        return (self.conv3(hidden, edge_index) + hidden).relu()


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
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        *,
        source_head_mode: str = "local",
        global_feature_dim: int = 0,
        backbone_mode: str = "plain_2",
        source_head_strategy: str = "shared",
        shortlist_mode: str = "disabled",
    ) -> None:
        super().__init__()
        if source_head_mode not in {"local", "global_context"}:
            raise ValueError("source_head_mode must be 'local' or 'global_context'.")
        if global_feature_dim < 0:
            raise ValueError("global_feature_dim must be non-negative.")
        if source_head_mode == "local" and global_feature_dim:
            raise ValueError("global_feature_dim is only used by global_context mode.")
        if source_head_strategy not in {"shared", "specialized_k"}:
            raise ValueError(
                "source_head_strategy must be 'shared' or 'specialized_k'."
            )
        if shortlist_mode not in {"disabled", "preliminary"}:
            raise ValueError(
                "shortlist_mode must be 'disabled' or 'preliminary'."
            )
        self.source_head_mode = source_head_mode
        self.source_head_strategy = source_head_strategy
        self.global_feature_dim = global_feature_dim
        self.shortlist_mode = shortlist_mode
        self.backbone_mode = backbone_mode
        self.backbone = GCNBackbone(
            input_dim, hidden_dim, dropout, mode=backbone_mode
        )
        source_input_dim = (
            hidden_dim
            if source_head_mode == "local"
            else hidden_dim * 5 + global_feature_dim
        )

        def make_source_head() -> nn.Module:
            if source_head_mode == "local":
                return nn.Linear(source_input_dim, 1)
            return nn.Sequential(
                nn.Linear(hidden_dim * 5 + global_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        if source_head_strategy == "shared":
            self.source_head = make_source_head()
            self.source_heads = None
        else:
            self.source_head = None
            self.source_heads = nn.ModuleList(
                [make_source_head() for _ in range(3)]
            )
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.preliminary_head = (
            nn.Linear(source_input_dim, 1)
            if shortlist_mode == "preliminary"
            else None
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        source_input, count_input, _ = self._encode_inputs(data)
        return self._score_source_inputs(source_input), self.count_head(count_input)

    def forward_all(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Score all nodes and optionally return preliminary shortlist logits."""
        source_input, count_input, _ = self._encode_inputs(data)
        source_logits = self._score_source_inputs(source_input)
        preliminary_logits = (
            self.preliminary_head(source_input.detach()).squeeze(-1)
            if self.preliminary_head is not None
            else None
        )
        return source_logits, self.count_head(count_input), preliminary_logits

    def forward_shortlisted(
        self, data, shortlist_size: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[bool, ...],
    ]:
        """Score only a safe preliminary top-M subset with per-graph fallback."""
        if self.preliminary_head is None:
            raise ValueError("Shortlist inference requires a preliminary head.")
        source_input, count_input, batch = self._encode_inputs(data)
        preliminary_logits = self.preliminary_head(source_input.detach()).squeeze(-1)
        shortlist_mask, fallback_flags = self._safe_shortlist_mask(
            preliminary_logits,
            data.candidate_mask.bool(),
            batch,
            shortlist_size,
        )
        output_shape = (
            (source_input.size(0),)
            if self.source_head_strategy == "shared"
            else (source_input.size(0), 3)
        )
        source_logits = source_input.new_full(output_shape, float("-inf"))
        source_logits[shortlist_mask] = self._score_source_inputs(
            source_input[shortlist_mask]
        )
        return (
            source_logits,
            self.count_head(count_input),
            preliminary_logits,
            shortlist_mask,
            fallback_flags,
        )

    def _encode_inputs(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(data.x, data.edge_index)
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
        if self.source_head_mode == "local":
            source_input = hidden
        else:
            candidate_mask = data.candidate_mask.bool()
            if not candidate_mask.any():
                raise ValueError("Every graph must contain a candidate node.")
            candidate_pool = torch.cat(
                [
                    global_mean_pool(hidden[candidate_mask], batch[candidate_mask]),
                    global_max_pool(hidden[candidate_mask], batch[candidate_mask]),
                ],
                dim=-1,
            )
            if candidate_pool.size(0) != all_pool.size(0):
                raise ValueError("Every graph in a batch must contain a candidate node.")
            global_features = getattr(data, "global_features", None)
            if global_features is None:
                raise ValueError("global_context mode requires data.global_features.")
            global_features = global_features.reshape(all_pool.size(0), -1)
            if global_features.size(1) != self.global_feature_dim:
                raise ValueError("data.global_features has an unexpected dimension.")
            graph_context = torch.cat(
                [observed_pool, candidate_pool, global_features], dim=-1
            )
            source_input = torch.cat([hidden, graph_context[batch]], dim=-1)
        return source_input, torch.cat([all_pool, observed_pool], dim=-1), batch

    def _score_source_inputs(self, source_input: torch.Tensor) -> torch.Tensor:
        if self.source_head_strategy == "shared":
            return self.source_head(source_input).squeeze(-1)
        return torch.cat(
            [head(source_input) for head in self.source_heads], dim=-1
        )

    @staticmethod
    def _safe_shortlist_mask(
        preliminary_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        batch: torch.Tensor,
        shortlist_size: int,
    ) -> tuple[torch.Tensor, tuple[bool, ...]]:
        if shortlist_size < 1:
            raise ValueError("shortlist_size must be positive.")
        shortlist = torch.zeros_like(candidate_mask, dtype=torch.bool)
        fallback_flags: list[bool] = []
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        for graph_index in range(graph_count):
            graph_candidates = candidate_mask & (batch == graph_index)
            candidate_nodes = torch.nonzero(
                graph_candidates, as_tuple=False
            ).flatten()
            fallback = False
            try:
                if candidate_nodes.numel() == 0:
                    raise ValueError("candidate set is empty")
                candidate_scores = preliminary_logits[candidate_nodes]
                if not torch.isfinite(candidate_scores).all():
                    raise ValueError("preliminary scores are not finite")
                if candidate_nodes.numel() <= shortlist_size:
                    selected = candidate_nodes
                else:
                    selected = candidate_nodes[
                        torch.topk(candidate_scores, shortlist_size).indices
                    ]
                if selected.numel() == 0:
                    raise ValueError("shortlist is empty")
            except (RuntimeError, ValueError):
                selected = candidate_nodes
                fallback = True
            shortlist[selected] = True
            fallback_flags.append(fallback)
        return shortlist, tuple(fallback_flags)
