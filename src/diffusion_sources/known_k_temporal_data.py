"""Source-blind two-observation inputs for the known-cardinality GCN."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from .dataset import graph_to_edge_index
from .features import SnapshotFeatureBuilder
from .temporal_replay import replay_early_mask


def _binary_mask(values: np.ndarray, n: int, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.shape != (n,) or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{name} must be a binary node mask")
    return mask.astype(bool, copy=False)


def make_observation(
    final_features: np.ndarray,
    early_mask: np.ndarray,
    candidate_mask: np.ndarray,
    k: int,
    edge_index: torch.Tensor,
) -> Data:
    """Build inference-only features; targets are deliberately not accepted."""
    final = np.asarray(final_features)
    if final.ndim != 2 or final.shape[1] != 5 or not np.isfinite(final).all():
        raise ValueError("final_features must have 5 finite columns")
    n = len(final)
    early = _binary_mask(early_mask, n, "early_mask")
    candidates = _binary_mask(candidate_mask, n, "candidate_mask")
    observed = _binary_mask(final[:, 0], n, "observed_mask")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k not in (1, 2, 3):
        raise ValueError("k must be an integer in 1..3")
    if int(candidates.sum()) < k:
        raise ValueError("candidate set is smaller than k")
    if (edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long
            or (edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n))):
        raise ValueError("edge_index does not match node count")
    cardinality = np.zeros((n, 3), dtype=np.float32)
    cardinality[:, int(k) - 1] = 1.0
    features = np.concatenate((final.astype(np.float32), early[:, None].astype(np.float32), cardinality), axis=1)
    return Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        candidate_mask=torch.from_numpy(candidates.copy()),
        observed_mask=torch.from_numpy(observed.copy()),
        early_observed_mask=torch.from_numpy(early.copy()),
    )


def load_known_k_split(
    data_dir: Path,
    split: Literal["train", "validation"],
    graph: nx.Graph,
    builder: SnapshotFeatureBuilder,
    feature_names: Sequence[str],
    indices: Sequence[int],
) -> list[Data]:
    """Replay checked early observations and attach targets only to train/eval cases."""
    if split not in ("train", "validation"):
        raise ValueError("Only train and validation are available to the pilot")
    if len(feature_names) != 5:
        raise ValueError("Known-k pilot requires exactly five final snapshot features")
    data_dir = Path(data_dir)
    config = yaml.safe_load((data_dir / "config.yaml").read_text(encoding="utf-8"))
    with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    edge_index = graph_to_edge_index(graph)
    cases = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("example indices must be integers")
        index = int(index)
        early = replay_early_mask(graph, config, arrays, index)
        candidates = _binary_mask(arrays["candidate_masks"][index], graph.number_of_nodes(), "candidate_mask")
        labels = _binary_mask(arrays["source_labels"][index], graph.number_of_nodes(), "source_labels")
        k = int(arrays["source_counts"][index])
        if labels.sum() != k or np.any(labels & ~candidates):
            raise ValueError(f"example {index}: invalid source labels or candidate mask")
        final = builder.build(
            arrays["features"][index, :, 0].astype(bool),
            list(feature_names),
            base_features=arrays["features"][index],
            candidate_mask=candidates,
        )
        case = make_observation(final, early, candidates, k, edge_index)
        case.source_labels = torch.from_numpy(labels.astype(np.float32))
        case.source_count = torch.tensor(k, dtype=torch.long)
        case.example_index = index
        cases.append(case)
    return cases
