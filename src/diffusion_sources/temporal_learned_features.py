"""Target-blind observable features and a portable, immutable linear scorer."""
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

FEATURE_NAMES = (
    'gcn_probability', 'early_observed', 'early_neighbor_fraction',
    'final_neighbor_fraction', 'log_degree_normalized', 'gcn_probability_x_early',
)


def _ids_probabilities(graph_size, candidate_ids, probabilities):
    if (not candidate_ids or any(isinstance(i, (bool, np.bool_)) or
            not isinstance(i, (int, np.integer)) or not 0 <= i < graph_size
            for i in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids)):
        raise ValueError('Candidate IDs must be unique valid integer node IDs')
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(candidate_ids),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Candidate probabilities must be aligned and finite in [0,1]')
    return p


def build_candidate_features(graph, candidate_ids, probabilities, early_mask, final_mask):
    n = graph.number_of_nodes()
    if graph.is_directed() or graph.is_multigraph() or set(graph.nodes) != set(range(n)):
        raise ValueError('Expected a simple undirected graph with contiguous node IDs')
    p = _ids_probabilities(n, candidate_ids, probabilities)
    masks = []
    for mask in (early_mask, final_mask):
        a = np.asarray(mask)
        if a.shape != (n,) or not np.isin(a, [0, 1]).all():
            raise ValueError('Masks must be binary and aligned with the full graph')
        masks.append(a.astype(np.float64))
    early, final = masks
    degree = np.array([graph.degree(i) for i in range(n)], dtype=np.float64)
    counts = np.zeros((n, 2), dtype=np.float64)
    edges = np.asarray(list(graph.edges), dtype=np.int64).reshape(-1, 2)
    if len(edges):
        u, v = edges.T
        for j, mask in enumerate(masks):
            np.add.at(counts[:, j], u, mask[v])
            np.add.at(counts[:, j], v, mask[u])
    fractions = np.divide(counts, degree[:, None], out=np.zeros_like(counts), where=degree[:, None] > 0)
    denom = np.log1p(degree.max(initial=0))
    normalized = np.log1p(degree) / denom if denom else np.zeros(n)
    ids = np.asarray(candidate_ids)
    return np.column_stack((p, early[ids], fractions[ids], normalized[ids], p * early[ids]))


@dataclass(frozen=True)
class LinearHead:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    C: float
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError('Unsupported linear-head schema or feature order')
        for name in ('feature_names', 'mean', 'scale', 'coefficients'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('mean', 'scale', 'coefficients'):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (6,) or not np.isfinite(a).all():
                raise ValueError(f'Invalid head {name}')
        if min(self.scale) <= 0 or not np.isfinite([self.intercept, self.C]).all() or self.C <= 0:
            raise ValueError('Head scale and C must be positive; parameters must be finite')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError('Unexpected linear-head fields')
        try:
            return cls(**payload)
        except (TypeError, OverflowError) as exc:
            raise ValueError('Malformed linear head') from exc

    def decision_function(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
            raise ValueError('Expected finite [candidates,6] features')
        return ((x - self.mean) / self.scale) @ np.asarray(self.coefficients) + self.intercept


def rank_candidates(candidate_ids, probabilities, features, predicted_count, head):
    if isinstance(predicted_count, (bool, np.bool_)) or not isinstance(predicted_count, (int, np.integer)) or not 1 <= predicted_count <= 3:
        raise ValueError('Predicted count must be an integer in 1..3')
    # IDs here are not constrained by a graph; validate their nonnegative integer domain.
    p = _ids_probabilities(max(candidate_ids, default=-1) + 1, candidate_ids, probabilities)
    scores = head.decision_function(features)
    if scores.shape != p.shape:
        raise ValueError('Feature rows must align with candidate IDs')
    order = sorted(range(len(candidate_ids)), key=lambda j: (-scores[j], -p[j], candidate_ids[j]))
    return frozenset(candidate_ids[j] for j in order[:min(predicted_count, len(order))])
