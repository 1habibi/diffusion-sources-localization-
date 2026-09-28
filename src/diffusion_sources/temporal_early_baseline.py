"""Expected early-only ranking: no learned scores and no arbitrary ID tie break."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import set_metrics
from .temporal_scoring import CandidateScores, PilotRecord, correct_sources
from .temporal_statistics import paired_bootstrap_ci

POLICY = 'early-first-uniform-ties-analytic-f1-v1'


def expected_early_f1(candidates: CandidateScores, true_sources: frozenset[int]) -> float:
    """Evaluate uniform selection within the early/late tie groups exactly.

    Prediction policy sees only candidates, early membership and the frozen
    predicted count. Targets enter solely to evaluate expected overlap.
    F1 = 2*overlap/(true count + predicted set size), whose denominator is fixed.
    """
    pool = set(candidates.candidate_ids)
    if not true_sources or not true_sources <= pool or len(true_sources) > 3:
        raise ValueError('Invalid true sources for early baseline.')
    early = {node for node, observed in zip(candidates.candidate_ids, candidates.early_observed) if observed}
    n, m = len(pool), len(early)
    predicted = min(candidates.predicted_count, n)
    early_slots = min(predicted, m)
    late_slots = predicted - early_slots
    overlap = early_slots * len(true_sources & early) / m if m else 0.
    if late_slots:
        overlap += late_slots * len(true_sources - early) / (n - m)
    return float(2 * overlap / (len(true_sources) + predicted))


def evaluate_early_baseline(records: Sequence[PilotRecord], beta: float) -> dict:
    if (not records or not np.isfinite(beta) or beta < 0
            or len({r.index for r in records}) != len(records)):
        raise ValueError('Requires unique paired records and finite nonnegative beta.')
    rows = []
    for r in records:
        early_f1 = expected_early_f1(r.candidates, r.true_sources)
        temporal = correct_sources(r.candidates, beta)
        snapshot = r.candidates.baseline_sources
        if len(temporal) != len(snapshot):
            raise ValueError(f'example {r.index}: count changed')
        snapshot_f1 = set_metrics(r.true_sources, snapshot)['f1']
        temporal_f1 = set_metrics(r.true_sources, temporal)['f1']
        rows.append({'index': r.index, 'k': len(r.true_sources),
                     'candidate_count': len(r.candidates.candidate_ids),
                     'early_count': sum(r.candidates.early_observed),
                     'predicted_count': r.candidates.predicted_count,
                     'predicted_set_size': len(snapshot),
                     'snapshot_f1': snapshot_f1, 'early_expected_f1': early_f1,
                     'temporal_f1': temporal_f1,
                     'temporal_minus_early': temporal_f1 - early_f1,
                     'snapshot_minus_early': snapshot_f1 - early_f1})

    fields = ('snapshot_f1', 'early_expected_f1', 'temporal_f1',
              'temporal_minus_early', 'snapshot_minus_early')

    def aggregate(subset):
        return {'n': len(subset), **{key: float(np.mean([row[key] for row in subset]))
                                    if subset else None for key in fields}}

    bins = {'1-10': (1, 10), '11-20': (11, 20), '21-50': (21, 50), '51+': (51, float('inf'))}
    return {'policy': POLICY, 'beta': beta, 'exploratory': True,
            'early_baseline_uses_gcn_count': True, 'count_unchanged': True,
            'tie_uncertainty': 'analytic expectation, not a sampled prediction set',
            'ci_scope': 'cascade bootstrap conditional on this graph/split/checkpoint; ties integrated out',
            'all': aggregate(rows),
            'by_k': {str(k): aggregate([row for row in rows if row['k'] == k]) for k in (1, 2, 3)},
            'by_candidates': {name: aggregate([row for row in rows if lo <= row['candidate_count'] <= hi])
                              for name, (lo, hi) in bins.items()},
            'f1_ci': list(paired_bootstrap_ci(np.array([row['temporal_minus_early'] for row in rows]),
                                             np.array([row['k'] for row in rows]))),
            'rows': rows}
