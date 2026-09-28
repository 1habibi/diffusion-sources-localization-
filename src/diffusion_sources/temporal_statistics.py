"""Exploratory paired evaluation, with fixed bootstrap and budget gates."""
from __future__ import annotations
from collections.abc import Sequence

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from .metrics import set_metrics, source_radius_hits, source_set_distances
from .temporal_scoring import PilotRecord, correct_sources


def paired_bootstrap_ci(deltas: np.ndarray, true_k: np.ndarray,
                         *, repetitions: int = 2000, seed: int = 9282026) -> tuple[float, float]:
    deltas, true_k = np.asarray(deltas), np.asarray(true_k)
    if (deltas.ndim != 1 or not len(deltas) or deltas.shape != true_k.shape
            or not np.isfinite(deltas).all() or not np.isin(true_k, [1, 2, 3]).all()
            or repetitions < 1):
        raise ValueError('Invalid paired bootstrap input.')
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(true_k == k) for k in sorted(set(true_k.tolist()))]
    means = [deltas[np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])].mean()
             for _ in range(repetitions)]
    lower, upper = np.quantile(means, [.025, .975])
    return float(lower), float(upper)


def _candidate_bin(n: int) -> str:
    return '1-10' if n <= 10 else '11-20' if n <= 20 else '21-50' if n <= 50 else '51+'


def evaluate_pairs(records: Sequence[PilotRecord], beta: float, graph: nx.Graph) -> dict:
    if not records:
        raise ValueError('No paired records.')
    rows = []
    hit_sources = all_sources = total_sources = 0
    for r in tqdm(records, desc='Paired geometry + metrics', unit='cascade'):
        if not 1 <= len(r.true_sources) <= 3 or not r.true_sources <= set(r.candidates.candidate_ids):
            raise ValueError(f'example {r.index}: invalid source targets')
        pred = correct_sources(r.candidates, beta)
        metrics = {}
        for label, nodes in [('baseline', r.candidates.baseline_sources), ('temporal', pred)]:
            metrics[label] = {**set_metrics(r.true_sources, nodes),
                             **source_set_distances(graph, r.true_sources, nodes),
                             **source_radius_hits(graph, r.true_sources, nodes)}
        rows.append({'index': r.index, 'k': len(r.true_sources),
                     'candidate_count': len(r.candidates.candidate_ids),
                     'true_sources': sorted(r.true_sources),
                     'baseline_sources': sorted(r.candidates.baseline_sources),
                     'temporal_sources': sorted(pred), **metrics})
        early_ids = {n for n, early in zip(r.candidates.candidate_ids, r.candidates.early_observed) if early}
        hits = len(r.true_sources & early_ids)
        hit_sources += hits
        total_sources += len(r.true_sources)
        all_sources += int(hits == len(r.true_sources))

    def aggregate(label, subset):
        keys = rows[0][label]
        return {'n': len(subset), **{key: float(np.mean([row[label][key] for row in subset]))
                                    if subset else None for key in keys}}

    report = {'beta': beta, 'exploratory': True, 'rows': rows}
    for label in ('baseline', 'temporal'):
        report[label] = {'all': aggregate(label, rows),
            'by_k': {str(k): aggregate(label, [row for row in rows if row['k'] == k]) for k in (1, 2, 3)},
            'by_candidates': {b: aggregate(label, [row for row in rows if _candidate_bin(row['candidate_count']) == b])
                              for b in ('1-10', '11-20', '21-50', '51+')}}
    report['delta_f1'] = report['temporal']['all']['f1'] - report['baseline']['all']['f1']
    report['delta_by_k'] = {str(k): (report['temporal']['by_k'][str(k)]['f1'] - report['baseline']['by_k'][str(k)]['f1'])
                            if report['baseline']['by_k'][str(k)]['n'] else None for k in (1, 2, 3)}
    report['count_unchanged'] = all(row['baseline']['count_accuracy'] == row['temporal']['count_accuracy'] for row in rows)
    if not report['count_unchanged']:
        raise ValueError('Temporal correction changed source count.')
    report['f1_ci'] = list(paired_bootstrap_ci(
        np.array([row['temporal']['f1'] - row['baseline']['f1'] for row in rows]),
        np.array([row['k'] for row in rows])))
    report['early_coverage'] = {'source_recall': hit_sources / total_sources,
        'all_source_fraction': all_sources / len(records),
        'empty_fraction': sum(r.early_empty for r in records) / len(records)}
    return report


def primary_gate(report: dict) -> dict:
    reasons = []
    if report['delta_f1'] < .02:
        reasons.append('delta F1 < 0.02')
    if report['f1_ci'][0] <= 0:
        reasons.append('paired CI lower <= 0')
    for k in ('2', '3'):
        delta = report['delta_by_k'].get(k)
        if delta is None or delta < -.02:
            reasons.append(f'k={k} missing or decline > 0.02')
    if not report['count_unchanged']:
        reasons.append('count changed')
    return {'passed': not reasons, 'reasons': reasons}


def confirmation_gate(primary: dict, repeats: Sequence[dict]) -> dict:
    reasons = []
    if not primary_gate(primary)['passed']:
        reasons.append('primary gate failed')
    if len(repeats) != 2:
        reasons.append('two frozen repeats required')
    deltas = [r['delta_f1'] for r in [primary, *repeats]]
    if any(delta <= 0 for delta in deltas):
        reasons.append('non-positive delta on a seed')
    if np.mean(deltas) < .02:
        reasons.append('mean delta < 0.02')
    return {'passed': not reasons, 'reasons': reasons, 'mean_delta_f1': float(np.mean(deltas))}
