"""Independent paired summary: average seeds per cascade BEFORE bootstrap."""
from __future__ import annotations
import json
import numpy as np
from .temporal_statistics import paired_bootstrap_ci
from .temporal_independent_artifacts import IndependentPaths, SEEDS, verify_opened
from .temporal_independent_inference import seed_identity
from .temporal_pilot_artifacts import read_stage, write_stage, sha256_file

F1_FIELDS = ('snapshot_f1', 'early_expected_f1', 'temporal_f1')
PROPORTIONS = ('precision', 'recall', 'f1', 'exact_set_accuracy',
               'count_accuracy', 'hit_at_1_hop', 'hit_at_2_hop')
METRICS = (*PROPORTIONS, 'count_mae', 'symmetric_set_distance')


def _finite(value, low=0., high=float('inf')):
    return isinstance(value, (int, float)) and np.isfinite(value) and low <= value <= high


def summarize_reports(reports: list[dict]) -> dict:
    if len(reports) != 3 or {r['seed'] for r in reports} != set(SEEDS):
        raise ValueError('Exactly three distinct frozen seeds required.')
    by_seed = {r['seed']: r for r in reports}
    ordered = [by_seed[s] for s in SEEDS]
    base = ordered[0]
    count_ok = True
    for report in ordered:
        if (report['beta'] != .5 or report['n'] != 1998 or len(report['rows']) != 1998
                or any(not report[key] or report[key] != base[key]
                       for key in ('dataset_hash', 'freeze_hash', 'early_mask_hash'))):
            raise ValueError('Unpaired report protocol, hashes or size.')
        count_ok &= report['count_unchanged'] is True
        for i, row in enumerate(report['rows']):
            if (row['index'] != i or row['k'] not in (1, 2, 3)
                    or not isinstance(row['candidate_count'], int) or row['candidate_count'] < row['k']
                    or len(row['true_sources']) != row['k'] or len(set(row['true_sources'])) != row['k']
                    or any(not isinstance(n, int) or n < 0 for n in row['true_sources'])
                    or any(row[key] != base['rows'][i][key]
                           for key in ('index', 'k', 'candidate_count', 'true_sources'))):
                raise ValueError('Unpaired or malformed cascade row.')
            if any(not _finite(row[key], 0., 1.) for key in F1_FIELDS):
                raise ValueError('Nonfinite/out-of-range F1.')
            for field, control in [('temporal_minus_snapshot', 'snapshot_f1'),
                                   ('temporal_minus_early', 'early_expected_f1')]:
                if (not _finite(row[field], -1., 1.) or
                        abs(row[field] - (row['temporal_f1'] - row[control])) > 1e-12):
                    raise ValueError('Reported delta disagrees with metrics.')
            for label in ('snapshot', 'temporal'):
                metrics = row[label]
                if (any(key not in metrics for key in METRICS)
                        or any(not _finite(v, 0., 1. if key in PROPORTIONS else float('inf'))
                               for key, v in metrics.items())
                        or abs(metrics['f1'] - row[f'{label}_f1']) > 1e-12):
                    raise ValueError('Malformed per-cascade metrics.')
            count_ok &= all(row['snapshot'][key] == row['temporal'][key]
                            for key in ('count_accuracy', 'count_mae'))
            if 'snapshot_sources' in row:
                count_ok &= len(row['snapshot_sources']) == len(row['temporal_sources'])
                count_ok &= len(row['snapshot_sources']) == row['predicted_set_size']
    k = np.array([r['k'] for r in base['rows']])
    if {n: int((k == n).sum()) for n in (1, 2, 3)} != {1: 666, 2: 666, 3: 666}:
        raise ValueError('Independent source-count strata must each contain 666 cascades.')
    values = {key: np.array([[r[key] for r in report['rows']] for report in ordered])
              for key in (*F1_FIELDS, 'temporal_minus_snapshot', 'temporal_minus_early')}
    per_seed = {}
    for j, seed in enumerate(SEEDS):
        per_seed[str(seed)] = {
            'means': {key: float(matrix[j].mean()) for key, matrix in values.items()},
            'snapshot_ci': list(paired_bootstrap_ci(values['temporal_minus_snapshot'][j], k)),
            'early_ci': list(paired_bootstrap_ci(values['temporal_minus_early'][j], k))}
    means = {key: float(matrix.mean(axis=1).mean()) for key, matrix in values.items()}
    sample_sd = {key: float(matrix.mean(axis=1).std(ddof=1)) for key, matrix in values.items()}

    def comparison(field):
        delta = values[field].mean(axis=0)
        return {'delta_f1': float(delta.mean()), 'ci': list(paired_bootstrap_ci(delta, k)),
                'per_seed_delta': {str(s): float(values[field][j].mean()) for j, s in enumerate(SEEDS)}}

    primary, secondary = comparison('temporal_minus_snapshot'), comparison('temporal_minus_early')
    primary_reasons = []
    if primary['delta_f1'] < .02: primary_reasons.append('mean delta F1 < 0.02')
    if primary['ci'][0] <= 0: primary_reasons.append('aggregate paired CI lower <= 0')
    if any(v <= 0 for v in primary['per_seed_delta'].values()): primary_reasons.append('non-positive delta on a seed')
    primary['delta_by_k'] = {str(n): float(values['temporal_minus_snapshot'][:, k == n].mean()) for n in (1, 2, 3)}
    for n in (2, 3):
        if primary['delta_by_k'][str(n)] < -.02: primary_reasons.append(f'k={n} decline > 0.02')
    if not count_ok: primary_reasons.append('count invariant violated')
    primary.update(passed=not primary_reasons, reasons=primary_reasons)
    secondary_reasons = []
    if not primary['passed']: secondary_reasons.append('primary gate failed')
    if secondary['ci'][0] <= 0: secondary_reasons.append('aggregate paired CI lower <= 0')
    if any(v <= 0 for v in secondary['per_seed_delta'].values()): secondary_reasons.append('non-positive delta on a seed')
    secondary.update(passed=not secondary_reasons, reasons=secondary_reasons)

    metric_summary = {}
    for label in ('snapshot', 'temporal'):
        keys = set(base['rows'][0][label])
        if any(set(r[label]) != keys for report in ordered for r in report['rows']):
            raise ValueError('Metric columns differ across cascades/seeds.')
        per_model = {key: [float(np.mean([r[label][key] for r in report['rows']]))
                           for report in ordered] for key in sorted(keys)}
        metric_summary[label] = {'mean': {key: float(np.mean(v)) for key, v in per_model.items()},
                                 'sample_sd': {key: float(np.std(v, ddof=1)) for key, v in per_model.items()}}

    def group(mask):
        return {'n_per_seed': int(mask.sum()),
                **{key: float(matrix[:, mask].mean()) if mask.any() else None for key, matrix in values.items()}}
    size = np.array([r['candidate_count'] for r in base['rows']])
    return {'evaluation_role': 'independent_confirmation', 'exploratory': False,
        'bootstrap_n': 1998, 'bootstrap': {'repetitions': 2000, 'seed': 9282026, 'strata': 'true_k',
                                         'aggregation': 'mean three seed deltas per cascade before resampling'},
        'beta': .5, 'seeds': list(SEEDS), 'count_unchanged': bool(count_ok),
        'means': means, 'sample_sd': sample_sd, 'primary': primary, 'secondary': secondary,
        'metrics': metric_summary, 'per_seed': per_seed,
        'by_k': {str(n): group(k == n) for n in (1, 2, 3)},
        'by_candidates': {'1-10': group(size <= 10), '11-20': group((size > 10) & (size <= 20)),
                          '21-50': group((size > 20) & (size <= 50)), '51+': group(size > 50)},
        'early_coverage': {str(r['seed']): r.get('early_coverage') for r in ordered},
        'dataset_hash': base['dataset_hash'], 'freeze_hash': base['freeze_hash'],
        'early_mask_hash': base['early_mask_hash'], 'early_baseline_uses_gcn_count': True,
        'interpretation': 'Independent new IC cascades on the same known Facebook topology; no transfer claim. '
                          'CI conditional on graph/protocol/three frozen checkpoints; subgroup tables descriptive. '
                          'Early-only uses shared learned count, ties analytically integrated; no early geometry. '
                          'Negative results retained without retuning; no best-seed selection.'}


def save_summary(paths: IndependentPaths) -> dict:
    verify_opened(paths)
    reports, hashes = [], {}
    for seed in SEEDS:
        stage = f'seed_{seed}'
        reports.append(read_stage(paths.reports, stage, seed_identity(paths, seed))[1])
        hashes[str(seed)] = sha256_file(paths.reports / stage / 'payload.json')
    identity = {'opened': sha256_file(paths.reports / 'opened/payload.json'), 'seed_payloads': hashes}
    if (paths.reports / 'summary').exists():
        result = read_stage(paths.reports, 'summary', identity)[1]
    else:
        result = summarize_reports(reports)
        verify_opened(paths)
        if any(sha256_file(paths.reports / f'seed_{seed}/payload.json') != hashes[str(seed)] for seed in SEEDS):
            raise ValueError('Seed reports changed during summary.')
        write_stage(paths.reports, 'summary', {'identity': identity}, result)
    print(json.dumps(result, indent=2), flush=True)
    print(f'Summary: {paths.reports / "summary/payload.json"}', flush=True)
    return result
