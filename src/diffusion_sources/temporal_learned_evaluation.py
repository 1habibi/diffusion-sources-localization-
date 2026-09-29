"""Paired exploratory comparisons against v3, not against an easier baseline."""
import hashlib
import json

import numpy as np
from tqdm.auto import tqdm

from .metrics import set_metrics, source_radius_hits, source_set_distances
from .temporal_learned_data import validate_table
from .temporal_learned_features import rank_candidates
from .temporal_scoring import CandidateScores, correct_sources
from .temporal_statistics import _candidate_bin, paired_bootstrap_ci

BINS = ('1-10', '11-20', '21-50', '51+')
TOL = 1e-12


def evaluate_reranker(table, head, graph, *, include_ci):
    validate_table(table)
    rows = []
    count_ok = cardinality_ok = True
    for j in tqdm(range(len(table.indices)), desc='Learned-v3 paired metrics', unit='cascade'):
        a,b = table.offsets[j:j+2]
        ids = tuple(map(int,table.candidate_ids[a:b]))
        truth = frozenset(i for i,label in zip(ids,table.labels[a:b]) if label)
        k = int(table.predicted_counts[j])
        candidates = CandidateScores(ids,tuple(table.probabilities[a:b]),tuple(table.early_flags[a:b].astype(bool)),table.snapshot_sources[j],k)
        if correct_sources(candidates,0) != table.snapshot_sources[j]:
            raise ValueError('Beta-zero snapshot control failed')
        predicted = {'snapshot':table.snapshot_sources[j], 'v3':correct_sources(candidates,.5),
                     'learned':rank_candidates(ids,table.probabilities[a:b],table.features[a:b],k,head)}
        cardinality_ok &= all(len(s) == min(k,len(ids)) for s in predicted.values())
        metrics = {name:{**set_metrics(truth,s), **source_set_distances(graph,truth,s),
                         **source_radius_hits(graph,truth,s)} for name,s in predicted.items()}
        count_ok &= len({m['count_accuracy'] for m in metrics.values()}) == 1
        rows.append(dict(index=int(table.indices[j]), k=int(table.true_counts[j]),
            predicted_count=k, candidate_count=len(ids), candidate_ids=list(ids),
            true_sources=sorted(truth), early_empty=bool(table.early_empty[j]),
            **{f'{name}_sources':sorted(s) for name,s in predicted.items()}, **metrics))

    def aggregate(name, subset):
        return {'n':len(subset), 'status':'evaluated' if subset else 'not_applicable',
                **{key:float(np.mean([r[name][key] for r in subset])) if subset else None for key in rows[0][name]}}

    result = dict(exploratory=True, rows=rows, count_unchanged=bool(count_ok), cardinality_unchanged=bool(cardinality_ok),
        early_mask_hash=table.early_mask_hash,
        head_hash=hashlib.sha256(json.dumps(head.to_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest())
    for name in ('snapshot','v3','learned'):
        result[name] = {'all':aggregate(name,rows),
            'by_k':{str(k):aggregate(name,[r for r in rows if r['k']==k]) for k in (1,2,3)},
            'by_candidates':{b:aggregate(name,[r for r in rows if _candidate_bin(r['candidate_count'])==b]) for b in BINS}}
    result['delta_f1'] = result['learned']['all']['f1']-result['v3']['all']['f1']
    result['delta_exact'] = result['learned']['all']['exact_set_accuracy']-result['v3']['all']['exact_set_accuracy']
    for group in ('k','candidates'):
        result[f'delta_by_{group}'] = {key:result['learned'][f'by_{group}'][key]['f1']-base['f1'] if base['n'] else None
            for key,base in result['v3'][f'by_{group}'].items()}
    if include_ci:
        result['f1_ci'] = list(paired_bootstrap_ci(np.array([r['learned']['f1']-r['v3']['f1'] for r in rows]),np.array([r['k'] for r in rows])))
    return result


def dev_gate(report):
    reasons=[]
    for key,threshold in (('delta_f1',.02),('delta_exact',0)):
        value=report.get(key)
        if value is None or not np.isfinite(value) or value < threshold-TOL:
            reasons.append(f'{key} < {threshold}')
    for key in ('count_unchanged','cardinality_unchanged'):
        if report.get(key) is not True:
            reasons.append(f'{key} failed')
    for k in ('1','2','3'):
        d=report.get('delta_by_k',{}).get(k)
        if d is None or not np.isfinite(d) or d < -.02-TOL:
            reasons.append(f'k={k} missing or decline > 0.02')
    return {'passed':not reasons,'reasons':reasons}


def validation_gate(report):
    reasons=list(dev_gate(report)['reasons'])
    for b in BINS:
        if b not in report.get('delta_by_candidates',{}):
            reasons.append(f'candidate bin {b} missing')
            continue
        value=report['delta_by_candidates'][b]
        if value is not None and (not np.isfinite(value) or value < -.02-TOL):
            reasons.append(f'candidate bin {b} decline > 0.02')
    ci=report.get('f1_ci')
    if ci is None or len(ci)!=2 or not np.isfinite(ci).all() or ci[0] <= 0 or ci[1] < ci[0]:
        reasons.append('paired CI lower <= 0 or invalid')
    return {'passed':not reasons,'reasons':reasons}


def check_saved_snapshot_f1(report, run_metrics, *, tolerance=1e-6):
    try:
        observed=report['snapshot']['all']
        expected=run_metrics['validation_prediction_metrics']['joint_estimated_k']['all']['f1']
        if observed['n']!=1998 or not np.isfinite([observed['f1'],expected]).all() or abs(observed['f1']-expected)>tolerance:
            raise ValueError('Snapshot control does not match full saved validation F1')
    except (KeyError,TypeError) as exc:
        raise ValueError('Invalid saved snapshot metric schema') from exc


def summarize_repeats(reports):
    if len(reports)!=3 or {r.get('seed') for r in reports}!={7026,7027,7028}:
        raise ValueError('Exactly three distinct frozen seeds are required')
    reports=sorted(reports,key=lambda r:r['seed'])
    first=reports[0]
    if not first['rows']:
        raise ValueError('Empty confirmation reports')
    alignment = ('index','k','candidate_ids','true_sources','candidate_count','early_empty')
    for r in reports[1:]:
        if r['head_hash']!=first['head_hash'] or r['early_mask_hash']!=first['early_mask_hash'] or len(r['rows'])!=len(first['rows']):
            raise ValueError('Head or full early masks or case count changed')
        if any(any(a[key]!=b[key] for key in alignment) for a,b in zip(first['rows'],r['rows'])):
            raise ValueError('Case truth/candidate/observation alignment mismatch')
    if len({r['index'] for r in first['rows']}) != len(first['rows']):
        raise ValueError('Duplicate case indices')
    deltas=np.array([[row['learned']['f1']-row['v3']['f1'] for row in r['rows']] for r in reports])
    per_case=deltas.mean(axis=0)
    k=np.array([r['k'] for r in first['rows']])
    ci=list(paired_bootstrap_ci(per_case,k))
    names=('snapshot','v3','learned')
    means={name:{metric:float(np.mean([r[name]['all'][metric] for r in reports])) for metric in first[name]['all'] if metric not in ('n','status')} for name in names}
    sd={name:{metric:float(np.std([r[name]['all'][metric] for r in reports],ddof=1)) for metric in means[name]} for name in names}
    delta_by_k={str(value):float(per_case[k==value].mean()) if np.any(k==value) else None for value in (1,2,3)}
    size=np.array([_candidate_bin(row['candidate_count']) for row in first['rows']])
    delta_by_candidates={b:float(per_case[size==b].mean()) if np.any(size==b) else None for b in BINS}
    mean_delta=float(per_case.mean())
    exact=float(np.mean([r['delta_exact'] for r in reports]))
    count_ok=all(r['count_unchanged'] for r in reports)
    cardinality_ok=all(r['cardinality_unchanged'] for r in reports)
    gate_input=dict(delta_f1=mean_delta,delta_exact=exact,delta_by_k=delta_by_k,
        delta_by_candidates=delta_by_candidates,count_unchanged=count_ok,cardinality_unchanged=cardinality_ok,f1_ci=ci)
    gate=validation_gate(gate_input)
    if any(float(d.mean())<=0 for d in deltas):
        gate['reasons'].append('Non-positive delta on a frozen seed')
        gate['passed']=False
    return dict(exploratory=True,n_cascades=len(k),means=means,sample_sd=sd,mean_delta_f1=mean_delta,
        sample_sd_delta_f1=float(np.std(deltas.mean(axis=1),ddof=1)),delta_exact=exact,
        delta_by_k=delta_by_k,delta_by_candidates=delta_by_candidates,f1_ci=ci,gate=gate,
        head_hash=first['head_hash'],early_mask_hash=first['early_mask_hash'],
        per_seed={str(r['seed']):{'delta_f1':r['delta_f1'],'f1_ci':r['f1_ci'],
                                 **{name:r[name] for name in names}} for r in reports},
        interpretation='Exploratory validation; one fixed reranker, shared frozen count; not 5994 independent cases.')
