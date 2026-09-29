from copy import deepcopy

import numpy as np
import pytest

from diffusion_sources.temporal_learned_evaluation import (
    evaluate_reranker, dev_gate, validation_gate, summarize_repeats, check_saved_snapshot_f1,
)
from diffusion_sources.temporal_learned_features import FEATURE_NAMES, LinearHead
from tests.fixtures.temporal_learned import tiny_graph, tiny_table


def test_manual_metrics_and_control():
    h = LinearHead(FEATURE_NAMES, (0.,)*6, (1.,)*6, (1.,0.,0.,0.,0.,0.), 0., 1.)
    report = evaluate_reranker(tiny_table(), h, tiny_graph(), include_ci=True)
    assert report['learned']['all']['f1'] == 1
    assert report['snapshot']['all']['f1'] == 1
    assert report['count_unchanged']
    assert len(report['rows']) == 6
    assert set(report['learned']['all']) >= {'precision','recall','exact_set_accuracy','symmetric_set_distance','hit_at_1_hop','hit_at_2_hop'}
    assert 'not_applicable' == report['learned']['by_candidates']['51+']['status']


def gate_report():
    return {'delta_f1': .02-1e-13, 'delta_exact':0., 'count_unchanged':True, 'cardinality_unchanged':True,
            'delta_by_k':{str(k):0. for k in (1,2,3)},
            'delta_by_candidates':{'1-10':.02,'11-20':None,'21-50':0.,'51+':0.},
            'f1_ci':[.001,.04]}


def test_gate_threshold_and_exact_zero():
    r = gate_report()
    assert dev_gate(r)['passed'] and validation_gate(r)['passed']
    for key,value in [('delta_f1',.019),('delta_exact',-.001),('count_unchanged',False),('cardinality_unchanged',False)]:
        changed = deepcopy(r); changed[key]=value
        assert not dev_gate(changed)['passed']


def test_missing_k_empty_bins_and_nonpositive_ci():
    r = gate_report()
    r['delta_by_k']['1'] = None
    assert not dev_gate(r)['passed']
    r = gate_report(); r['delta_by_candidates']['51+'] = -.021
    assert not validation_gate(r)['passed']
    r = gate_report(); r['f1_ci'][0] = 0
    assert not validation_gate(r)['passed']


def test_saved_snapshot_control():
    r = {'snapshot': {'all': {'n':1998, 'f1':.36}}}
    metrics = {'validation_prediction_metrics': {'joint_estimated_k': {'all': {'f1':.36}}}}
    check_saved_snapshot_f1(r, metrics)
    with pytest.raises(ValueError):
        check_saved_snapshot_f1(r, {'validation_prediction_metrics': {'joint_estimated_k': {'all': {'f1':.4}}}})
    r['snapshot']['all']['n']=6
    with pytest.raises(ValueError):
        check_saved_snapshot_f1(r, metrics)


def reports():
    h = LinearHead(FEATURE_NAMES,(0.,)*6,(1.,)*6,(1.,0.,0.,0.,0.,0.),0.,1.)
    result=[]
    for seed in (7026,7027,7028):
        r = evaluate_reranker(tiny_table(),h,tiny_graph(),include_ci=True)
        r['seed']=seed
        result.append(r)
    return result


def test_summary_bootstraps_cascades_not_seed_rows(monkeypatch):
    observed=[]
    def bootstrap(delta,k,**kw):
        observed.append(len(delta))
        return (0.,0.)
    monkeypatch.setattr('diffusion_sources.temporal_learned_evaluation.paired_bootstrap_ci',bootstrap)
    r = summarize_repeats(reports())
    assert observed[-1] == 6
    assert r['n_cascades'] == 6
    assert not r['gate']['passed']


@pytest.mark.parametrize('field', ['seed','early_mask_hash','head_hash','truth','candidate_ids'])
def test_summary_rejects_unaligned_or_duplicate_seeds(field):
    rs = reports()
    if field == 'seed': rs[1]['seed']=7026
    elif field in ('early_mask_hash','head_hash'): rs[1][field]='different'
    elif field == 'truth': rs[1]['rows'][0]['true_sources']=[5]
    else: rs[1]['rows'][0]['candidate_ids']=[0,1,2,3,4]
    with pytest.raises(ValueError):
        summarize_repeats(rs)
