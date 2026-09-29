"""Three bounded CPU fits, fit-only scaling, and selection without refitting."""
import importlib.metadata
import inspect
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .temporal_learned_collect import Budget
from .temporal_learned_data import validate_table
from .temporal_learned_evaluation import dev_gate, evaluate_reranker
from .temporal_learned_features import FEATURE_NAMES, LinearHead, rank_candidates

C_GRID = (.1,1.,10.)


def cascade_class_weights(table):
    validate_table(table)
    weights=np.empty(len(table.labels),dtype=np.float64)
    for a,b,k in zip(table.offsets[:-1],table.offsets[1:],table.true_counts):
        labels=table.labels[a:b].astype(bool)
        weights[a:b]=np.where(labels,.5/k,.5/(b-a-k))
    weights*=len(weights)/len(table.indices)
    if not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Invalid cascade/class sample weights')
    return weights


def effective_parameters(C):
    kwargs=dict(C=float(C),solver='lbfgs',fit_intercept=True,max_iter=1000,tol=1e-8,class_weight=None)
    # sklearn 1.9 deprecated penalty; l1_ratio=0 specifies the same L2 model.
    penalty=inspect.signature(LogisticRegression).parameters['penalty'].default
    if penalty=='deprecated':
        kwargs['l1_ratio']=0.
    else:
        kwargs['penalty']='l2'
    return kwargs


def fit_head(table,C):
    validate_table(table)
    if isinstance(C,bool) or C not in C_GRID:
        raise ValueError('C must belong to the fixed grid .1/1/10')
    scaler=StandardScaler().fit(table.features)
    scaled=scaler.transform(table.features)
    model=LogisticRegression(**effective_parameters(C))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaled,table.labels,sample_weight=cascade_class_weights(table))
    head=LinearHead(FEATURE_NAMES,tuple(scaler.mean_),tuple(scaler.scale_),
                    tuple(model.coef_[0]),float(model.intercept_[0]),float(C))
    expected=model.decision_function(scaled)
    restored=LinearHead.from_dict(head.to_dict())
    if not np.allclose(expected,restored.decision_function(table.features),atol=1e-12,rtol=1e-10):
        raise ValueError('Portable scorer differs from sklearn scores')
    for a,b,k in zip(table.offsets[:-1],table.offsets[1:],table.predicted_counts):
        ids=tuple(map(int,table.candidate_ids[a:b]))
        order=sorted(range(b-a),key=lambda j:(-expected[a+j],-table.probabilities[a+j],ids[j]))
        actual=rank_candidates(ids,table.probabilities[a:b],table.features[a:b],int(k),restored)
        if actual!=frozenset(ids[j] for j in order[:min(k,b-a)]):
            raise ValueError('Portable scorer changes selected source sets')
    return head


def select_head(fit,dev,graph,*,budget_seconds=1800):
    budget=Budget(budget_seconds)
    validate_table(fit); validate_table(dev)
    if set(map(int,fit.indices)) & set(map(int,dev.indices)):
        raise ValueError('Fit/dev case indices overlap')
    candidates=[]
    heads=[]
    for C in C_GRID:
        budget.check()
        start=time.monotonic()
        print(f'Fit C={C:g}: {len(fit.indices)} cascades, {len(fit.labels)} candidate rows',flush=True)
        h=fit_head(fit,C)
        budget.check()
        report=evaluate_reranker(dev,h,graph,include_ci=False,budget=budget)
        budget.check()
        candidates.append(dict(C=C,report=report,elapsed_seconds=time.monotonic()-start,effective_parameters=effective_parameters(C)))
        heads.append(h)
        print(f'C={C:g}: dev F1={report["learned"]["all"]["f1"]:.6f}; delta-v3={report["delta_f1"]:+.6f}',flush=True)
    maximum=max(c['report']['learned']['all']['f1'] for c in candidates)
    # Compare each candidate with the maximum; running-best comparisons can
    # incorrectly chain near-ties and choose the larger C.
    best=next(j for j,c in enumerate(candidates) if maximum-c['report']['learned']['all']['f1']<=1e-12)
    chosen=heads[best]
    report=candidates[best]['report']
    gate=dev_gate(report)
    print(f'Selected C={chosen.C:g}; dev gate={gate["passed"]}; {gate["reasons"]}',flush=True)
    budget.check()
    return chosen,dict(exploratory=True,selected_C=chosen.C,head=chosen.to_dict(),dev_report=report,gate=gate,
        candidates=candidates,fit_indices=list(map(int,fit.indices)),dev_indices=list(map(int,dev.indices)),
        fit_rows=len(fit.labels),scaler_fit_only=True,refit=False,sklearn_version=importlib.metadata.version('scikit-learn'),
        elapsed_seconds=time.monotonic()-budget.start)
