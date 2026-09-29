from dataclasses import replace
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diffusion_sources.temporal_learned_training import cascade_class_weights, fit_head, select_head
from diffusion_sources.temporal_learned_features import LinearHead, rank_candidates
from tests.fixtures.temporal_learned import tiny_graph, tiny_table


def test_weights_equal_cascade_and_class_mass():
    t=tiny_table(); w=cascade_class_weights(t)
    assert w.mean() == pytest.approx(1)
    for a,b in zip(t.offsets[:-1],t.offsets[1:]):
        assert w[a:b].sum() == pytest.approx(6)
        assert w[a:b][t.labels[a:b]].sum() == pytest.approx(3)
        assert w[a:b][~t.labels[a:b]].sum() == pytest.approx(3)


def test_scaler_fit_only_and_json_sklearn_score_parity():
    t=tiny_table(); h=fit_head(t,1.)
    scaler=StandardScaler().fit(t.features)
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8).fit(
        scaler.transform(t.features),t.labels,sample_weight=cascade_class_weights(t))
    np.testing.assert_allclose(h.mean, scaler.mean_,rtol=0,atol=0)
    np.testing.assert_allclose(h.decision_function(t.features),model.decision_function(scaler.transform(t.features)),rtol=1e-10,atol=1e-12)
    restored=LinearHead.from_dict(h.to_dict())
    for a,b,k in zip(t.offsets[:-1],t.offsets[1:],t.predicted_counts):
        assert rank_candidates(tuple(t.candidate_ids[a:b]),t.probabilities[a:b],t.features[a:b],int(k),h) == rank_candidates(tuple(t.candidate_ids[a:b]),t.probabilities[a:b],t.features[a:b],int(k),restored)


def test_three_c_smallest_tie_and_no_refit(monkeypatch):
    fit=tiny_table(); dev=replace(tiny_table(),indices=np.arange(6)+10)
    from diffusion_sources.temporal_learned_training import fit_head as real_fit
    calls=[]
    def watched(table,C):
        calls.append((table,C))
        return real_fit(table,C)
    monkeypatch.setattr('diffusion_sources.temporal_learned_training.fit_head',watched)
    h, result=select_head(fit,dev,tiny_graph())
    assert [c for _,c in calls] == [.1,1.,10.]
    assert all(t is fit for t,_ in calls)
    # C=1 and C=10 tie at perfect F1; C=.1 ranks one source incorrectly.
    assert h.C == 1. and len(result['candidates']) == 3
    np.testing.assert_equal(h.mean,fit.features.mean(axis=0))


def test_fit_dev_overlap_rejected():
    with pytest.raises(ValueError, match='overlap'):
        select_head(tiny_table(),tiny_table(),tiny_graph())


def test_convergence_warning_is_error(monkeypatch):
    def bad_fit(*args,**kwargs):
        warnings.warn('not converged',ConvergenceWarning)
    monkeypatch.setattr(LogisticRegression,'fit',bad_fit)
    with pytest.raises(ConvergenceWarning):
        fit_head(tiny_table(),1.)


def test_timeout_stops_between_fits(monkeypatch):
    ticks=iter([0.,0.,2.,3.])
    monkeypatch.setattr('diffusion_sources.temporal_learned_training.time.monotonic',lambda:next(ticks))
    with pytest.raises(TimeoutError):
        select_head(tiny_table(),replace(tiny_table(),indices=np.arange(6)+10),tiny_graph(),budget_seconds=1)
