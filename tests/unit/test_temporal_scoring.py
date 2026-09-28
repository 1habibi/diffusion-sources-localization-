from pathlib import Path
import numpy as np
import pytest
import torch


def test_candidate_pool_count_and_beta_zero():
    from diffusion_sources.temporal_scoring import CandidateScores, correct_sources
    c = CandidateScores((0, 1, 2), (.9, .8, .1), (False, False, True), frozenset({0}), 1)
    assert correct_sources(c, 0.) == frozenset({0})
    assert correct_sources(c, 1.) == frozenset({2})
    c = CandidateScores((8,), (.2,), (True,), frozenset({8}), 3)
    assert correct_sources(c, 1.) == frozenset({8})


def test_ties_are_deterministic_and_zero_keeps_original():
    from diffusion_sources.temporal_scoring import CandidateScores, correct_sources
    c = CandidateScores((2, 1), (.5, .5), (True, True), frozenset({2}), 1)
    assert correct_sources(c, 0.) == frozenset({2})
    assert correct_sources(c, .1) == frozenset({1})
    c = CandidateScores((0, 1), (.25, .5), (True, False), frozenset({1}), 1)
    assert correct_sources(c, .25) == frozenset({1})


@pytest.mark.parametrize('ids,scores,early,baseline,count', [
    ((), (), (), frozenset(), 1),
    ((1, 1), (.1, .2), (True, True), frozenset({1}), 1),
    ((1,), (float('nan'),), (True,), frozenset({1}), 1),
    ((1,), (.1, .2), (True,), frozenset({1}), 1),
    ((1,), (.1,), (), frozenset({1}), 1),
    ((1,), (.1,), (True,), frozenset({2}), 1),
])
def test_bad_scoring_input(ids, scores, early, baseline, count):
    from diffusion_sources.temporal_scoring import CandidateScores, correct_sources
    with pytest.raises(ValueError):
        correct_sources(CandidateScores(ids, scores, early, baseline, count), .1)


def test_train_selection_smallest_beta_on_tie():
    from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord, select_beta
    c = CandidateScores((0, 1), (.9, .1), (False, False), frozenset({0}), 1)
    beta, table = select_beta([PilotRecord(0, frozenset({0}), c, True)], [1., 0.])
    assert beta == 0. and table == {1.: 1., 0.: 1.}
    with pytest.raises(ValueError):
        select_beta([], [0.])


def test_negative_beta_rejected():
    from diffusion_sources.temporal_scoring import CandidateScores, correct_sources
    with pytest.raises(ValueError):
        correct_sources(CandidateScores((0,), (.9,), (True,), frozenset({0}), 1), -.1)


@pytest.mark.parametrize('split', ['test', 'holdout'])
def test_forbidden_split_rejected_before_read(split, tmp_path):
    from diffusion_sources.temporal_scoring import collect_records
    with pytest.raises(ValueError, match='split'):
        collect_records(tmp_path / 'missing', tmp_path / 'missing', split, [0], torch.device('cpu'))


def test_collect_real_inference(temporal_dataset, tmp_path, monkeypatch):
    from diffusion_sources.temporal_scoring import collect_records, correct_sources
    import yaml
    from diffusion_sources.models import JointSourceCountGCN
    data, _ = temporal_dataset
    run = tmp_path / 'run'
    run.mkdir()
    config = {'data': {'feature_names': ['observed_infected', 'log_degree_normalized']},
              'model': {'hidden_dim': 8, 'dropout': 0.}, 'training': {'seed': 13}}
    (run / 'config.yaml').write_text(yaml.safe_dump(config))
    torch.save(JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.).state_dict(), run / 'best_model.pt')
    accesses = {}
    original_load = np.load
    class CountedArchive:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __contains__(self, key):
            return key in self.archive
        def __getitem__(self, key):
            accesses[key] = accesses.get(key, 0) + 1
            return self.archive[key]
    def counted_load(path, *args, **kwargs):
        archive = original_load(path, *args, **kwargs)
        return CountedArchive(archive) if str(path).endswith('train.npz') else archive
    # Loader without a context manager also owns this archive: keep original
    # for its one-time conversion, and count just the replay's context-managed archive.
    replay_opened = False
    def first_only(path, *args, **kwargs):
        nonlocal replay_opened
        if str(path).endswith('train.npz') and not replay_opened:
            replay_opened = True
            return counted_load(path, *args, **kwargs)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', first_only)
    records = collect_records(data, run, 'train', [0, 1, 2], torch.device('cpu'))
    assert [r.index for r in records] == [0, 1, 2]
    for r in records:
        assert correct_sources(r.candidates, 0.) == r.candidates.baseline_sources
        assert r.true_sources <= set(r.candidates.candidate_ids)
        assert len(correct_sources(r.candidates, 1.)) == len(r.candidates.baseline_sources)
    assert max(accesses.values()) == 1, 'Replay must unpack each full-split field once, not once per cascade'
    with pytest.raises(ValueError):
        collect_records(data, run, 'train', [0, 0], torch.device('cpu'))
