import itertools

import numpy as np
import pytest
import torch
import yaml

from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.temporal_learned_collect import check_train_partition, collect_candidates
from diffusion_sources.temporal_scoring import CandidateScores, correct_sources


def partition_case():
    config = {'simulation': {'source_counts': [1,2,3], 'probabilities': [.01,.02,.03],
                             'distance_ranges': [{'min':1,'max':2}, {'min':3,'max':5}], 'max_steps':3},
              'observation': {'fractions':[1.,.75,.5], 'false_positive_count':0}}
    conditions = list(itertools.product(range(2), [.01,.02,.03], [1.,.75,.5], [1,2,3]))
    repeats = [conditions[i % 54] for i in range(9990)]
    arrays = {'source_counts': np.array([r[3] for r in repeats]),
              'probabilities': np.array([r[1] for r in repeats]),
              'observation_fractions': np.array([r[2] for r in repeats]),
              'simulation_seeds': np.arange(9990), 'observation_seeds': np.arange(9990)+100000}
    return config, arrays


def test_partition_balanced_and_seed_union_disjoint():
    fit, dev = check_train_partition(*partition_case())
    assert fit == list(range(1620)) and dev == list(range(1620,2160))


@pytest.mark.parametrize('field,index,value', [
    ('source_counts', 0, 2), ('probabilities', 0, .8),
    ('observation_seeds', 1620, 0), ('simulation_seeds', 1, 0),
    ('observation_seeds', 1, -1),
])
def test_partition_tampering_stops(field, index, value):
    cfg, arrays = partition_case()
    arrays[field][index] = value
    with pytest.raises(ValueError):
        check_train_partition(cfg, arrays)


@pytest.fixture
def small_run(temporal_dataset, tmp_path):
    data, cfg = temporal_dataset
    run = tmp_path/'run'; run.mkdir()
    model = JointSourceCountGCN(input_dim=2, hidden_dim=4, dropout=0)
    config = {'data': {'feature_names':['observed_infected','log_degree_normalized']},
              'model': {'input_dim':2,'hidden_dim':4,'dropout':0}}
    (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    torch.save(model.state_dict(), run/'best_model.pt')
    return data, run


def test_real_tiny_replay_and_frozen_model(small_run):
    table = collect_candidates(*small_run, 'train', [0,1,2,3,4,5], torch.device('cpu'))
    assert table.indices.tolist() == list(range(6))
    assert table.true_counts.tolist() == [1,2,3,1,2,3]
    assert len(table.early_mask_hash) == 64


def test_labels_removed_before_gcn(small_run, monkeypatch):
    forward = JointSourceCountGCN.forward
    def checked(self, example):
        assert 'source_labels' not in example and 'source_count' not in example
        return forward(self, example)
    monkeypatch.setattr(JointSourceCountGCN, 'forward', checked)
    collect_candidates(*small_run, 'train', [0,1], torch.device('cpu'))


def test_candidate_coverage_no_filtering(small_run, monkeypatch):
    def corrupt(*args):
        raise ValueError('replay mismatch')
    monkeypatch.setattr('diffusion_sources.temporal_learned_collect.replay_early_mask', corrupt)
    with pytest.raises(ValueError, match='replay mismatch'):
        collect_candidates(*small_run, 'train', [0,1], torch.device('cpu'))


def test_control_beta_zero(small_run):
    table = collect_candidates(*small_run, 'train', [0], torch.device('cpu'))
    c = CandidateScores(tuple(table.candidate_ids), tuple(table.probabilities),
                        tuple(table.early_flags.astype(bool)), table.snapshot_sources[0], int(table.predicted_counts[0]))
    assert correct_sources(c, 0) == table.snapshot_sources[0]


def test_deadline_and_changed_inputs_stop(small_run, monkeypatch):
    ticks = iter([0., 0., 2., 3., 4.])
    monkeypatch.setattr('diffusion_sources.temporal_learned_collect.time.monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        collect_candidates(*small_run, 'train', [0], torch.device('cpu'), budget_seconds=1)


def test_changed_inputs_stop(small_run, monkeypatch):
    from diffusion_sources.temporal_replay import replay_early_mask
    def changing(*args):
        early = replay_early_mask(*args)
        path = small_run[1]/'config.yaml'
        path.write_text(path.read_text()+'\n# changed\n', encoding='utf-8')
        return early
    monkeypatch.setattr('diffusion_sources.temporal_learned_collect.replay_early_mask', changing)
    with pytest.raises(ValueError, match='changed'):
        collect_candidates(*small_run, 'train', [0], torch.device('cpu'))
