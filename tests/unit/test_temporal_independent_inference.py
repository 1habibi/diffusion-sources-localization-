import json
import networkx as nx
import pytest
import torch
import numpy as np
from tests.fixtures.temporal_independent import make_case
from tests.unit.test_temporal_independent_generation import fake_generate
from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord


def controlled_records():
    records = []
    for i in range(1998):
        k = i % 3 + 1
        truth, baseline = frozenset(range(k)), frozenset(range(10-k, 10))
        scores = tuple(.6 if node in truth else .8 if node in baseline else .1 for node in range(10))
        c = CandidateScores(tuple(range(10)), scores, tuple(node <= k for node in range(10)), baseline, k)
        records.append(PilotRecord(i, truth, c, False))
    return records


def sealed_case(tmp_path, monkeypatch):
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs
    from diffusion_sources import temporal_independent_generation as g
    paths = make_case(tmp_path)
    freeze_inputs(paths)
    monkeypatch.setattr(g, 'generate_dataset', fake_generate)
    g.generate_and_seal(paths)
    return paths


def test_no_open_no_inference(tmp_path, monkeypatch):
    from diffusion_sources.temporal_independent_inference import evaluate_seed
    paths = sealed_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='opened'):
        evaluate_seed(paths, 7026, torch.device('cpu'))
    assert not (paths.reports / 'seed_7026').exists()


def test_open_and_three_seed_flow(tmp_path, monkeypatch):
    from diffusion_sources import temporal_independent_inference as inference
    paths = sealed_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='confirmation'):
        inference.open_evaluation(paths, 'yes')
    assert not (paths.reports / 'opened').exists()
    marker = inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    assert inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT') == marker
    def collect(paths, seed, device):
        assert (paths.reports / 'opened/complete').is_file()
        return controlled_records()
    monkeypatch.setattr(inference, 'collect_independent_records', collect)
    for seed in (7026, 7027, 7028):
        result = inference.evaluate_seed(paths, seed, torch.device('cpu'))
        assert result['n'] == 1998 and result['count_unchanged']
        assert result['evaluation_role'] == 'independent_confirmation'
        assert all(row['temporal_f1'] == 1 and row['snapshot_f1'] == 0 for row in result['rows'])
        assert result['rows'][0]['early_expected_f1'] == .5
        assert inference.evaluate_seed(paths, seed, torch.device('cpu')) == result
    monkeypatch.setattr(inference, 'collect_independent_records', lambda *a: pytest.fail('resume'))
    inference.evaluate_seed(paths, 7026, torch.device('cpu'))
    (paths.runs[7026] / 'best_model.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        inference.evaluate_seed(paths, 7026, torch.device('cpu'))


@pytest.mark.parametrize('failure', ['wrong_seed', 'duplicate', 'reordered', 'interrupted'])
def test_failed_collection_not_published(tmp_path, monkeypatch, failure):
    from diffusion_sources import temporal_independent_inference as inference
    paths = sealed_case(tmp_path, monkeypatch)
    inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    def collect(*args):
        if failure == 'interrupted':
            raise RuntimeError('interrupted')
        records = controlled_records()
        if failure == 'duplicate':
            records[-1] = records[0]
        if failure == 'reordered':
            records.reverse()
        return records
    monkeypatch.setattr(inference, 'collect_independent_records', collect)
    with pytest.raises((ValueError, RuntimeError)):
        inference.evaluate_seed(paths, 999 if failure == 'wrong_seed' else 7026, torch.device('cpu'))
    assert not (paths.reports / 'seed_7026').exists()
    assert (paths.reports / 'opened/complete').exists()


def test_real_collector_authorization_and_sources_free_model(tmp_path, monkeypatch):
    from diffusion_sources import temporal_independent_generation as g
    from diffusion_sources import temporal_independent_inference as inference
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs
    from diffusion_sources.models import JointSourceCountGCN
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request_cleanup = lambda: torch.set_num_threads(previous_threads)
    paths = make_case(tmp_path, real_checkpoint=True)
    freeze_inputs(paths)
    def produce(cfg, root):
        fake_generate(cfg, root)
        with np.load(root / 'independent_holdout.npz') as z:
            arrays = {key: z[key] for key in z.files}
        arrays['source_counts'] = np.arange(1998) % 3 + 1
        arrays['source_labels'] = np.arange(10)[None, :] < arrays['source_counts'][:, None]
        arrays['features'][:, 0, 0] = 1
        np.savez(root / 'independent_holdout.npz', **arrays)
    monkeypatch.setattr(g, 'generate_dataset', produce)
    g.generate_and_seal(paths)
    inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    def replay(graph, config, archive, index):
        assert (paths.reports / 'opened/complete').exists()
        return np.zeros(10, dtype=bool) if index == 0 else np.arange(10) < 4
    monkeypatch.setattr(inference, 'replay_early_mask', replay)
    original = JointSourceCountGCN.forward
    def checked(model, data):
        assert not model.training and not torch.is_grad_enabled()
        assert 'source_labels' not in data and 'source_count' not in data
        return original(model, data)
    monkeypatch.setattr(JointSourceCountGCN, 'forward', checked)
    monkeypatch.setattr(torch.optim.Adam, '__init__', lambda *a, **kw: pytest.fail('No optimizer'))
    try:
        records = inference.collect_independent_records(paths, 7026, torch.device('cpu'))
    finally:
        request_cleanup()
    assert len(records) == 1998 and records[0].early_empty
    assert [len(r.true_sources) for r in records[:3]] == [1, 2, 3]
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA unavailable'):
        inference.collect_independent_records(paths, 7026, torch.device('cuda'))


@pytest.mark.parametrize('artifact', ['data', 'config', 'cache'])
def test_changed_input_after_open_is_rejected(tmp_path, monkeypatch, artifact):
    from diffusion_sources import temporal_independent_inference as inference
    paths = sealed_case(tmp_path, monkeypatch)
    inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    monkeypatch.setattr(inference, 'collect_independent_records', lambda *a: pytest.fail('No target read'))
    if artifact == 'data':
        with (paths.data / 'independent_holdout.npz').open('ab') as f: f.write(b'changed')
    elif artifact == 'config':
        with (paths.runs[7026] / 'config.yaml').open('a') as f: f.write('\nchanged: true\n')
    else:
        with (paths.reference / 'config.yaml').open('a') as f: f.write('\nchanged: true\n')
    with pytest.raises(ValueError):
        inference.evaluate_seed(paths, 7026, torch.device('cpu'))


def test_changed_early_masks_rejected_between_seeds(tmp_path, monkeypatch):
    from dataclasses import replace
    from diffusion_sources import temporal_independent_inference as inference
    paths = sealed_case(tmp_path, monkeypatch)
    inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    monkeypatch.setattr(inference, 'collect_independent_records', lambda *a: controlled_records())
    inference.evaluate_seed(paths, 7026, torch.device('cpu'))
    records = controlled_records()
    records[0] = replace(records[0], candidates=replace(records[0].candidates, early_observed=(False,)*10), early_empty=True)
    monkeypatch.setattr(inference, 'collect_independent_records', lambda *a: records)
    with pytest.raises(ValueError, match='masks differ'):
        inference.evaluate_seed(paths, 7027, torch.device('cpu'))
    assert not (paths.reports / 'seed_7027').exists()
