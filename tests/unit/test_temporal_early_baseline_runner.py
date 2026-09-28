import hashlib
import json

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def baseline_run(tmp_path, monkeypatch):
    from diffusion_sources import temporal_pilot_cli as pilot
    from diffusion_sources.temporal_pilot_artifacts import write_stage
    from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord
    from diffusion_sources import temporal_early_baseline_runner as runner
    data, run, old, out = [tmp_path / name for name in ('data', 'run', 'pilot', 'analysis')]
    data.mkdir(); run.mkdir()
    (data / 'graph.npz').write_bytes(b'graph')
    (data / 'config.yaml').write_text('simulation: {}')
    np.savez(data / 'train.npz', source_counts=[1])
    np.savez(data / 'validation.npz', source_counts=np.ones(1998))
    (run / 'config.yaml').write_text(yaml.safe_dump({'training': {'seed': 7026}, 'data': {}}))
    (run / 'best_model.pt').write_bytes(b'checkpoint')
    (run / 'metrics.json').write_text('{}')
    def historical(identity):
        # Real pilot predates the newly added analysis files and git revision.
        identity['code']['revision'] = 'historical-pilot-commit'
        identity['code']['source_hashes'].pop('temporal_early_baseline.py', None)
        identity['code']['source_hashes'].pop('temporal_early_baseline_runner.py', None)
        return identity
    write_stage(old, 'select', {'identity': historical(pilot._selection_identity(data, run))}, {'beta': .5})
    c = CandidateScores((0, 1, 2), (.9, .1, .5), (True, True, False), frozenset({0}), 1)
    records = [PilotRecord(i, frozenset({0}), c, False) for i in range(1998)]
    masks = [(r.index, c.candidate_ids, c.early_observed, False) for r in records]
    mask_hash = hashlib.sha256(json.dumps(masks).encode()).hexdigest()
    report = {'beta': .5, 'seed': 7026, 'baseline': {'all': {'n': 1998, 'f1': 1.}},
              'temporal': {'all': {'n': 1998, 'f1': 1.}}, 'early_mask_hash': mask_hash}
    write_stage(old, 'validation', {'identity': historical(pilot._validation_identity(data, run, old, .5))}, report)
    calls = []
    def collect(data_dir, run_dir, split, indices, device):
        assert data_dir == data and run_dir == run and split == 'validation'
        assert indices == list(range(1998))
        calls.append(split)
        return records
    monkeypatch.setattr(runner, 'collect_records', collect)
    return runner, data, run, old, out, records, calls


def test_runner_reproduces_saved_control_then_saves_and_resumes(baseline_run, capsys):
    runner, data, run, old, out, _, calls = baseline_run
    original_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in old.rglob('*') if p.is_file()}
    report = runner.run_seed(data, run, old, out, torch.device('cpu'))
    assert report['all']['early_expected_f1'] == .5
    assert report['all']['temporal_minus_early'] == .5
    assert (out / 'seed_7026' / 'predictions.csv').is_file()
    assert 'early_expected_f1' in capsys.readouterr().out
    assert runner.run_seed(data, run, old, out, torch.device('cpu')) == report
    assert calls == ['validation']
    assert original_hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_hashes}


def test_changed_inputs_rejected_before_inference(baseline_run):
    runner, data, run, old, out, _, calls = baseline_run
    (run / 'best_model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='input'):
        runner.run_seed(data, run, old, out, torch.device('cpu'))
    assert not calls and not out.exists()


def test_changed_record_masks_rejected_before_saving(baseline_run):
    from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord
    runner, data, run, old, out, records, _ = baseline_run
    c = records[0].candidates
    records[0] = PilotRecord(0, frozenset({0}), CandidateScores(c.candidate_ids, c.scores,
        (True, False, False), c.baseline_sources, c.predicted_count), False)
    with pytest.raises(ValueError, match='mask'):
        runner.run_seed(data, run, old, out, torch.device('cpu'))
    assert not (out / 'seed_7026').exists()


def test_baseline_or_temporal_mismatch_rejected(baseline_run):
    from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord
    runner, data, run, old, out, records, _ = baseline_run
    c = records[0].candidates
    records[0] = PilotRecord(0, frozenset({0}), CandidateScores(c.candidate_ids, c.scores,
        c.early_observed, frozenset({1}), c.predicted_count), False)
    with pytest.raises(ValueError, match='F1'):
        runner.run_seed(data, run, old, out, torch.device('cpu'))
    assert not (out / 'seed_7026').exists()


def test_output_must_not_overlap_old_pilot(baseline_run):
    runner, data, run, old, _, _, calls = baseline_run
    with pytest.raises(ValueError, match='overlap'):
        runner.run_seed(data, run, old, old / 'new', torch.device('cpu'))
    assert not calls


def test_summary_requires_three_consistent_distinct_seeds():
    from diffusion_sources.temporal_early_baseline_runner import summarize_seeds
    def report(seed, delta=.1):
        return {'seed': seed, 'beta': .5, 'early_mask_hash': 'same',
                'all': {'n': 1998, 'snapshot_f1': .35, 'early_expected_f1': .4,
                        'temporal_f1': .4 + delta, 'temporal_minus_early': delta},
                'f1_ci': [.01, .2]}
    reports = [report(seed) for seed in (7026, 7027, 7028)]
    result = summarize_seeds(reports)
    assert result['mean_temporal_minus_early'] == pytest.approx(.1)
    assert result['positive_delta_all_seeds']
    assert not summarize_seeds([report(7026), report(7027, -.1), report(7028)])['positive_delta_all_seeds']
    for bad in (reports[:2], [reports[0]] * 3, [*reports[:2], {**reports[2], 'beta': .25}],
                [*reports[:2], {**reports[2], 'early_mask_hash': 'different'}]):
        with pytest.raises(ValueError):
            summarize_seeds(bad)
