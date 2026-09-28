"""Research-stage wiring with controlled frozen inference, real scoring and artifacts."""
import itertools
import json

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def pilot_flow(temporal_dataset, tmp_path, monkeypatch):
    import diffusion_sources.temporal_pilot_cli as cli
    from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord
    data, config = temporal_dataset
    config['simulation'].update(distance_ranges=[{'min': 1, 'max': 2}, {'min': 3, 'max': 5}],
                                probabilities=[.01, .02, .03])
    config['observation']['fractions'] = [1., .75, .5]
    (data / 'config.yaml').write_text(yaml.safe_dump(config))
    conditions = list(itertools.product([0, 1], [.01, .02, .03], [1., .75, .5], [1, 2, 3])) * 10
    np.savez(data / 'train.npz', source_counts=[r[3] for r in conditions],
             probabilities=[r[1] for r in conditions], observation_fractions=[r[2] for r in conditions])
    np.savez(data / 'validation.npz', source_counts=np.tile([1, 2, 3], 666))
    runs = []
    for seed in (7026, 7027, 7028):
        run = tmp_path / f'run_{seed}'; run.mkdir()
        cfg = {'data': {'feature_names': ['observed_infected']}, 'model': {'hidden_dim': 8},
               'training': {'seed': seed}, 'loss': {}}
        (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
        (run / 'best_model.pt').write_bytes(f'frozen {seed}'.encode())
        (run / 'metrics.json').write_text(json.dumps({'validation_prediction_metrics': {
            'joint_estimated_k': {'all': {'f1': .36299632966299633}}}}))
        runs.append(run)
    calls = []
    corrupted_masks = set()

    def frozen_records(data_dir, run_dir, split, indices, device):
        assert data_dir == data and split in ('train', 'validation')
        seed = yaml.safe_load((run_dir / 'config.yaml').read_text())['training']['seed']
        calls.append((seed, split, tuple(indices)))
        records = []
        for index in indices:
            k = index % 3 + 1
            true = frozenset(range(k))
            count = k
            if split == 'train':
                baseline = frozenset(range(3, 3 + k))
            else:
                ordinal = index // 3
                correct_n = {1: 145, 2: 280, 3: 298}[k]
                baseline = true if ordinal < correct_n else frozenset(range(3, 3 + k))
                if k == 3 and ordinal in range(298, 302):
                    count, baseline = 2, frozenset({0, 3})
                elif k == 3 and ordinal in range(302, 304):
                    baseline = frozenset({0, 3, 4})
            ids = tuple(range(6))
            early = tuple(n in true and seed not in corrupted_masks for n in ids)
            candidates = CandidateScores(ids, tuple(.6 if n in baseline else .2 for n in ids),
                                         early, baseline, count)
            records.append(PilotRecord(index, true, candidates, not any(early)))
        return records

    # Only the unavailable frozen inference boundary is doubled; replay/inference
    # has separate real integration coverage. Stage wiring/scoring/evaluation are real.
    monkeypatch.setattr(cli, 'collect_records', frozen_records)
    return cli, data, runs, tmp_path / 'pilot', calls, corrupted_masks


def test_successful_three_seed_flow_is_locked_and_read_isolated(pilot_flow, monkeypatch):
    cli, data, runs, out, calls, _ = pilot_flow
    opened = []
    original_load = np.load
    original_hash = cli.sha256_file
    original_select = cli.select_beta
    tuning = []

    def guarded_load(path, *args, **kwargs):
        opened.append(str(path))
        assert not str(path).endswith(('test.npz', 'holdout.npz'))
        return original_load(path, *args, **kwargs)

    def guarded_hash(path):
        opened.append(str(path))
        assert not str(path).endswith(('test.npz', 'holdout.npz'))
        return original_hash(path)

    def tune(records, grid):
        tuning.append(len(records))
        return original_select(records, grid)

    monkeypatch.setattr(np, 'load', guarded_load)
    monkeypatch.setattr(cli, 'sha256_file', guarded_hash)
    monkeypatch.setattr(cli, 'select_beta', tune)
    selected = cli.select(data, runs[0], out, torch.device('cpu'))
    assert selected['beta'] == .5 and selected['n'] == 540
    assert not any(p.endswith('validation.npz') for p in opened)
    primary = cli.validate(data, runs[0], out, torch.device('cpu'))
    assert primary['baseline']['all']['f1'] == pytest.approx(.36299632966299633, abs=1e-12)
    assert primary['gate']['passed'] and primary['beta'] == .5
    result = cli.confirm(data, runs[0], out, runs[1:], torch.device('cpu'))
    assert result['passed'] and result['beta'] == .5
    assert set(result['per_seed_delta']) == {'7026', '7027', '7028'}
    assert tuning == [540]
    assert [(seed, split, len(indices)) for seed, split, indices in calls] == [
        (7026, 'train', 540), (7026, 'validation', 1998),
        (7027, 'validation', 1998), (7028, 'validation', 1998)]
    for stage in ('validation', 'seed_7027', 'seed_7028'):
        assert (out / stage / 'predictions.csv').is_file()
        payload = json.loads((out / stage / 'payload.json').read_text())
        assert payload['early_mask_hash'] == primary['early_mask_hash']
        assert payload['beta'] == selected['beta']
    assert cli.confirm(data, runs[0], out, runs[1:], torch.device('cpu')) == result
    assert len(calls) == 4 and tuning == [540], 'Resume must not infer or retune'


def test_saved_control_mismatch_does_not_complete_validation(pilot_flow):
    cli, data, runs, out, _, _ = pilot_flow
    cli.select(data, runs[0], out, torch.device('cpu'))
    path = runs[0] / 'metrics.json'
    metrics = json.loads(path.read_text())
    metrics['validation_prediction_metrics']['joint_estimated_k']['all']['f1'] = .5
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match='baseline F1'):
        cli.validate(data, runs[0], out, torch.device('cpu'))
    assert not (out / 'validation').exists()


def test_failed_primary_gate_does_not_infer_repeat(pilot_flow):
    cli, data, runs, out, calls, _ = pilot_flow
    from diffusion_sources.temporal_pilot_artifacts import write_stage
    from diffusion_sources.temporal_statistics import primary_gate
    cli.select(data, runs[0], out, torch.device('cpu'))
    report = {'delta_f1': 0., 'f1_ci': [0., 0.], 'delta_by_k': {'1': 0., '2': 0., '3': 0.},
              'count_unchanged': True}
    report['gate'] = primary_gate(report)
    identity = cli._validation_identity(data, runs[0], out, .5)
    write_stage(out, 'validation', {'identity': identity}, report)
    with pytest.raises(ValueError, match='Primary gate failed'):
        cli.confirm(data, runs[0], out, runs[1:], torch.device('cpu'))
    assert len(calls) == 1
    assert not (out / 'seed_7027').exists()


def test_repeat_masks_mismatch_does_not_complete_confirmation(pilot_flow):
    cli, data, runs, out, _, corrupted_masks = pilot_flow
    cli.select(data, runs[0], out, torch.device('cpu'))
    cli.validate(data, runs[0], out, torch.device('cpu'))
    corrupted_masks.add(7027)
    with pytest.raises(ValueError, match='Early masks differ'):
        cli.confirm(data, runs[0], out, runs[1:], torch.device('cpu'))
    assert not (out / 'seed_7027').exists()
    assert not (out / 'confirmation').exists()
