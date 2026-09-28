import json
from pathlib import Path
import subprocess
import importlib
import torch


def code_cells():
    path = Path(__file__).parents[2] / 'notebooks/colab_temporal_v3_pilot.ipynb'
    return {c['id']: ''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells']
            if c['cell_type'] == 'code'}


def test_baseline_setup_preserves_old_output_and_does_not_launch_inference(tmp_path, monkeypatch):
    from diffusion_sources import temporal_early_baseline_runner as runner
    def no_inference(*args, **kwargs):
        raise AssertionError('Setup must not infer')
    monkeypatch.setattr(runner, 'run_seed', no_inference)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    env = {'REPO': tmp_path, 'DRIVE_ROOT': tmp_path, 'PILOT_DIR': tmp_path / 'old', 'DEVICE': 'cpu'}
    exec(code_cells()['baseline_setup'], env)
    assert env['PILOT_DIR'] == tmp_path / 'old'
    assert env['BASELINE_DIR'] != env['PILOT_DIR']
    assert env['BASELINE_RESULTS'] == {}


def test_setup_reloads_cached_scoring_before_runner(tmp_path, monkeypatch):
    from diffusion_sources import temporal_early_baseline as scoring
    from diffusion_sources import temporal_early_baseline_runner as runner
    stale = object()
    monkeypatch.setattr(scoring, 'evaluate_early_baseline', stale)
    monkeypatch.setattr(runner, 'run_seed', stale)
    real_reload = importlib.reload
    reloaded = []
    def reload_module(module):
        reloaded.append(module.__name__)
        return real_reload(module)
    monkeypatch.setattr(importlib, 'reload', reload_module)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    env = {'REPO': tmp_path, 'DRIVE_ROOT': tmp_path,
           'PILOT_DIR': tmp_path / 'old', 'DEVICE': 'cpu'}
    exec(code_cells()['baseline_setup'], env)
    assert reloaded == ['diffusion_sources.temporal_early_baseline',
                        'diffusion_sources.temporal_early_baseline_runner']
    assert callable(scoring.evaluate_early_baseline)
    assert env['run_seed'] is runner.run_seed and callable(env['run_seed'])
    assert runner.evaluate_early_baseline is scoring.evaluate_early_baseline


def test_each_baseline_cell_runs_only_its_frozen_seed_with_visible_summary(tmp_path, capsys):
    c = code_cells()
    def run(data, run_dir, pilot, out, device):
        assert device == torch.device('cpu')
        return {'seed': int(run_dir.name.replace('seed_', '')), 'rows': [], 'all': {'n': 1998}}
    env = {'run_seed': run, 'torch': torch, 'DATA_DIR': tmp_path / 'data',
           'RUN_DIR': tmp_path / 's1b/seed_7026', 'RUN_ROOT': tmp_path,
           'PILOT_DIR': tmp_path / 'old', 'BASELINE_DIR': tmp_path / 'new',
           'DEVICE': 'cpu', 'BASELINE_RESULTS': {}}
    for seed in (7026, 7027, 7028):
        exec(c[f'baseline_{seed}'], env)
        assert env['BASELINE_RESULTS'][seed]['seed'] == seed
    assert set(env['BASELINE_RESULTS']) == {7026, 7027, 7028}


def test_baseline_final_cell_prints_and_saves_summary(tmp_path, capsys):
    from diffusion_sources.temporal_pilot_artifacts import write_stage
    # The real aggregator is tested with reports separately; external inference
    # is absent here. Execute the cell to check visible kernel output + artifact.
    def summary(reports):
        assert [r['seed'] for r in reports] == [7026, 7027, 7028]
        return {'mean_temporal_minus_early': .1, 'positive_delta_all_seeds': True}
    env = {'BASELINE_RESULTS': {seed: {'seed': seed} for seed in (7026, 7027, 7028)},
           'BASELINE_DIR': tmp_path, 'summarize_seeds': summary, 'write_stage': write_stage,
           'sha256_file': lambda p: 'checked-input', 'read_stage': None, 'json': json}
    exec(code_cells()['baseline_summary'], env)
    assert 'mean_temporal_minus_early' in capsys.readouterr().out
    assert (tmp_path / 'summary/complete').is_file()
