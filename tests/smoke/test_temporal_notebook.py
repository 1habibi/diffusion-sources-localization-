import json
from pathlib import Path
import subprocess
import sys
import pytest

NOTEBOOK = Path(__file__).parents[2] / 'notebooks/colab_temporal_v3_pilot.ipynb'


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    result = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            source = ''.join(cell['source'])
            compile(source, cell['id'], 'exec')
            result[cell['id']] = source
    return result


def test_notebook_setup_uses_kernel_python_and_stops_on_error(monkeypatch):
    code = cells()['setup']
    calls = []
    def external(command, **kwargs):
        assert kwargs['check'] is True
        calls.append(command)
    monkeypatch.setattr(subprocess, 'run', external)
    env = {'REPO': Path(__file__).parents[2], 'REPOSITORY': 'unused', 'DEVICE': 'cpu', '__name__': '__test__'}
    exec(code, env)
    assert any(c[:3] == [sys.executable, '-m', 'pip'] for c in calls)
    assert env['diffusion_sources'].__file__
    def failing(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', failing)
    with pytest.raises(subprocess.CalledProcessError):
        exec(code, env)


def test_notebook_stage_cells_call_distinct_stages_without_training():
    c = cells()
    stages = []
    env = {'run_stage': lambda stage: stages.append(stage)}
    for cell in ('smoke', 'select', 'validate'):
        exec(c[cell], env)
    assert stages == ['smoke', 'select', 'validate']


def test_notebook_confirm_skipped_on_primary_failure(tmp_path):
    c = cells()
    (tmp_path / 'validation').mkdir()
    (tmp_path / 'validation' / 'metrics.json').write_text(json.dumps({'gate': {'passed': False}}))
    def forbidden(*args, **kwargs):
        raise AssertionError('confirmation must not run after failed gate')
    exec(c['confirm'], {'PILOT_DIR': tmp_path, 'RUN_ROOT': tmp_path, 'run_stage': forbidden,
                         'json': json})


def test_notebook_helper_invokes_checked_cli(monkeypatch, tmp_path):
    env = {'DATA_DIR': tmp_path / 'data', 'RUN_DIR': tmp_path / 'run',
           'PILOT_DIR': tmp_path / 'pilot', 'REPO': tmp_path, 'DEVICE': 'cpu'}
    commands = []
    for folder, names in [('data', ['graph.npz', 'config.yaml', 'train.npz']),
                          ('run', ['config.yaml', 'best_model.pt', 'metrics.json'])]:
        (tmp_path / folder).mkdir()
        for name in names:
            (tmp_path / folder / name).write_text('fixture')
    monkeypatch.setattr(subprocess, 'run', lambda command, **kw: commands.append((command, kw)))
    exec(cells()['paths'], env)
    env['run_stage']('smoke')
    cmd, kw = commands[-1]
    assert cmd[:5] == [sys.executable, '-u', '-m', 'diffusion_sources.temporal_pilot_cli', 'smoke']
    assert kw['check'] is True
