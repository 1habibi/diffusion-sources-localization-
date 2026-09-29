import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


def cells():
    path = Path(__file__).parents[2] / 'notebooks/colab_temporal_v3_independent_evaluation.ipynb'
    return {c['id']: ''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells']
            if c['cell_type'] == 'code'}


def test_seven_cells_and_independent_routing(capsys):
    source = cells()
    assert list(source) == ['setup', 'freeze', 'seal', 'open_7026', 'seed_7027', 'seed_7028', 'summary']
    calls = []
    paths = SimpleNamespace(reports=Path('/reports'), data=Path('/new'), generation_config=Path('/config'))
    def opened(paths, answer):
        if answer != 'OPEN_INDEPENDENT_HOLDOUT': raise ValueError('confirmation')
        calls.append('open')
    ns = {'PATHS': paths, 'torch': torch, 'DEVICE': 'cpu', 'json': json,
          'open_evaluation': opened, 'input': lambda _: 'OPEN_INDEPENDENT_HOLDOUT',
          'evaluate_seed': lambda p, seed, device: calls.append(seed),
          'generate_and_seal': lambda p: calls.append('seal') or {'n': 1998},
          'save_summary': lambda p: calls.append('disk-summary') or {'primary': {'passed': True}, 'secondary': {'passed': True}}}
    exec(source['seal'], ns)
    assert calls == ['seal']
    exec(source['open_7026'], ns)
    exec(source['seed_7027'], ns)
    exec(source['seed_7028'], ns)
    exec(source['summary'], ns)
    assert calls == ['seal', 'open', 7026, 7027, 7028, 'disk-summary']
    assert '1998' in capsys.readouterr().out


def test_wrong_confirmation_never_runs_model():
    source = cells()
    def opened(paths, answer): raise ValueError('confirmation')
    ns = {'PATHS': object(), 'torch': torch, 'DEVICE': 'cpu', 'input': lambda _: 'yes',
          'open_evaluation': opened, 'evaluate_seed': lambda *a: pytest.fail('Must not run')}
    with pytest.raises(ValueError, match='confirmation'):
        exec(source['open_7026'], ns)


def test_freeze_never_generates_or_opens(tmp_path, capsys):
    from diffusion_sources.temporal_independent_artifacts import IndependentPaths
    calls = []
    raw = tmp_path / 'data/raw/facebook_combined.txt.gz'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'existing raw graph')
    def frozen(paths):
        calls.append('freeze')
        assert paths.reference == Path('/content/drive/MyDrive/diffusion-sources/data/facebook_main')
        assert set(paths.runs) == {7026, 7027, 7028}
        return {'role': 'parameter_artifact_freeze'}
    def forbidden(*args): pytest.fail('Freeze must never run next stages')
    ns = {'Path': Path, 'REPO': tmp_path, 'json': json,
          'ARTIFACTS': SimpleNamespace(IndependentPaths=IndependentPaths, freeze_inputs=frozen),
          'GENERATION': SimpleNamespace(generate_and_seal=forbidden),
          'INFERENCE': SimpleNamespace(open_evaluation=forbidden, evaluate_seed=forbidden),
          'SUMMARY': SimpleNamespace(save_summary=forbidden)}
    exec(cells()['freeze'], ns)
    assert calls == ['freeze']
    assert 'dataset seed: 4007026' in capsys.readouterr().out


@pytest.mark.parametrize('case', ['fresh', 'restart', 'unavailable'])
def test_missing_raw_graph_provision_before_freeze(tmp_path, monkeypatch, case):
    from diffusion_sources.temporal_independent_artifacts import IndependentPaths
    from diffusion_sources import download
    raw = tmp_path / 'repo/data/raw/facebook_combined.txt.gz'
    drive_root = tmp_path / 'drive'
    calls = []
    if case == 'restart':
        manifest = drive_root / 'reports/runs/temporal_v3_independent_evaluation/v1/freeze/manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'identity': {'raw_graph': 'frozen-raw-sha'}}))
    def provision(url, output, **kwargs):
        calls.append(('raw', url, output, kwargs))
        if case == 'unavailable': raise OSError('network unavailable')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'raw graph boundary')
    monkeypatch.setattr(download, 'download_file', provision)
    def frozen(paths):
        assert raw.is_file(), 'Fresh clone requires raw graph before freeze'
        calls.append(('freeze',))
        return {'role': 'parameter_artifact_freeze'}
    def forbidden(*args): pytest.fail('No generation/open/inference')
    ns = {'Path': lambda value: drive_root if str(value) == '/content/drive/MyDrive/diffusion-sources' else Path(value),
          'REPO': tmp_path / 'repo', 'json': json,
          'ARTIFACTS': SimpleNamespace(IndependentPaths=IndependentPaths, freeze_inputs=frozen),
          'GENERATION': SimpleNamespace(generate_and_seal=forbidden),
          'INFERENCE': SimpleNamespace(open_evaluation=forbidden, evaluate_seed=forbidden),
          'SUMMARY': SimpleNamespace(save_summary=forbidden)}
    if case == 'unavailable':
        with pytest.raises(RuntimeError, match='facebook_combined.txt.gz'):
            exec(cells()['freeze'], ns)
        assert len(calls) == 1
    else:
        exec(cells()['freeze'], ns)
        assert calls[0][1:3] == (download.FACEBOOK_URL, raw)
        assert calls[0][3].get('expected_sha256') == ('frozen-raw-sha' if case == 'restart' else None)
        assert calls[1] == ('freeze',)


def test_setup_boundary_and_cache_refresh(tmp_path, monkeypatch, capsys):
    import sys
    import subprocess
    import importlib
    import types
    source = cells()
    mounts, commands, imports = [], [], []
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name == 'diffusion_sources' or name.startswith('diffusion_sources.')}
    saved_path = list(sys.path)
    monkeypatch.setitem(sys.modules, 'google.colab', SimpleNamespace(drive=SimpleNamespace(mount=lambda p: mounts.append(p))))
    monkeypatch.setenv('DIFFUSION_COLAB_REPO', str(tmp_path))
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: commands.append(args))
    # Import only the new module boundary; setup must never invoke its stages.
    monkeypatch.setattr(importlib, 'import_module', lambda name: imports.append(name) or
                        SimpleNamespace(__file__=str(tmp_path / 'src/diffusion_sources/artifacts.py')))
    monkeypatch.setitem(sys.modules, 'diffusion_sources.temporal_independent_summary', types.ModuleType('stale'))
    ns = {'REPORT_7026': 'stale', 'save_summary': 'stale'}
    try:
        exec(source['setup'], ns)
    finally:
        # This emulates a Colab kernel reset, not a reset of the whole pytest
        # process: preserve modules referenced by already-collected old tests.
        for name in list(sys.modules):
            if name == 'diffusion_sources' or name.startswith('diffusion_sources.'):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
    assert mounts and len(commands) == 3
    assert imports == [f'diffusion_sources.temporal_independent_{x}' for x in ('artifacts', 'generation', 'inference', 'summary')]
    assert 'REPORT_7026' not in ns and 'save_summary' not in ns
    assert all(command[0] == sys.executable for command in commands[1:])
    assert not any('-r' in command for command in commands)  # No requirements.txt in this repo.
    assert '--no-deps' in commands[-1]  # Keep Colab's preinstalled CUDA PyTorch.
    assert 'Kernel Python:' in capsys.readouterr().out
