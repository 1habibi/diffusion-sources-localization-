from pathlib import Path
import numpy as np
import pytest
import torch
import yaml


def test_output_cannot_overlap_inputs(tmp_path):
    from diffusion_sources.temporal_pilot_cli import guard_output
    data, run = tmp_path / 'data', tmp_path / 'run'
    for out in (data, data / 'child', run / 'child', tmp_path):
        with pytest.raises(ValueError):
            guard_output(data, run, out)
    guard_output(data, run, tmp_path / 'pilot')


def test_balance_requires_complete_original_cycles():
    from diffusion_sources.temporal_pilot_cli import check_train_balance
    cfg = {'simulation': {'distance_ranges': [{'min': 1, 'max': 2}, {'min': 3, 'max': 5}],
           'probabilities': [.01, .02, .03], 'source_counts': [1, 2, 3]},
           'observation': {'fractions': [1., .75, .5]}}
    import itertools
    c = list(itertools.product([0, 1], [.01, .02, .03], [1., .75, .5], [1, 2, 3])) * 10
    a = {'source_counts': np.array([r[3] for r in c]), 'probabilities': np.array([r[1] for r in c]),
         'observation_fractions': np.array([r[2] for r in c])}
    check_train_balance(cfg, a)
    a['source_counts'][0] = 3
    with pytest.raises(ValueError, match='balance'):
        check_train_balance(cfg, a)


def test_select_rejects_non_primary_without_validation_read(temporal_dataset, tmp_path):
    from diffusion_sources.temporal_pilot_cli import select
    data, _ = temporal_dataset
    run = tmp_path / 'run'; run.mkdir()
    (run / 'config.yaml').write_text(yaml.safe_dump({'training': {'seed': 13}}))
    with pytest.raises(ValueError, match='7026'):
        select(data, run, tmp_path / 'out', torch.device('cpu'))
    assert not (tmp_path / 'out').exists()


def test_validate_requires_locked_selection(tmp_path):
    from diffusion_sources.temporal_pilot_cli import validate
    data, run, out = (tmp_path / p for p in ('data', 'run', 'out'))
    data.mkdir(); run.mkdir()
    (run / 'config.yaml').write_text(yaml.safe_dump({'training': {'seed': 7026}}))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(data, run, out, torch.device('cpu'))
    assert not out.exists()


def test_confirm_rejects_missing_primary_gate(tmp_path):
    from diffusion_sources.temporal_pilot_cli import confirm
    with pytest.raises(ValueError):
        confirm(tmp_path / 'data', tmp_path / 'run', tmp_path / 'out', [], torch.device('cpu'))


def test_cli_help_and_error_return():
    from diffusion_sources.temporal_pilot_cli import main
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0
    code = main(['smoke', '--data-dir', 'missing-data', '--run-dir', 'missing-run',
                  '--output-dir', 'missing-out'])
    assert code != 0
