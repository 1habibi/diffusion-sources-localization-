"""Staged, no-training temporal-v3 paired pilot (development data only)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import load_graph_archive
from .metrics import set_metrics
from .temporal_pilot_artifacts import sha256_file, write_stage, read_stage
from .temporal_replay import POLICY_VERSION
from .temporal_scoring import collect_records, correct_sources, select_beta
from .temporal_statistics import evaluate_pairs, primary_gate, confirmation_gate

GRID = [0., .1, .25, .5, 1.]


def guard_output(data_dir: Path, run_dir: Path, output_dir: Path) -> None:
    target = Path(output_dir).resolve()
    for path in (data_dir, run_dir):
        protected = Path(path).resolve()
        if target == protected or target in protected.parents or protected in target.parents:
            raise ValueError('Output directory overlaps protected input data/run directory.')


def _load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def _require_seed(run_dir: Path, seed: int):
    config = _load_config(run_dir / 'config.yaml')
    if config['training']['seed'] != seed:
        raise ValueError(f'Requires frozen seed {seed}.')
    return config


def _code_identity():
    package = Path(__file__).parent
    hashes = {p.name: sha256_file(p) for p in sorted(package.glob('*.py'))}
    revision = subprocess.run(['git', '-C', str(package.parent.parent), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    return {'revision': revision.stdout.strip() if revision.returncode == 0 else None, 'source_hashes': hashes}


def _identity(data: Path, run: Path, splits: tuple[str, ...]):
    paths = {'graph': data / 'graph.npz', 'generation_config': data / 'config.yaml',
             'model_config': run / 'config.yaml', 'checkpoint': run / 'best_model.pt'}
    paths.update({split: data / f'{split}.npz' for split in splits})
    return {'inputs': {key: sha256_file(path) for key, path in paths.items()},
            'code': _code_identity(), 'policy': POLICY_VERSION,
            't1': 1, 'rng': [5000000, 'observation_seed', 1]}


def _resume(root, stage, identity):
    if (root / stage).exists():
        return read_stage(root, stage, identity)[1]
    if list(root.glob(f'.{stage}-*')):
        raise ValueError(f'partial stage {stage}; use a new output directory.')
    return None


def _save(root, stage, identity, payload, device):
    write_stage(root, stage, {'identity': identity, 'device': str(device),
                             'torch_version': torch.__version__, 'numpy_version': np.__version__}, payload)


def smoke(data_dir: Path, run_dir: Path, output_dir: Path, device: torch.device) -> dict:
    guard_output(data_dir, run_dir, output_dir)
    identity = _identity(data_dir, run_dir, ('train',))
    identity['indices'] = list(range(6))
    previous = _resume(output_dir, 'smoke', identity)
    if previous is not None:
        return previous
    started = time.perf_counter()
    records = collect_records(data_dir, run_dir, 'train', list(range(6)), device)
    identical = all(correct_sources(r.candidates, 0.) == r.candidates.baseline_sources for r in records)
    if not identical:
        raise ValueError('beta=0 differs from frozen model.')
    result = {'n': len(records), 'beta_zero_identity': identical,
              'duration_seconds': time.perf_counter() - started, 'exploratory': True}
    _save(output_dir, 'smoke', identity, result, device)
    return result


def check_train_balance(config: dict, archive) -> None:
    simulation, observation = config['simulation'], config['observation']
    conditions = list(itertools.product(simulation['distance_ranges'], simulation['probabilities'],
                                        observation['fractions'], simulation['source_counts']))
    if len(conditions) != 54 or len(archive['source_counts']) < 540:
        raise ValueError('train balance requires 540 examples and 54 original conditions.')
    unique = {(str(dr), p, f, k) for dr, p, f, k in conditions}
    if len(unique) != 54:
        raise ValueError('train balance has duplicate conditions.')
    for i in range(540):
        _, p, f, k = conditions[i % 54]
        for key, expected in [('source_counts', k), ('probabilities', p), ('observation_fractions', f)]:
            if not np.isclose(archive[key][i], expected, atol=1e-8, rtol=0):
                raise ValueError(f'train balance mismatch at example {i}: {key}')


def _selection_identity(data, run):
    identity = _identity(data, run, ('train',))
    identity.update(indices=list(range(540)), grid=GRID)
    optional = run / 'train_predictions.csv'
    identity['train_predictions'] = sha256_file(optional) if optional.is_file() else None
    return identity


def _check_train_predictions(records, path: Path) -> str:
    if not path.is_file():
        return 'unavailable: full train F1 is not compared with subset F1'
    with path.open(newline='', encoding='utf-8') as stream:
        saved = {int(row['example']): float(row['f1']) for row in csv.DictReader(stream)
                 if row['method'] == 'joint_estimated_k'}
    for record in records:
        actual = set_metrics(record.true_sources, record.candidates.baseline_sources)['f1']
        if record.index not in saved or abs(actual - saved[record.index]) > 1e-6:
            raise ValueError(f'train baseline predictions differ at example {record.index}')
    return 'matched saved per-example predictions'


def select(data_dir: Path, run_dir: Path, output_dir: Path, device: torch.device) -> dict:
    guard_output(data_dir, run_dir, output_dir)
    _require_seed(run_dir, 7026)
    identity = _selection_identity(data_dir, run_dir)
    previous = _resume(output_dir, 'select', identity)
    if previous is not None:
        return previous
    with np.load(data_dir / 'train.npz', allow_pickle=False) as archive:
        check_train_balance(_load_config(data_dir / 'config.yaml'), archive)
    records = collect_records(data_dir, run_dir, 'train', list(range(540)), device)
    note = _check_train_predictions(records, run_dir / 'train_predictions.csv')
    beta, table = select_beta(records, GRID)
    result = {'beta': beta, 'train_f1': {str(k): v for k, v in table.items()},
              'n': len(records), 'train_control_check': note, 'exploratory': True}
    _save(output_dir, 'select', identity, result, device)
    return result


def _selection(data, run, root):
    return read_stage(root, 'select', _selection_identity(data, run))[1]


def _validation_identity(data, run, root, beta):
    identity = _identity(data, run, ('validation',))
    identity.update(beta=beta, selection_hash=sha256_file(root / 'select' / 'payload.json'),
                    saved_metrics=sha256_file(run / 'metrics.json'),
                    bootstrap={'repetitions': 2000, 'seed': 9282026})
    return identity


def _evaluate(data, run, beta, device, primary):
    with np.load(data / 'validation.npz', allow_pickle=False) as archive:
        if len(archive['source_counts']) != 1998:
            raise ValueError('Full validation must contain 1998 examples.')
    records = collect_records(data, run, 'validation', list(range(1998)), device)
    _, graph = load_graph_archive(data / 'graph.npz')
    result = evaluate_pairs(records, beta, graph)
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    expected = metrics['validation_prediction_metrics']['joint_estimated_k']['all']['f1']
    if (not np.isfinite(expected) or abs(result['baseline']['all']['f1'] - expected) > 1e-6
            or (primary and abs(expected - .362996329663) > 1e-6)):
        raise ValueError('baseline F1 does not match saved S1b checkpoint metrics.')
    masks = [(r.index, r.candidates.candidate_ids, r.candidates.early_observed, r.early_empty) for r in records]
    result['early_mask_hash'] = hashlib.sha256(json.dumps(masks).encode()).hexdigest()
    result['seed'] = _load_config(run / 'config.yaml')['training']['seed']
    return result


def validate(data_dir: Path, run_dir: Path, output_dir: Path, device: torch.device) -> dict:
    guard_output(data_dir, run_dir, output_dir)
    _require_seed(run_dir, 7026)
    beta = _selection(data_dir, run_dir, output_dir)['beta']
    identity = _validation_identity(data_dir, run_dir, output_dir, beta)
    previous = _resume(output_dir, 'validation', identity)
    if previous is not None:
        return previous
    result = _evaluate(data_dir, run_dir, beta, device, primary=True)
    result['gate'] = primary_gate(result)
    _save(output_dir, 'validation', identity, result, device)
    return result


def _comparable(config):
    return {'model': config['model'], 'loss': config.get('loss', {}),
            'data': {key: value for key, value in config['data'].items() if key not in ('directory', 'distance_cache')},
            'training': {key: value for key, value in config['training'].items()
                         if key not in ('seed', 'device', 'resume', 'resume_from')}}


def confirm(data_dir: Path, run_dir: Path, output_dir: Path,
             repeat_runs: list[Path], device: torch.device) -> dict:
    if len(repeat_runs) != 2:
        raise ValueError('Exactly two frozen repeat runs are required.')
    guard_output(data_dir, run_dir, output_dir)
    primary_config = _require_seed(run_dir, 7026)
    beta = _selection(data_dir, run_dir, output_dir)['beta']
    primary = read_stage(output_dir, 'validation', _validation_identity(data_dir, run_dir, output_dir, beta))[1]
    if not primary_gate(primary)['passed']:
        raise ValueError('Primary gate failed; confirmation is not allowed.')
    runs = {}
    for run in repeat_runs:
        guard_output(data_dir, run, output_dir)
        config = _load_config(run / 'config.yaml')
        seed = config['training']['seed']
        if seed not in (7027, 7028) or seed in runs or _comparable(config) != _comparable(primary_config):
            raise ValueError('Frozen repeats must be comparable unique seeds 7027 and 7028.')
        runs[seed] = run
    repeats, hashes = [], {}
    for seed, run in sorted(runs.items()):
        identity = _validation_identity(data_dir, run, output_dir, beta)
        stage = f'seed_{seed}'
        result = _resume(output_dir, stage, identity)
        if result is None:
            result = _evaluate(data_dir, run, beta, device, primary=False)
            if result['early_mask_hash'] != primary['early_mask_hash']:
                raise ValueError('Early masks differ between frozen seeds.')
            _save(output_dir, stage, identity, result, device)
        if result['beta'] != beta or result['early_mask_hash'] != primary['early_mask_hash']:
            raise ValueError('Repeated beta/masks differ from primary.')
        repeats.append(result)
        hashes[stage] = sha256_file(output_dir / stage / 'payload.json')
    identity = {'primary': sha256_file(output_dir / 'validation' / 'payload.json'), 'repeats': hashes}
    previous = _resume(output_dir, 'confirmation', identity)
    if previous is not None:
        return previous
    result = {**confirmation_gate(primary, repeats), 'beta': beta, 'exploratory': True,
              'per_seed_delta': {str(r['seed']): r['delta_f1'] for r in [primary, *repeats]}}
    _save(output_dir, 'confirmation', identity, result, device)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'select', 'validate', 'confirm'])
    for name in ('data-dir', 'run-dir', 'output-dir'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--repeat-run', action='append', type=Path, default=[])
    args = parser.parse_args(argv)
    try:
        params = (args.data_dir, args.run_dir, args.output_dir)
        device = torch.device(args.device)
        if args.stage == 'confirm':
            result = confirm(*params, args.repeat_run, device)
        else:
            if args.repeat_run:
                raise ValueError('--repeat-run is only allowed for confirm.')
            result = {'smoke': smoke, 'select': select, 'validate': validate}[args.stage](*params, device)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))
        print(f'Saved under: {args.output_dir.resolve()}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Temporal pilot error: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
