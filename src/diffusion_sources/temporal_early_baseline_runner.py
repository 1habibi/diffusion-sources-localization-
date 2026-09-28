"""Read-only replay/inference for the early-only ranking ablation, no tuning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .temporal_early_baseline import POLICY, evaluate_early_baseline
from .temporal_pilot_artifacts import read_stage, sha256_file, write_stage
from .temporal_pilot_cli import _identity, _load_config, _resume, guard_output
from .temporal_scoring import collect_records


def _old_stage(root: Path, stage: str):
    """Authenticate artifacts and the historical source modules, not a new git SHA.

    Adding this analysis module/notebook changes HEAD and the package file list,
    but must not require rerunning the pilot. Every module hashed by the pilot
    must still exist unchanged. Inputs are checked separately against current files.
    """
    manifest = json.loads((root / stage / 'manifest.json').read_text(encoding='utf-8'))
    manifest, payload = read_stage(root, stage, manifest['identity'])
    package = Path(__file__).parent
    for name, digest in manifest['identity']['code']['source_hashes'].items():
        if Path(name).name != name or sha256_file(package / name) != digest:
            raise ValueError(f'Historical pilot code changed: {name}')
    return manifest, payload


def run_seed(data_dir: Path, run_dir: Path, pilot_dir: Path,
             output_dir: Path, device: torch.device) -> dict:
    data_dir, run_dir, pilot_dir, output_dir = map(Path, (data_dir, run_dir, pilot_dir, output_dir))
    guard_output(data_dir, run_dir, output_dir)
    guard_output(pilot_dir, pilot_dir, output_dir)
    seed = _load_config(run_dir / 'config.yaml')['training']['seed']
    if seed not in (7026, 7027, 7028):
        raise ValueError('Requires frozen seed 7026/7027/7028.')
    source_stage = 'validation' if seed == 7026 else f'seed_{seed}'
    print(f'[early baseline {seed}] Проверка сохранённых отчётов и input hashes...', flush=True)
    _, selection = _old_stage(pilot_dir, 'select')
    saved_manifest, saved = _old_stage(pilot_dir, source_stage)
    old_identity = saved_manifest['identity']
    current = _identity(data_dir, run_dir, ('validation',))
    for key in ('inputs', 'distance_cache', 'policy', 't1', 'rng'):
        if old_identity[key] != current[key]:
            raise ValueError(f'Historical input/policy mismatch: {key}')
    if old_identity['saved_metrics'] != sha256_file(run_dir / 'metrics.json'):
        raise ValueError('Historical input mismatch: saved metrics')
    selection_hash = sha256_file(pilot_dir / 'select' / 'payload.json')
    beta = selection['beta']
    if (old_identity['selection_hash'] != selection_hash or old_identity['beta'] != beta
            or saved['beta'] != beta or saved['seed'] != seed
            or saved['baseline']['all']['n'] != 1998 or saved['temporal']['all']['n'] != 1998):
        raise ValueError('Historical beta/seed/example count mismatch.')
    _, primary = _old_stage(pilot_dir, 'validation')
    if saved['early_mask_hash'] != primary['early_mask_hash']:
        raise ValueError('Historical masks differ between seeds.')
    identity = {'current_inputs_code': current, 'policy': POLICY, 'beta': beta,
                'source_report': sha256_file(pilot_dir / source_stage / 'payload.json'),
                'selection': selection_hash, 'bootstrap': {'seed': 9282026, 'repetitions': 2000}}
    stage = f'seed_{seed}'
    previous = _resume(output_dir, stage, identity)
    if previous is not None:
        print(f'[early baseline {seed}] Возобновление: сохранённый результат.', flush=True)
        print(json.dumps({k: v for k, v in previous.items() if k != 'rows'}, ensure_ascii=False, indent=2), flush=True)
        return previous
    print(f'[early baseline {seed}] Только validation1998: replay + frozen inference, без обучения.', flush=True)
    records = collect_records(data_dir, run_dir, 'validation', list(range(1998)), device)
    if [r.index for r in records] != list(range(1998)):
        raise ValueError('Missing, duplicate or reordered validation examples.')
    masks = [(r.index, r.candidates.candidate_ids, r.candidates.early_observed, r.early_empty) for r in records]
    mask_hash = hashlib.sha256(json.dumps(masks).encode()).hexdigest()
    if mask_hash != saved['early_mask_hash']:
        raise ValueError('Replayed early masks do not match pilot.')
    report = evaluate_early_baseline(records, beta)
    for model, field in [('baseline', 'snapshot_f1'), ('temporal', 'temporal_f1')]:
        expected = saved[model]['all']['f1']
        if not np.isfinite(expected) or abs(report['all'][field] - expected) > 1e-6:
            raise ValueError(f'Replayed {model} F1 does not match pilot.')
    report.update(seed=seed, early_mask_hash=mask_hash)
    write_stage(output_dir, stage, {'identity': identity, 'device': str(device),
                                   'torch_version': torch.__version__, 'numpy_version': np.__version__}, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False, indent=2), flush=True)
    print(f'[early baseline {seed}] Сохранено: {output_dir / stage}', flush=True)
    return report


def summarize_seeds(reports: list[dict]) -> dict:
    if (len(reports) != 3 or {r['seed'] for r in reports} != {7026, 7027, 7028}
            or len({r['beta'] for r in reports}) != 1
            or len({r['early_mask_hash'] for r in reports}) != 1
            or any(r['all']['n'] != 1998 for r in reports)):
        raise ValueError('Requires three distinct frozen seeds with matching beta/masks/full validation.')
    fields = ('snapshot_f1', 'early_expected_f1', 'temporal_f1', 'temporal_minus_early')
    values = {key: np.array([r['all'][key] for r in reports], dtype=float) for key in fields}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Non-finite seed metrics.')
    return {'exploratory': True, 'early_baseline_uses_gcn_count': True,
            'means': {key: float(v.mean()) for key, v in values.items()},
            'sample_sd': {key: float(v.std(ddof=1)) for key, v in values.items()},
            'mean_temporal_minus_early': float(values['temporal_minus_early'].mean()),
            'positive_delta_all_seeds': bool((values['temporal_minus_early'] > 0).all()),
            'positive_ci_lower_all_seeds': all(r['f1_ci'][0] > 0 for r in reports),
            'per_seed': {str(r['seed']): {'all': r['all'], 'f1_ci': r['f1_ci']} for r in reports},
            'interpretation': 'Ranking contribution only; shared learned count; same exploratory validation split.'}
