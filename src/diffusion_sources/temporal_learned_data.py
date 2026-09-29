"""Validated ragged tables and append-only authenticated numeric caches."""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import platform
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from .temporal_learned_features import FEATURE_NAMES
from .temporal_pilot_artifacts import sha256_file


@dataclass
class CandidateTable:
    features: np.ndarray
    candidate_ids: np.ndarray
    probabilities: np.ndarray
    early_flags: np.ndarray
    labels: np.ndarray
    offsets: np.ndarray
    indices: np.ndarray
    predicted_counts: np.ndarray
    true_counts: np.ndarray
    early_empty: np.ndarray
    snapshot_sources: tuple[frozenset[int], ...]
    early_mask_hash: str


def _integers(a, shape, name):
    a = np.asarray(a)
    if a.shape != shape or a.dtype.kind not in 'iu':
        raise ValueError(f'{name}: expected integer array of shape {shape}')
    return a


def validate_table(t):
    x = np.asarray(t.features)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    r, n = len(x), len(t.indices)
    if not n or not r:
        raise ValueError('Empty table')
    offsets = _integers(t.offsets, (n+1,), 'offsets')
    if offsets[0] != 0 or offsets[-1] != r or (np.diff(offsets) <= 0).any():
        raise ValueError('Invalid ragged offsets')
    ids = _integers(t.candidate_ids, (r,), 'candidate IDs')
    indices = _integers(t.indices, (n,), 'indices')
    if (ids < 0).any() or (indices < 0).any() or len(np.unique(indices)) != n:
        raise ValueError('Invalid or duplicate indices')
    for name in ('predicted_counts', 'true_counts'):
        a = _integers(getattr(t, name), (n,), name)
        if ((a < 1) | (a > 3)).any():
            raise ValueError(f'Invalid {name}')
    for name, shape in (('labels', (r,)), ('early_flags', (r,)), ('early_empty', (n,))):
        a = np.asarray(getattr(t, name))
        if a.shape != shape or a.dtype.kind not in 'biuf' or not np.isin(a, [0, 1]).all():
            raise ValueError(f'Invalid binary {name}')
    p = np.asarray(t.probabilities)
    if p.shape != (r,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    if not np.array_equal(x[:, 0], p) or not np.array_equal(x[:, 1], t.early_flags):
        raise ValueError('Feature/probability/early alignment mismatch')
    if not np.allclose(x[:, 5], p * t.early_flags, rtol=0, atol=1e-15) or ((x < 0) | (x > 1)).any():
        raise ValueError('Invalid observable feature ranges or interaction')
    if len(t.snapshot_sources) != n or not re.fullmatch('[0-9a-f]{64}', t.early_mask_hash):
        raise ValueError('Invalid source sets or full-mask digest')
    for j, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        pool = set(map(int, ids[a:b]))
        sources = t.snapshot_sources[j]
        if len(pool) != b-a or not set(sources) <= pool or len(sources) != min(t.predicted_counts[j], b-a):
            raise ValueError('Invalid candidate pool or snapshot cardinality')
        if np.sum(t.labels[a:b]) != t.true_counts[j] or not 0 < t.true_counts[j] < b-a:
            raise ValueError('Labels/count mismatch or missing positive/negative class')
        if t.early_empty[j] and np.any(t.early_flags[a:b]):
            raise ValueError('Early-empty flag contradicts candidate observation')


def subset_table(t, positions):
    validate_table(t)
    pos = np.asarray(positions)
    if pos.ndim != 1 or not len(pos) or pos.dtype.kind not in 'iu' or len(np.unique(pos)) != len(pos) or ((pos < 0) | (pos >= len(t.indices))).any():
        raise ValueError('Subset must contain distinct whole-cascade positions')
    chunks = [np.arange(t.offsets[j], t.offsets[j+1]) for j in pos]
    rows = np.concatenate(chunks)
    result = CandidateTable(
        *[getattr(t, name)[rows].copy() for name in ('features', 'candidate_ids', 'probabilities', 'early_flags', 'labels')],
        np.r_[0, np.cumsum([len(c) for c in chunks])],
        *[getattr(t, name)[pos].copy() for name in ('indices', 'predicted_counts', 'true_counts', 'early_empty')],
        tuple(t.snapshot_sources[j] for j in pos), t.early_mask_hash)
    validate_table(result)
    return result


def guard_output(output, protected):
    out = Path(output).resolve()
    for path in protected:
        path = Path(path).resolve()
        if out == path or out in path.parents or path in out.parents:
            raise ValueError(f'Output overlaps protected input: {path}')


def cache_identity(data_dir, run_dir, split, indices, device):
    if split not in ('train', 'validation'):
        raise ValueError('Only train and validation are permitted')
    values = np.asarray(indices)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu' or (values < 0).any() or len(np.unique(values)) != len(values):
        raise ValueError('Invalid cache indices')
    data, run = Path(data_dir), Path(run_dir)
    files = {'graph': data/'graph.npz', 'generation': data/'config.yaml',
             'split': data/f'{split}.npz', 'config': run/'config.yaml', 'checkpoint': run/'best_model.pt'}
    packages = ('numpy', 'torch', 'torch-geometric', 'scikit-learn', 'networkx', 'scipy', 'PyYAML')
    runtime = {p: importlib.metadata.version(p) for p in packages}
    runtime['python'] = platform.python_version()
    device = torch.device(device)
    runtime.update(device=str(device), dtype='float64-features/float32-gcn',
                   cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' and torch.cuda.is_available() else None)
    source_names = ('temporal_learned_features.py', 'temporal_learned_data.py', 'temporal_learned_collect.py',
                    'temporal_replay.py', 'temporal_scoring.py', 'features.py', 'models.py', 'inference.py', 'dataset.py')
    source_dir = Path(__file__).parent
    sources = {p: hashlib.sha256((source_dir/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
               for p in source_names if (source_dir/p).exists()}
    return dict(schema_version=1, split=split, indices=list(map(int, values)),
                inputs={k: sha256_file(p) for k, p in files.items()}, runtime=runtime,
                sources=sources, feature_names=list(FEATURE_NAMES),
                early_policy='independent-bernoulli-t1-seedsequence-v1', t1=1, beta=.5)


def _target(root, stage):
    if not re.fullmatch('[a-zA-Z0-9_-]+', stage):
        raise ValueError('Invalid cache stage')
    return Path(root)/stage


def save_cache(root, stage, identity, table):
    validate_table(table)
    target = _target(root, stage)
    if target.exists():
        raise FileExistsError(f'Cache already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if list(target.parent.glob(f'.{stage}-*')):
        raise ValueError('Partial cache exists; choose a new output root')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{stage}-', dir=target.parent))
    arrays = {name: np.asarray(getattr(table, name)) for name in (
        'features', 'candidate_ids', 'probabilities', 'early_flags', 'labels',
        'offsets', 'indices', 'predicted_counts', 'true_counts', 'early_empty')}
    arrays['snapshot_ids'] = np.array([i for s in table.snapshot_sources for i in sorted(s)], dtype=np.int64)
    arrays['snapshot_offsets'] = np.r_[0, np.cumsum([len(s) for s in table.snapshot_sources])]
    np.savez_compressed(temporary/'cache.npz', **arrays)
    manifest = dict(schema_version=1, identity=identity, early_mask_hash=table.early_mask_hash,
                    file_hashes={'cache.npz': sha256_file(temporary/'cache.npz')})
    (temporary/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    (temporary/'complete').write_text('complete\n', encoding='utf-8')
    temporary.rename(target)


def load_cache(root, stage, identity):
    target = _target(root, stage)
    if not (target/'complete').is_file():
        raise ValueError('Incomplete or missing cache')
    try:
        manifest = json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        if manifest['schema_version'] != 1 or manifest['identity'] != identity or manifest['file_hashes'] != {'cache.npz': sha256_file(target/'cache.npz')}:
            raise ValueError('Cache identity or file hash mismatch')
        with np.load(target/'cache.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        offsets = _integers(arrays.pop('snapshot_offsets'), (len(arrays['indices'])+1,), 'snapshot offsets')
        ids = arrays.pop('snapshot_ids')
        if offsets[0] != 0 or offsets[-1] != len(ids) or (np.diff(offsets) <= 0).any() or ids.dtype.kind not in 'iu':
            raise ValueError('Invalid snapshot offsets/IDs')
        snapshots = tuple(frozenset(map(int, ids[a:b])) for a,b in zip(offsets[:-1], offsets[1:]))
        if any(len(s) != b-a for s,a,b in zip(snapshots, offsets[:-1], offsets[1:])):
            raise ValueError('Duplicate snapshot IDs')
        table = CandidateTable(**arrays, snapshot_sources=snapshots, early_mask_hash=manifest['early_mask_hash'])
        validate_table(table)
        return table
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid cache artifacts') from exc
