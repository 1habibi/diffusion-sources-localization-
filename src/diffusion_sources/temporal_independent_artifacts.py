"""Authenticated state for the independent, no-training evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import sys
from importlib import metadata
import numpy as np
import torch
import yaml

from .dataset import load_graph_archive
from .generation import graph_from_config
from .temporal_pilot_artifacts import read_stage, write_stage, sha256_file
from .temporal_pilot_cli import _identity, _code_identity, _comparable
from .temporal_early_baseline import POLICY

SEEDS = (7026, 7027, 7028)
DATA_FILES = ('graph.npz', 'independent_holdout.npz', 'config.yaml', 'generation_summary.json')


@dataclass(frozen=True)
class IndependentPaths:
    repo: Path
    reference: Path
    snapshot_holdout: Path
    pilot: Path
    baseline: Path
    runs: dict[int, Path]
    data: Path
    reports: Path
    generation_config: Path


def metadata_seed_union(paths: list[Path]) -> set[int]:
    result = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            arrays = [archive[name] for name in ('simulation_seeds', 'observation_seeds')]
            for values in arrays:
                if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
                        or np.any(values < 0) or len(np.unique(values)) != len(values)):
                    raise ValueError(f'Invalid seed metadata: {path}')
            if len(arrays[0]) != len(arrays[1]):
                raise ValueError(f'Unaligned seed metadata: {path}')
            result.update(int(v) for values in arrays for v in values)
    return result


def assert_disjoint_paths(outputs: list[Path], protected: list[Path]) -> None:
    resolved_outputs = [Path(p).resolve() for p in outputs]
    resolved_protected = [Path(p).resolve() for p in protected]
    for i, out in enumerate(resolved_outputs):
        for other in resolved_protected + resolved_outputs[:i]:
            if out == other or out in other.parents or other in out.parents:
                raise ValueError('output/input overlap')


def load_yaml(path: Path) -> dict:
    result = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'Invalid config: {path}')
    return result


def topology(path: Path) -> dict:
    graph_id, graph = load_graph_archive(path)
    return _graph_topology(graph_id, graph)


def _graph_topology(graph_id, graph) -> dict:
    nodes = sorted(graph.nodes())
    if nodes != list(range(len(nodes))) or graph.is_directed():
        raise ValueError('Topology requires contiguous labeled undirected nodes.')
    # Authentication only: historical feature/cache fingerprints stay untouched.
    edges = np.asarray(sorted((min(u, v), max(u, v)) for u, v in graph.edges()),
                       dtype='<i8').reshape(-1, 2)
    return {'id': graph_id, 'nodes': len(nodes), 'edges': hashlib.sha256(edges.tobytes()).hexdigest(),
            'policy': 'canonical-labelled-undirected-v1'}


def _runtime_versions() -> dict:
    return {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            **{name: metadata.version(name) for name in ('torch-geometric', 'networkx', 'scipy', 'PyYAML')}}


def reference_archives(paths: IndependentPaths) -> list[Path]:
    return [paths.reference / f'{s}.npz' for s in ('train', 'validation', 'test')] + [
        paths.snapshot_holdout / 'final_holdout.npz']


def _historical(root: Path, stage: str) -> tuple[dict, dict]:
    manifest = json.loads((root / stage / 'manifest.json').read_text(encoding='utf-8'))
    identity = manifest['identity']
    manifest, payload = read_stage(root, stage, identity)
    code = identity.get('code', identity.get('current_inputs_code', {}).get('code'))
    if code is None or not code.get('source_hashes'):
        raise ValueError(f'Missing historical code identity: {stage}')
    for name, digest in code['source_hashes'].items():
        if Path(name).name != name or sha256_file(Path(__file__).parent / name) != digest:
            raise ValueError(f'Historical code changed: {name}')
    return identity, payload


def _freeze_identity(paths: IndependentPaths) -> dict:
    if set(paths.runs) != set(SEEDS):
        raise ValueError('Requires all three frozen seeds.')
    assert_disjoint_paths([paths.data, paths.reports], [paths.repo, paths.reference,
        paths.snapshot_holdout, paths.pilot, paths.baseline, *paths.runs.values()])
    cfg = load_yaml(paths.generation_config)
    raw = Path(cfg['graph']['path'])
    raw = raw if raw.is_absolute() else paths.repo / raw
    archives = reference_archives(paths)
    metadata_seed_union(archives)  # Target-blind schema validation.
    frozen_topology = topology(paths.reference / 'graph.npz')
    raw_id, raw_graph = graph_from_config({**cfg['graph'], 'path': str(raw)})
    if _graph_topology(raw_id, raw_graph) != frozen_topology:
        raise ValueError('Raw/reference graph topology mismatch.')
    selection_id, selection = _historical(paths.pilot, 'select')
    if selection['beta'] != .5 or selection['n'] != 540:
        raise ValueError('Locked beta/train selection mismatch.')
    selected_now = _identity(paths.reference, paths.runs[7026], ('train',))
    for key in ('inputs', 'distance_cache', 'policy', 't1', 'rng'):
        if selection_id[key] != selected_now[key]:
            raise ValueError(f'Selection input mismatch: {key}')
    optional = paths.runs[7026] / 'train_predictions.csv'
    if selection_id.get('train_predictions') != (sha256_file(optional) if optional.is_file() else None):
        raise ValueError('Selection train predictions changed.')
    old = {'select': sha256_file(paths.pilot / 'select/payload.json')}
    current_runs = {}
    baseline_ids = {}
    primary_mask = None
    comparable = None
    for seed, run in sorted(paths.runs.items()):
        run_cfg = load_yaml(run / 'config.yaml')
        if run_cfg['training']['seed'] != seed:
            raise ValueError('Frozen run seed mismatch.')
        comp = _comparable(run_cfg)
        if comparable is not None and comp != comparable:
            raise ValueError('Frozen model configs are not comparable.')
        comparable = comp
        stage = 'validation' if seed == 7026 else f'seed_{seed}'
        saved_id, saved = _historical(paths.pilot, stage)
        now = _identity(paths.reference, run, ('validation',))
        for key in ('inputs', 'distance_cache', 'policy', 't1', 'rng'):
            if saved_id[key] != now[key]:
                raise ValueError(f'Historical input mismatch: {seed}/{key}')
        metrics_hash = sha256_file(run / 'metrics.json')
        if (saved_id['saved_metrics'] != metrics_hash or saved_id['beta'] != .5
                or saved_id['selection_hash'] != old['select']
                or saved['beta'] != .5 or saved['seed'] != seed
                or saved['baseline']['all']['n'] != 1998 or saved['temporal']['all']['n'] != 1998):
            raise ValueError('Historical metrics/beta/seed mismatch.')
        metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
        expected = metrics['validation_prediction_metrics']['joint_estimated_k']['all']['f1']
        if not np.isfinite(expected) or abs(expected - saved['baseline']['all']['f1']) > 1e-6:
            raise ValueError('Historical saved control F1 mismatch.')
        primary_mask = saved['early_mask_hash'] if primary_mask is None else primary_mask
        if primary_mask != saved['early_mask_hash']:
            raise ValueError('Historical early masks differ.')
        report_hash = sha256_file(paths.pilot / stage / 'payload.json')
        base_id, base = _historical(paths.baseline, f'seed_{seed}')
        base_now = base_id['current_inputs_code']
        if (any(base_now[k] != now[k] for k in ('inputs', 'distance_cache', 'policy', 't1', 'rng'))
                or base_id['beta'] != .5 or base_id['policy'] != POLICY
                or base_id['source_report'] != report_hash or base_id['selection'] != old['select']
                or base['seed'] != seed or base['beta'] != .5 or base['all']['n'] != 1998
                or base['early_mask_hash'] != primary_mask):
            raise ValueError('Historical early baseline identity mismatch.')
        for field, model in [('snapshot_f1', 'baseline'), ('temporal_f1', 'temporal')]:
            actual = base['all'][field]
            if not np.isfinite(actual) or abs(actual - saved[model]['all']['f1']) > 1e-6:
                raise ValueError('Historical early baseline F1 mismatch.')
        baseline_ids[str(seed)] = sha256_file(paths.baseline / f'seed_{seed}/payload.json')
        old[stage] = report_hash
        current_runs[str(seed)] = {'identity': now, 'metrics': metrics_hash}
    read_stage(paths.baseline, 'summary', {'seed_payloads': baseline_ids})
    gen_path = paths.generation_config.resolve()
    return {'runs': current_runs, 'historical_pilot': old, 'historical_baseline': baseline_ids,
            'historical_summary': sha256_file(paths.baseline / 'summary/payload.json'),
            'reference_archives': {str(p.resolve()): sha256_file(p) for p in archives},
            'generation_config': sha256_file(gen_path), 'raw_graph': sha256_file(raw),
            'topology': frozen_topology, 'code': _code_identity(),
            'versions': _runtime_versions(),
            'paths': {'data': str(paths.data.resolve()), 'reports': str(paths.reports.resolve())},
            'protocol': {'beta': .5, 'seeds': list(SEEDS), 't1': 1, 'max_steps': 3,
                         'dataset_seed': 4007026, 'n': 1998,
                         'bootstrap': {'repetitions': 2000, 'seed': 9282026},
                         'primary_min_delta': .02, 'k_min_delta': -.02}}


def stage_identity(paths: IndependentPaths, stage: str) -> dict:
    if stage == 'freeze':
        return _freeze_identity(paths)
    verify_freeze(paths)
    seal_id = {'freeze': sha256_file(paths.reports / 'freeze/payload.json'),
               'files': {name: sha256_file(paths.data / name) for name in DATA_FILES}}
    if stage == 'seal':
        return seal_id
    verify_seal(paths)
    return {'freeze': seal_id['freeze'], 'seal': sha256_file(paths.reports / 'seal/payload.json'),
            'dataset': seal_id['files']['independent_holdout.npz']}


def verify_freeze(paths: IndependentPaths) -> dict:
    return read_stage(paths.reports, 'freeze', _freeze_identity(paths))[1]


def freeze_inputs(paths: IndependentPaths) -> dict:
    identity = _freeze_identity(paths)
    if (paths.reports / 'freeze').exists():
        return read_stage(paths.reports, 'freeze', identity)[1]
    payload = {'beta': .5, 'seeds': list(SEEDS), 'role': 'parameter_artifact_freeze', 'identity': identity}
    write_stage(paths.reports, 'freeze', {'identity': identity}, payload)
    print('Frozen checkpoints/configs authenticated; independent labels unopened.', flush=True)
    return payload


def verify_seal(paths: IndependentPaths) -> dict:
    if not (paths.reports / 'seal/complete').is_file():
        raise ValueError('Dataset is not sealed.')
    return read_stage(paths.reports, 'seal', stage_identity(paths, 'seal'))[1]


def verify_opened(paths: IndependentPaths) -> dict:
    if not (paths.reports / 'opened/complete').is_file():
        raise ValueError('Independent dataset is not opened.')
    return read_stage(paths.reports, 'opened', stage_identity(paths, 'opened'))[1]
