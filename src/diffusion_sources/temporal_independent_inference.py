"""Explicit, append-only opening and frozen independent inference; never trains."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import torch
from tqdm.auto import tqdm

from .dataset import load_graph_archive, load_pyg_split
from .inference import predict_joint
from .models import JointSourceCountGCN
from .temporal_replay import FIELDS, replay_early_mask
from .temporal_scoring import CandidateScores, PilotRecord, correct_sources, _readonly_feature_builder
from .temporal_statistics import evaluate_pairs
from .temporal_early_baseline import evaluate_early_baseline
from .temporal_pilot_artifacts import read_stage, write_stage, sha256_file
from .temporal_independent_artifacts import (IndependentPaths, SEEDS, load_yaml,
    verify_opened, verify_seal, stage_identity)


def open_evaluation(paths: IndependentPaths, confirmation: str) -> dict:
    if confirmation != 'OPEN_INDEPENDENT_HOLDOUT':
        raise ValueError('Explicit confirmation OPEN_INDEPENDENT_HOLDOUT required.')
    verify_seal(paths)
    identity = stage_identity(paths, 'opened')
    if (paths.reports / 'opened').exists():
        return verify_opened(paths)
    marker = {'evaluation_status': 'opened', 'evaluation_role': 'independent_confirmation',
              'confirmation': confirmation, 'identity': identity,
              'warning': 'Targets are authorized now; no retuning or replacement dataset.'}
    write_stage(paths.reports, 'opened', {'identity': identity}, marker)
    print('Independent evaluation OPENED; parameters remain frozen.', flush=True)
    return marker


def collect_independent_records(paths: IndependentPaths, seed: int,
                               device: torch.device) -> list[PilotRecord]:
    verify_opened(paths)  # Must precede every target-array access.
    if seed not in SEEDS:
        raise ValueError('Requires frozen seed 7026/7027/7028.')
    config = load_yaml(paths.runs[seed] / 'config.yaml')
    generation = load_yaml(paths.data / 'config.yaml')
    names, mc = config['data']['feature_names'], config['model']
    if (config['training']['seed'] != seed or mc.get('source_head_strategy', 'shared') != 'shared'
            or int(mc.get('input_dim', len(names))) != len(names)):
        raise ValueError('Frozen model seed/head/input dimension mismatch.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; choose an available device.')
    _, graph = load_graph_archive(paths.data / 'graph.npz')
    builder = _readonly_feature_builder(graph, config['data'])
    with np.load(paths.data / 'independent_holdout.npz', allow_pickle=False) as compressed:
        archive = {field: compressed[field] for field in FIELDS}
    if any(len(values) != 1998 for values in archive.values()):
        raise ValueError('Independent archive must contain exactly 1998 rows.')
    examples = load_pyg_split(paths.data / 'independent_holdout.npz', graph,
                             feature_names=names, feature_builder=builder,
                             progress_description='Loading independent holdout')
    if len(examples) != 1998:
        raise ValueError('Incomplete independent examples.')
    model = JointSourceCountGCN(input_dim=len(names), hidden_dim=int(mc.get('hidden_dim', 64)),
        dropout=float(mc.get('dropout', .2)), source_head_mode=mc.get('source_head_mode', 'local'),
        global_feature_dim=int(mc.get('global_feature_dim', 0)),
        backbone_mode=mc.get('backbone_mode', 'plain_2'),
        source_head_strategy=mc.get('source_head_strategy', 'shared'),
        shortlist_mode=mc.get('shortlist_mode', 'disabled')).to(device)
    model.load_state_dict(torch.load(paths.runs[seed] / 'best_model.pt',
                                   map_location=device, weights_only=True))
    model.eval()
    records = []
    with torch.inference_mode():
        for index in tqdm(range(1998), desc=f'Independent replay + frozen seed {seed}', unit='cascade'):
            early = replay_early_mask(graph, generation, archive, index)
            example = examples[index].clone().to(device)
            # Targets belong to evaluation/replay, never the model input.
            del example['source_labels']
            del example['source_count']
            logits, counts = model(example)
            baseline = predict_joint(logits, counts, example.candidate_mask)
            ids = torch.where(example.candidate_mask)[0].cpu().tolist()
            truth = frozenset(np.flatnonzero(archive['source_labels'][index]).tolist())
            if not truth or not truth <= set(ids):
                raise ValueError(f'example {index}: invalid independent targets')
            scores = baseline.scores.cpu().tolist()
            candidates = CandidateScores(tuple(ids), tuple(float(scores[i]) for i in ids),
                tuple(bool(early[i]) for i in ids), baseline.sources, baseline.source_count)
            records.append(PilotRecord(index, truth, candidates, not bool(early.any())))
    return records


def seed_identity(paths: IndependentPaths, seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError('Requires all frozen seeds, no replacement seed.')
    verify_opened(paths)
    return {**stage_identity(paths, 'opened'), 'opened': sha256_file(paths.reports / 'opened/payload.json'),
            'seed': seed, 'beta': .5}


def evaluate_seed(paths: IndependentPaths, seed: int, device: torch.device) -> dict:
    identity = seed_identity(paths, seed)
    stage = f'seed_{seed}'
    if (paths.reports / stage).exists():
        result = read_stage(paths.reports, stage, identity)[1]
        print(f'Resumed authenticated report: {paths.reports / stage}', flush=True)
        return result
    if list(paths.reports.glob(f'.{stage}-*')):
        raise ValueError('Partial seed evaluation; no overwrite allowed.')
    records = collect_independent_records(paths, seed, device)
    if [r.index for r in records] != list(range(1998)):
        raise ValueError('Incomplete, duplicated or reordered independent records.')
    mask_rows = [(r.index, r.candidates.candidate_ids, r.candidates.early_observed, r.early_empty)
                 for r in records]
    mask_hash = hashlib.sha256(json.dumps(mask_rows).encode()).hexdigest()
    for other in SEEDS:
        if (paths.reports / f'seed_{other}').exists():
            prior = read_stage(paths.reports, f'seed_{other}', seed_identity(paths, other))[1]
            if prior['early_mask_hash'] != mask_hash:
                raise ValueError('Paired early masks differ across frozen seeds.')
    for r in records:
        if (correct_sources(r.candidates, 0) != r.candidates.baseline_sources
                or len(correct_sources(r.candidates, .5)) != len(r.candidates.baseline_sources)):
            raise ValueError('Control identity/count invariant violated.')
    _, graph = load_graph_archive(paths.data / 'graph.npz')
    geometry = evaluate_pairs(records, .5, graph)
    early = evaluate_early_baseline(records, .5)
    rows = []
    for pair, expected in zip(geometry['rows'], early['rows'], strict=True):
        if (any(pair[key] != expected[key] for key in ('index', 'k', 'candidate_count'))
                or pair['baseline']['f1'] != expected['snapshot_f1']
                or pair['temporal']['f1'] != expected['temporal_f1']):
            raise ValueError('Expected early/geometry row pairing mismatch.')
        rows.append({**expected, 'true_sources': pair['true_sources'],
                     'snapshot_sources': pair['baseline_sources'], 'temporal_sources': pair['temporal_sources'],
                     'snapshot': pair['baseline'], 'temporal': pair['temporal'],
                     'temporal_minus_snapshot': expected['temporal_f1'] - expected['snapshot_f1']})
    result = {'seed': seed, 'beta': .5, 'n': 1998, 'evaluation_role': 'independent_confirmation',
        'exploratory': False, 'dataset_hash': identity['dataset'], 'freeze_hash': identity['freeze'],
        'early_mask_hash': mask_hash, 'count_unchanged': True, 'beta_zero_identity': True,
        'early_policy': early['policy'], 'early_baseline_uses_gcn_count': True,
        'early_ties': early['tie_uncertainty'], 'early_coverage': geometry['early_coverage'],
        'snapshot': geometry['baseline'], 'temporal': geometry['temporal'],
        'early_expected': {k: early[k] for k in ('all', 'by_k', 'by_candidates')},
        'snapshot_comparison_ci': geometry['f1_ci'], 'early_comparison_ci': early['f1_ci'], 'rows': rows}
    if seed_identity(paths, seed) != identity:
        raise ValueError('Inputs changed during evaluation.')
    write_stage(paths.reports, stage, {'identity': identity, 'device': str(device)}, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)
    print(f'Report: {paths.reports / stage / "metrics.json"}', flush=True)
    return result
