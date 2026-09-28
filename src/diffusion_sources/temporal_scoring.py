"""Frozen-model scores and a label-free temporal correction."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from .dataset import load_graph_archive, load_pyg_split
from .features import SnapshotFeatureBuilder
from .inference import predict_joint
from .metrics import set_metrics
from .models import JointSourceCountGCN
from .temporal_replay import FIELDS, replay_early_mask


@dataclass(frozen=True)
class CandidateScores:
    candidate_ids: tuple[int, ...]
    scores: tuple[float, ...]
    early_observed: tuple[bool, ...]
    baseline_sources: frozenset[int]
    predicted_count: int

    def __post_init__(self):
        n = len(self.candidate_ids)
        if (not n or len(set(self.candidate_ids)) != n or min(self.candidate_ids) < 0
                or len(self.scores) != n or len(self.early_observed) != n
                or not np.isfinite(self.scores).all() or any(not 0 <= s <= 1 for s in self.scores)
                or self.predicted_count not in (1, 2, 3)
                or not self.baseline_sources <= set(self.candidate_ids)
                or len(self.baseline_sources) != min(self.predicted_count, n)):
            raise ValueError('Invalid candidate scoring input.')


@dataclass(frozen=True)
class PilotRecord:
    index: int
    true_sources: frozenset[int]
    candidates: CandidateScores
    early_empty: bool


def correct_sources(candidates: CandidateScores, beta: float) -> frozenset[int]:
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and non-negative.')
    if beta == 0:
        return candidates.baseline_sources
    ranked = sorted(zip(candidates.candidate_ids, candidates.scores, candidates.early_observed),
                    key=lambda row: (-(row[1] + beta * int(row[2])), -row[1], row[0]))
    return frozenset(row[0] for row in ranked[:min(candidates.predicted_count, len(ranked))])


def select_beta(records: Sequence[PilotRecord], grid: Sequence[float]) -> tuple[float, dict[float, float]]:
    if not records or not grid:
        raise ValueError('Selection requires records and a beta grid.')
    table = {float(beta): float(np.mean([
        set_metrics(r.true_sources, correct_sources(r.candidates, float(beta)))['f1'] for r in records
    ])) for beta in grid}
    selected = min(table, key=lambda beta: (-table[beta], beta))
    return selected, table


def _readonly_feature_builder(graph, data_config):
    # Never pass a writable frozen cache path to the existing feature builder.
    builder = SnapshotFeatureBuilder(graph, distance_cap=int(data_config.get('distance_cap', 10)))
    configured = data_config.get('distance_cache')
    if configured is not None and Path(configured).exists():
        with np.load(configured, allow_pickle=False) as archive:
            fingerprint = str(archive['graph_fingerprint'].item())
            matrix = np.asarray(archive['distances'], dtype=np.uint16)
        if fingerprint != builder.graph_fingerprint:
            raise ValueError('Distance cache graph fingerprint does not match.')
        if matrix.shape != (builder.node_count, builder.node_count):
            raise ValueError('Distance cache shape does not match graph node count.')
        builder._distance_matrix = matrix
    else:
        print(f'Distance cache unavailable ({configured}); deterministic in-memory recomputation only.', flush=True)
    return builder


def collect_records(data_dir: Path, run_dir: Path, split: str,
                    indices: Sequence[int], device: torch.device) -> list[PilotRecord]:
    if split not in ('train', 'validation'):
        raise ValueError('Only train/validation split is allowed.')
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or int(i) != i for i in indices):
        raise ValueError('indices must be unique non-negative integers.')
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    config = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))
    generation_config = yaml.safe_load((data_dir / 'config.yaml').read_text(encoding='utf-8'))
    names = config['data']['feature_names']
    mc = config['model']
    if mc.get('source_head_strategy', 'shared') != 'shared':
        raise ValueError('Temporal pilot requires shared source head.')
    if int(mc.get('input_dim', len(names))) != len(names):
        raise ValueError('model.input_dim does not match feature_names.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use CPU smoke or select a Colab GPU.')
    _, graph = load_graph_archive(data_dir / 'graph.npz')
    builder = _readonly_feature_builder(graph, config['data'])
    with np.load(data_dir / f'{split}.npz', allow_pickle=False) as compressed:
        # NpzFile.__getitem__ decompresses a whole split array on every access.
        # Materialize once before per-cascade replay, not thousands of times.
        archive = {field: compressed[field] for field in FIELDS}
        if max(indices) >= len(archive['source_counts']):
            raise ValueError('indices outside split.')
        examples = load_pyg_split(data_dir / f'{split}.npz', graph, feature_names=names,
                                  feature_builder=builder, limit=max(indices) + 1,
                                  progress_description=f'Loading {split}')
        model = JointSourceCountGCN(
            input_dim=len(names), hidden_dim=int(mc.get('hidden_dim', 64)),
            dropout=float(mc.get('dropout', .2)), source_head_mode=mc.get('source_head_mode', 'local'),
            global_feature_dim=int(mc.get('global_feature_dim', 0)),
            backbone_mode=mc.get('backbone_mode', 'plain_2'),
            source_head_strategy=mc.get('source_head_strategy', 'shared'),
            shortlist_mode=mc.get('shortlist_mode', 'disabled')).to(device)
        model.load_state_dict(torch.load(run_dir / 'best_model.pt', map_location=device, weights_only=True))
        model.eval()
        records = []
        with torch.inference_mode():
            for index in tqdm(indices, desc=f'Replay + frozen inference ({split})', unit='cascade'):
                early = replay_early_mask(graph, generation_config, archive, index)
                # Data.to mutates its receiver; do not retain one GPU graph per
                # cascade in the CPU examples list throughout validation.
                example = examples[index].clone().to(device)
                logits, counts = model(example)
                baseline = predict_joint(logits, counts, example.candidate_mask)
                ids = torch.where(example.candidate_mask)[0].cpu().tolist()
                true = frozenset(np.flatnonzero(archive['source_labels'][index]).tolist())
                if not true or not true <= set(ids):
                    raise ValueError(f'example {index}: labels outside candidate mask.')
                values = baseline.scores.cpu().numpy()
                candidates = CandidateScores(tuple(ids), tuple(float(values[i]) for i in ids),
                    tuple(bool(early[i]) for i in ids), baseline.sources, baseline.source_count)
                records.append(PilotRecord(index, true, candidates, not bool(early.any())))
        return records
