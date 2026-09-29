"""Read-only replay and frozen inference; labels never enter model inputs."""
import hashlib
import itertools
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
import yaml

from .dataset import graph_to_edge_index, load_graph_archive
from .features import GLOBAL_SCALAR_FEATURE_NAMES
from .inference import predict_joint
from .models import JointSourceCountGCN
from .temporal_learned_data import CandidateTable, cache_identity, validate_table
from .temporal_learned_features import build_candidate_features
from .temporal_replay import FIELDS, replay_early_mask
from .temporal_scoring import CandidateScores, _readonly_feature_builder, correct_sources


def check_train_partition(config, archive):
    sim, obs = config['simulation'], config['observation']
    conditions = list(itertools.product(sim['distance_ranges'], sim['probabilities'], obs['fractions'], sim['source_counts']))
    if len(conditions) != 54 or sim['max_steps'] != 3 or set(sim['source_counts']) != {1,2,3}:
        raise ValueError('Expected fixed 54-condition protocol')
    if len(archive['source_counts']) != 9990:
        raise ValueError('Expected 9990 train cascades')
    for name in ('source_counts', 'probabilities', 'observation_fractions', 'simulation_seeds', 'observation_seeds'):
        values = np.asarray(archive[name])
        if values.shape != (9990,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid train metadata: {name}')
    fit, dev = list(range(1620)), list(range(1620,2160))
    for name, position in (('source_counts',3), ('probabilities',1), ('observation_fractions',2)):
        expected = np.array([conditions[i % 54][position] for i in range(2160)])
        if not np.allclose(archive[name][:2160], expected, rtol=0, atol=1e-8):
            raise ValueError(f'Unbalanced or reordered train {name}')
    for name in ('simulation_seeds', 'observation_seeds'):
        values = np.asarray(archive[name])
        if values.dtype.kind not in 'iu' or (values < 0).any() or len(np.unique(values)) != 9990:
            raise ValueError(f'Invalid or duplicate {name}')
    left = set(map(int, archive['simulation_seeds'][fit])) | set(map(int, archive['observation_seeds'][fit]))
    right = set(map(int, archive['simulation_seeds'][dev])) | set(map(int, archive['observation_seeds'][dev]))
    if left & right:
        raise ValueError('Fit/dev simulation+observation seed unions overlap')
    return fit, dev


class Budget:
    """Soft budget checked at operation boundaries, not a solver interrupt."""
    def __init__(self, seconds):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Budget must be positive and finite')
        self.seconds = seconds
        self.start = time.monotonic()

    def check(self):
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError(f'Stage exceeded {self.seconds:g}s boundary budget')


def collect_candidates(data_dir, run_dir, split, indices, device, *, budget_seconds=1800):
    budget = Budget(budget_seconds)
    budget.check()
    data, run, device = Path(data_dir), Path(run_dir), torch.device(device)
    identity = cache_identity(data, run, split, indices, device)
    budget.check()
    config = yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    generation = yaml.safe_load((data/'config.yaml').read_text(encoding='utf-8'))
    names, mc = config['data']['feature_names'], config['model']
    if mc.get('source_head_strategy', 'shared') != 'shared' or int(mc.get('input_dim', len(names))) != len(names):
        raise ValueError('Requires shared source head and matching feature dimensions')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    _, graph = load_graph_archive(data/'graph.npz')
    builder = _readonly_feature_builder(graph, config['data'])
    budget.check()
    with np.load(data/f'{split}.npz', allow_pickle=False) as compressed:
        archive = {f: compressed[f] for f in FIELDS}
    if max(indices) >= len(archive['source_counts']):
        raise ValueError('Indices outside split')
    budget.check()
    model = JointSourceCountGCN(input_dim=len(names), hidden_dim=int(mc.get('hidden_dim',64)),
        dropout=float(mc.get('dropout',.2)), source_head_mode=mc.get('source_head_mode','local'),
        global_feature_dim=int(mc.get('global_feature_dim',0)), backbone_mode=mc.get('backbone_mode','plain_2'),
        source_head_strategy=mc.get('source_head_strategy','shared'), shortlist_mode=mc.get('shortlist_mode','disabled')).to(device)
    model.load_state_dict(torch.load(run/'best_model.pt', map_location=device, weights_only=True))
    model.eval()
    edge_index = graph_to_edge_index(graph)
    budget.check()
    features, ids_rows, probabilities, flags, labels = [], [], [], [], []
    offsets, predicted, true_counts, empties, snapshots = [0], [], [], [], []
    masks_digest = hashlib.sha256()
    with torch.inference_mode():
        for index in tqdm(indices, desc=f'Replay + frozen inference ({split})', unit='cascade'):
            budget.check()
            early = replay_early_mask(graph, generation, archive, index)
            budget.check()
            final = archive['features'][index, :, 0].astype(bool)
            mask = archive['candidate_masks'][index].astype(bool)
            x = builder.build(final, names, base_features=archive['features'][index], candidate_mask=mask)
            budget.check()
            # This object is constructed without any supervision fields.
            example = Data(x=torch.from_numpy(x).float(), edge_index=edge_index,
                candidate_mask=torch.from_numpy(mask), observed_mask=torch.from_numpy(final),
                global_features=torch.from_numpy(x[:1, [i for i,name in enumerate(names) if name in GLOBAL_SCALAR_FEATURE_NAMES]]).float()).to(device)
            logits, counts = model(example)
            baseline = predict_joint(logits, counts, example.candidate_mask)
            budget.check()
            ids = np.flatnonzero(mask).astype(np.int64)
            p = baseline.scores.detach().cpu().numpy()[ids].astype(np.float64)
            target = archive['source_labels'][index]
            if not np.isin(target, [0,1]).all() or np.any(target[~mask]):
                raise ValueError('Source labels outside candidate pool or nonbinary')
            c = CandidateScores(tuple(map(int,ids)), tuple(p), tuple(early[ids]), baseline.sources, baseline.source_count)
            if correct_sources(c, 0) != baseline.sources:
                raise ValueError('Native beta-zero control mismatch')
            features.append(build_candidate_features(graph, tuple(map(int,ids)), p, early, final))
            ids_rows.append(ids); probabilities.append(p); flags.append(early[ids]); labels.append(target[ids].astype(bool))
            offsets.append(offsets[-1]+len(ids)); predicted.append(baseline.source_count)
            true_counts.append(int(archive['source_counts'][index])); empties.append(not early.any()); snapshots.append(baseline.sources)
            masks_digest.update(np.asarray([index], dtype='<i8').tobytes())
            masks_digest.update(early.astype(np.uint8).tobytes())
            del example, logits, counts, baseline
            budget.check()
    table = CandidateTable(np.concatenate(features), np.concatenate(ids_rows), np.concatenate(probabilities),
        np.concatenate(flags), np.concatenate(labels), np.array(offsets,dtype=np.int64), np.array(indices,dtype=np.int64),
        np.array(predicted,dtype=np.int64), np.array(true_counts,dtype=np.int64), np.array(empties,dtype=bool),
        tuple(snapshots), masks_digest.hexdigest())
    validate_table(table)
    if cache_identity(data, run, split, indices, device) != identity:
        raise ValueError('Inputs changed during collection')
    budget.check()
    return table
