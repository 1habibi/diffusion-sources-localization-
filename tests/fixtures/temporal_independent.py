"""Real saved artifacts, with only expensive generation/inference doubled."""
from pathlib import Path
import json
import numpy as np
import yaml
from diffusion_sources.temporal_independent_artifacts import IndependentPaths
from diffusion_sources.temporal_pilot_cli import _identity, _selection_identity, _validation_identity
from diffusion_sources.temporal_pilot_artifacts import write_stage, sha256_file


def generation_config(raw):
    return {'graph': {'id': 'ego_facebook', 'kind': 'edge_list', 'path': str(raw)},
            'simulation': {'source_counts': [1, 2, 3], 'probabilities': [.01, .02, .03],
                           'max_steps': 3, 'distance_ranges': [{'min': 1, 'max': 2}, {'min': 3, 'max': 5}]},
            'observation': {'fractions': [1., .75, .5], 'false_positive_count': 0},
            'dataset': {'seed': 4007026, 'splits': {'independent_holdout': 1998},
                        'min_candidates': 5, 'max_infected_fraction': .5,
                        'max_attempt_factor': 100, 'distance_cache_size': 512, 'show_progress': True}}


def save_graph(root, changed=False):
    root.mkdir(parents=True, exist_ok=True)
    edges = [(i, i + 1) for i in range(9)]
    if changed:
        edges[-1] = (0, 9)
    np.savez(root / 'graph.npz', graph_id='ego_facebook', node_count=10, edges=edges)


def make_case(tmp_path, monkeypatch=None, absent_cache=False, real_checkpoint=False):
    from diffusion_sources.temporal_early_baseline import POLICY
    repo = tmp_path / 'repo'
    repo.mkdir()
    raw = repo / 'raw.txt'
    raw.write_text(''.join(f'{i} {i+1}\n' for i in range(9)))
    config = generation_config(raw)
    cfg = repo / 'generation.yaml'
    cfg.write_text(yaml.safe_dump(config))
    reference, holdout, pilot, baseline = [tmp_path / x for x in ('reference', 'holdout', 'pilot', 'baseline')]
    save_graph(reference)
    holdout.mkdir()
    old_cfg = json.loads(json.dumps(config))
    old_cfg['dataset'].update(seed=7026, splits={'train': 9990, 'validation': 1998, 'test': 1998})
    (reference / 'config.yaml').write_text(yaml.safe_dump(old_cfg))
    for i, split in enumerate(('train', 'validation', 'test')):
        np.savez(reference / f'{split}.npz', simulation_seeds=[10 + 4*i], observation_seeds=[11 + 4*i])
    np.savez(holdout / 'final_holdout.npz', simulation_seeds=[30], observation_seeds=[31])
    runs = {}
    for seed in (7026, 7027, 7028):
        run = tmp_path / f'run_{seed}'
        run.mkdir()
        run_cfg = {'training': {'seed': seed}, 'model': {'hidden_dim': 8, 'dropout': 0.},
                   'data': {'feature_names': ['observed_infected', 'log_degree_normalized']}}
        if absent_cache:
            run_cfg['data']['distance_cache'] = str(tmp_path / f'cache_{seed}.npz')
        (run / 'config.yaml').write_text(yaml.safe_dump(run_cfg))
        if real_checkpoint:
            import torch
            from diffusion_sources.models import JointSourceCountGCN
            torch.manual_seed(seed)
            torch.save(JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.).state_dict(),
                       run / 'best_model.pt')
        else:
            (run / 'best_model.pt').write_bytes(f'checkpoint {seed}'.encode())
        metrics = {'validation_prediction_metrics': {'joint_estimated_k': {'all': {'f1': .35}}}}
        (run / 'metrics.json').write_text(json.dumps(metrics))
        runs[seed] = run
    write_stage(pilot, 'select', {'identity': _selection_identity(reference, runs[7026])}, {'beta': .5, 'n': 540})
    for seed in runs:
        stage = 'validation' if seed == 7026 else f'seed_{seed}'
        payload = {'seed': seed, 'beta': .5, 'early_mask_hash': 'old masks',
                   'baseline': {'all': {'n': 1998, 'f1': .35}},
                   'temporal': {'all': {'n': 1998, 'f1': .55}}}
        identity = _validation_identity(reference, runs[seed], pilot, .5)
        write_stage(pilot, stage, {'identity': identity}, payload)
        identity = {'current_inputs_code': _identity(reference, runs[seed], ('validation',)),
                    'policy': POLICY, 'beta': .5,
                    'source_report': sha256_file(pilot / stage / 'payload.json'),
                    'selection': sha256_file(pilot / 'select/payload.json'),
                    'bootstrap': {'seed': 9282026, 'repetitions': 2000}}
        write_stage(baseline, f'seed_{seed}', {'identity': identity},
                    {'seed': seed, 'beta': .5, 'early_mask_hash': 'old masks',
                     'all': {'n': 1998, 'snapshot_f1': .35, 'temporal_f1': .55, 'early_expected_f1': .4}})
    summary_id = {'seed_payloads': {str(s): sha256_file(baseline / f'seed_{s}/payload.json') for s in runs}}
    # Existing notebook uses this exact seed payload hash identity.
    write_stage(baseline, 'summary', {'identity': summary_id}, {'exploratory': True})
    return IndependentPaths(repo, reference, holdout, pilot, baseline, runs,
                            tmp_path / 'new_data', tmp_path / 'new_reports', cfg)
