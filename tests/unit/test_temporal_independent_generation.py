import copy
import json
import numpy as np
import pytest
import yaml
from tests.fixtures.temporal_independent import make_case, generation_config, save_graph


def fake_generate(cfg, root, *, n=1998, changed_graph=False, bad_seed=False):
    save_graph(root, changed_graph)
    simulation_seeds = 4007026 + 2 * np.arange(n)
    if bad_seed:
        simulation_seeds[0] = 10
    np.savez(root / 'independent_holdout.npz',
             features=np.zeros((n, 10, 2), dtype=np.float32),
             candidate_masks=np.ones((n, 10), dtype=bool),
             source_labels=np.zeros((n, 10), dtype=bool),
             infected_masks=np.ones((n, 10), dtype=bool),
             source_counts=np.ones(n, dtype=np.int64),
             probabilities=np.full(n, .01), observation_fractions=np.ones(n),
             simulation_seeds=simulation_seeds, observation_seeds=simulation_seeds + 1)
    (root / 'config.yaml').write_text(yaml.safe_dump(cfg))
    (root / 'generation_summary.json').write_text(json.dumps({'accepted': {'independent_holdout': n}}))


def test_exact_config_and_reject_changes(tmp_path):
    from diffusion_sources.temporal_independent_generation import validate_generation_config
    cfg = generation_config(tmp_path / 'graph.txt')
    assert validate_generation_config(cfg, tmp_path)['dataset']['seed'] == 4007026
    for section, field, value in [('dataset', 'seed', 4007027), ('dataset', 'splits', {'test': 1998}),
                                  ('simulation', 'max_steps', 4), ('observation', 'hide_source_count', 1),
                                  ('dataset', 'extra_filter', True)]:
        bad = copy.deepcopy(cfg)
        bad[section][field] = value
        with pytest.raises(ValueError, match='protocol'):
            validate_generation_config(bad, tmp_path)


def test_seal_only_metadata_and_resume(tmp_path, monkeypatch):
    from diffusion_sources import temporal_independent_generation as g
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs, verify_seal
    paths = make_case(tmp_path)
    freeze_inputs(paths)
    monkeypatch.setattr(g, 'generate_dataset', fake_generate)
    with np.load(paths.reference / 'train.npz') as a:
        cls = type(a)
    get = cls.__getitem__
    def guard(archive, key):
        assert key in ('simulation_seeds', 'observation_seeds', 'graph_id', 'node_count', 'edges')
        return get(archive, key)
    monkeypatch.setattr(cls, '__getitem__', guard)
    sealed = g.generate_and_seal(paths)
    assert sealed['evaluation_status'] == 'sealed_unopened'
    assert sealed['n'] == 1998
    assert not (paths.reports / 'opened').exists()
    monkeypatch.setattr(g, 'generate_dataset', lambda *a: pytest.fail('Resume must not generate'))
    assert g.generate_and_seal(paths) == verify_seal(paths) == sealed


@pytest.mark.parametrize('case', ['size', 'graph', 'seed', 'config', 'partial', 'unsealed', 'interrupted', 'persist'])
def test_bad_generation_not_published_as_sealed(tmp_path, monkeypatch, case):
    from diffusion_sources import temporal_independent_generation as g
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs
    paths = make_case(tmp_path)
    freeze_inputs(paths)
    if case == 'partial':
        (paths.data.parent / f'.{paths.data.name}-interrupted').mkdir()
    elif case == 'unsealed':
        paths.data.mkdir()
    def generate(cfg, root):
        if case == 'interrupted':
            raise RuntimeError('interrupted')
        fake_generate(cfg, root, n=1997 if case == 'size' else 1998,
                      changed_graph=case == 'graph', bad_seed=case == 'seed')
        if case == 'config':
            (root / 'config.yaml').write_text(yaml.safe_dump({'changed': True}))
    monkeypatch.setattr(g, 'generate_dataset', generate)
    if case == 'persist':
        def interrupt(*args):
            raise RuntimeError('persist interrupted')
        monkeypatch.setattr(g, 'write_stage', interrupt)
    with pytest.raises((ValueError, RuntimeError)):
        g.generate_and_seal(paths)
    assert not (paths.reports / 'seal').exists()
    if case == 'persist':
        with pytest.raises(ValueError, match='no overwrite'):
            g.generate_and_seal(paths)


def test_entire_attempt_window_checked_before_generator(tmp_path, monkeypatch):
    from diffusion_sources import temporal_independent_generation as g
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs
    paths = make_case(tmp_path)
    np.savez(paths.snapshot_holdout / 'final_holdout.npz',
             simulation_seeds=[4406624], observation_seeds=[4406625])
    freeze_inputs(paths)
    monkeypatch.setattr(g, 'generate_dataset', lambda *args: pytest.fail('Must stop before generation'))
    with pytest.raises(ValueError, match='generation window'):
        g.generate_and_seal(paths)
