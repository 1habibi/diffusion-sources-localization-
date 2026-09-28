import numpy as np
import pytest


def test_metadata_only(tmp_path, monkeypatch):
    from diffusion_sources.temporal_independent_artifacts import metadata_seed_union
    p = tmp_path / 'reference.npz'
    np.savez(p, simulation_seeds=[10, 12], observation_seeds=[11, 13],
             source_labels=np.ones((2, 4)), source_counts=[1, 1])
    with np.load(p) as a:
        cls = type(a)
    real_get = cls.__getitem__
    reads = []
    def checked_get(archive, name):
        assert name in ('simulation_seeds', 'observation_seeds')
        reads.append(name)
        return real_get(archive, name)
    monkeypatch.setattr(cls, '__getitem__', checked_get)
    assert metadata_seed_union([p]) == {10, 11, 12, 13}
    assert reads == ['simulation_seeds', 'observation_seeds']


@pytest.mark.parametrize('sim,obs', [([10, 10], [11, 13]), ([10.5], [11]),
                                    ([-1], [11]), ([10], [11, 13]), ([], []),
                                    ([[10]], [[11]]), ([True], [False])])
def test_invalid_seed_metadata_rejected(tmp_path, sim, obs):
    from diffusion_sources.temporal_independent_artifacts import metadata_seed_union
    p = tmp_path / 'bad.npz'
    np.savez(p, simulation_seeds=sim, observation_seeds=obs)
    with pytest.raises(ValueError, match='seed'):
        metadata_seed_union([p])


def test_parent_and_child_outputs_rejected(tmp_path):
    from diffusion_sources.temporal_independent_artifacts import assert_disjoint_paths
    protected = tmp_path / 'frozen'
    for out in (protected, protected / 'nested', tmp_path):
        with pytest.raises(ValueError, match='overlap'):
            assert_disjoint_paths([out], [protected])
    assert_disjoint_paths([tmp_path / 'new'], [protected])


def test_freeze_authenticates_old_artifacts_and_resumes(tmp_path):
    from tests.fixtures.temporal_independent import make_case
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs, verify_freeze
    paths = make_case(tmp_path)
    before = {p: p.read_bytes() for root in (paths.pilot, paths.baseline, paths.reference)
              for p in root.rglob('*') if p.is_file()}
    frozen = freeze_inputs(paths)
    assert frozen['beta'] == .5 and frozen['seeds'] == [7026, 7027, 7028]
    assert not paths.data.exists()
    assert verify_freeze(paths) == frozen == freeze_inputs(paths)
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('what', ['checkpoint', 'config', 'metrics', 'graph', 'raw', 'reference', 'cache'])
def test_mutations_fail_before_new_stage(tmp_path, what):
    import yaml
    from tests.fixtures.temporal_independent import make_case
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs, verify_freeze
    paths = make_case(tmp_path)
    freeze_inputs(paths)
    targets = {'checkpoint': paths.runs[7027] / 'best_model.pt',
               'config': paths.runs[7027] / 'config.yaml',
               'metrics': paths.runs[7027] / 'metrics.json', 'graph': paths.reference / 'graph.npz',
               'raw': paths.repo / 'raw.txt', 'reference': paths.reference / 'test.npz'}
    if what == 'cache':
        cfg = paths.runs[7027] / 'config.yaml'
        value = yaml.safe_load(cfg.read_text())
        value['data']['distance_cache'] = str(paths.repo / 'cache.npz')
        cfg.write_text(yaml.safe_dump(value))
    else:
        targets[what].write_bytes(b'changed')
    with pytest.raises((ValueError, OSError, KeyError, TypeError)):
        verify_freeze(paths)
    assert not (paths.reports / 'seal').exists()


def test_appearing_cache_is_not_silent_switch(tmp_path):
    from tests.fixtures.temporal_independent import make_case
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs, verify_freeze
    paths = make_case(tmp_path, absent_cache=True)
    freeze_inputs(paths)
    (tmp_path / 'cache_7026.npz').write_bytes(b'cache arrived')
    with pytest.raises(ValueError, match='input'):
        verify_freeze(paths)


@pytest.mark.parametrize('case', ['beta', 'historical_code', 'missing_checkpoint', 'partial', 'alias'])
def test_invalid_preflight_has_no_usable_freeze(tmp_path, case):
    import json
    from dataclasses import replace
    from tests.fixtures.temporal_independent import make_case
    from diffusion_sources.temporal_pilot_artifacts import sha256_file
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs
    paths = make_case(tmp_path)
    manifest_path = paths.pilot / 'select/manifest.json'
    if case == 'beta':
        p = paths.pilot / 'select/payload.json'
        p.write_text(json.dumps({'beta': .25, 'n': 540}))
        m = json.loads(manifest_path.read_text())
        m['file_hashes']['payload.json'] = sha256_file(p)
        manifest_path.write_text(json.dumps(m))
    elif case == 'historical_code':
        m = json.loads(manifest_path.read_text())
        m['identity']['code']['source_hashes']['temporal_scoring.py'] = 'bad'
        manifest_path.write_text(json.dumps(m))
    elif case == 'missing_checkpoint':
        (paths.runs[7026] / 'best_model.pt').unlink()
    elif case == 'partial':
        (paths.reports / '.freeze-interrupted').mkdir(parents=True)
    else:
        paths = replace(paths, reports=paths.pilot / 'nested')
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze_inputs(paths)
    assert not (paths.reports / 'freeze').exists()


def test_git_revision_change_requires_same_frozen_code(tmp_path, monkeypatch):
    from tests.fixtures.temporal_independent import make_case
    from diffusion_sources import temporal_independent_artifacts as a
    paths = make_case(tmp_path)
    a.freeze_inputs(paths)
    real = a._code_identity
    monkeypatch.setattr(a, '_code_identity', lambda: {**real(), 'revision': 'changed'})
    with pytest.raises(ValueError, match='identity'):
        a.verify_freeze(paths)
