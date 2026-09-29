from dataclasses import replace

import numpy as np
import pytest
import torch

from diffusion_sources.temporal_learned_data import (
    cache_identity, load_cache, save_cache, subset_table, validate_table, guard_output,
)
from tests.fixtures.temporal_learned import tiny_table


def test_table_roundtrip_and_whole_cascade_subset(tmp_path):
    table = tiny_table()
    save_cache(tmp_path, 'train', {'v': 1}, table)
    loaded = load_cache(tmp_path, 'train', {'v': 1})
    np.testing.assert_equal(loaded.features, table.features)
    assert loaded.snapshot_sources == table.snapshot_sources
    subset = subset_table(loaded, [3, 1])
    assert subset.indices.tolist() == [3, 1]
    assert subset.offsets.tolist() == [0, 6, 12]
    assert subset.true_counts.tolist() == [1, 2]
    for positions in ([], [0, 0], [-1], [6], [True]):
        with pytest.raises(ValueError):
            subset_table(table, positions)


@pytest.mark.parametrize('name,value', [
    ('offsets', np.array([1, 6, 12, 18, 24, 30, 36])),
    ('offsets', np.array([0, 6, 6, 18, 24, 30, 36])),
    ('indices', np.array([0, 0, 2, 3, 4, 5])),
    ('labels', np.zeros(36)), ('labels', np.ones(36)*2),
    ('probabilities', np.ones(36)*np.nan),
    ('candidate_ids', np.zeros(36, dtype=int)),
    ('predicted_counts', np.zeros(6, dtype=int)),
    ('features', np.ones((36, 6))*np.inf),
])
def test_offsets_labels_and_duplicate_indices_rejected(name, value):
    with pytest.raises(ValueError):
        validate_table(replace(tiny_table(), **{name: value}))


def test_identity_runtime_device_and_input_change(tmp_path, monkeypatch):
    data, run = tmp_path/'data', tmp_path/'run'
    data.mkdir(); run.mkdir()
    for path in (data/'graph.npz', data/'config.yaml', data/'train.npz',
                 run/'config.yaml', run/'best_model.pt'):
        path.write_bytes(b'fixture')
    (run/'config.yaml').write_text('data: {}\n')
    a = cache_identity(data, run, 'train', [0, 1], torch.device('cpu'))
    b = cache_identity(data, run, 'train', [0, 1], torch.device('cuda:0'))
    assert a != b
    assert 'runtime' in a and 'sources' in a
    (data/'train.npz').write_bytes(b'changed')
    assert cache_identity(data, run, 'train', [0, 1], torch.device('cpu')) != a


def test_partial_or_tampered_cache_rejected(tmp_path):
    table = tiny_table()
    save_cache(tmp_path, 'train', {}, table)
    with pytest.raises(FileExistsError):
        save_cache(tmp_path, 'train', {}, table)
    with pytest.raises(ValueError):
        load_cache(tmp_path, 'train', {'changed': True})
    (tmp_path/'train'/'cache.npz').write_bytes(b'tamper')
    with pytest.raises(ValueError):
        load_cache(tmp_path, 'train', {})
    (tmp_path/'.dev-partial').mkdir()
    with pytest.raises(ValueError):
        save_cache(tmp_path, 'dev', {}, table)


def test_output_overlap_and_windows_alias(tmp_path):
    protected = tmp_path/'data'
    for output in (protected, protected/'reports', tmp_path, protected/'..'/'data'):
        with pytest.raises(ValueError):
            guard_output(output, [protected])
    guard_output(tmp_path/'results', [protected])


def test_forbidden_split_never_opens_archive(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Forbidden split accessed the filesystem')
    monkeypatch.setattr('diffusion_sources.temporal_learned_data.sha256_file', forbidden)
    for split in ('test', 'final_holdout', 'independent_holdout', '../train'):
        with pytest.raises(ValueError):
            cache_identity(tmp_path, tmp_path, split, [0], torch.device('cpu'))


@pytest.mark.parametrize('name', ['labels','early_flags','early_empty'])
def test_object_arrays_rejected_before_cache_write(name, tmp_path):
    table=tiny_table()
    table=replace(table,**{name:getattr(table,name).astype(object)})
    with pytest.raises(ValueError,match='binary'):
        save_cache(tmp_path,'objects',{},table)
    assert not (tmp_path/'objects').exists()


def test_distance_dependency_presence_content_and_resume(tmp_path):
    data,run=tmp_path/'data',tmp_path/'run'
    data.mkdir(); run.mkdir()
    distance=tmp_path/'distances.npz'
    for path in (data/'graph.npz',data/'config.yaml',data/'train.npz',run/'best_model.pt'):
        path.write_bytes(b'fixture')
    (run/'config.yaml').write_text('data:\n  distance_cache: '+distance.as_posix()+'\n')
    def identity():
        return cache_identity(data,run,'train',[0],torch.device('cpu'))
    absent=identity()
    save_cache(tmp_path/'results','train',absent,tiny_table())
    distance.write_bytes(b'first distance matrix')
    present=identity()
    assert present!=absent
    with pytest.raises(ValueError):
        load_cache(tmp_path/'results','train',present)
    distance.write_bytes(b'second distance matrix')
    assert identity()!=present
    distance.unlink()
    assert identity()==absent


@pytest.mark.parametrize('module',['diffusion.py','observations.py','metrics.py','temporal_statistics.py'])
def test_shared_source_change_rejects_resume(module,tmp_path,monkeypatch):
    from pathlib import Path
    data,run=tmp_path/'data',tmp_path/'run'
    data.mkdir(); run.mkdir()
    for path in (data/'graph.npz',data/'config.yaml',data/'train.npz',run/'best_model.pt'):
        path.write_bytes(b'fixture')
    (run/'config.yaml').write_text('data: {}\n')
    before=cache_identity(data,run,'train',[0],torch.device('cpu'))
    save_cache(tmp_path/'results','train',before,tiny_table())
    original=Path.read_bytes
    def changed(path):
        value=original(path)
        return value+b'\n# simulated code update\n' if path.name==module else value
    monkeypatch.setattr(Path,'read_bytes',changed)
    after=cache_identity(data,run,'train',[0],torch.device('cpu'))
    with pytest.raises(ValueError):
        load_cache(tmp_path/'results','train',after)
