import copy
import inspect
import networkx as nx
import numpy as np
import pytest
from diffusion_sources.dataset import load_graph_archive


def test_source_blind_sampling_and_empty():
    from diffusion_sources.temporal_replay import sample_early_nodes
    assert sample_early_nodes([], .5, 13) == frozenset()
    assert sample_early_nodes([4, 1, 3], 1., 13) == frozenset({1, 3, 4})
    assert sample_early_nodes([4, 1, 3], .5, 13) == sample_early_nodes([3, 4, 1], .5, 13)
    assert set(inspect.signature(sample_early_nodes).parameters) == {'infected_by_t1', 'fraction', 'observation_seed', 't1'}


@pytest.mark.parametrize('fraction,t1', [(0., 1), (1.1, 1), (.5, 0), (float('nan'), 1)])
def test_invalid_policy(fraction, t1):
    from diffusion_sources.temporal_replay import sample_early_nodes
    with pytest.raises(ValueError):
        sample_early_nodes([1], fraction, 13, t1=t1)


def test_exact_replay(temporal_dataset):
    from diffusion_sources.temporal_replay import replay_early_mask
    path, config = temporal_dataset
    _, graph = load_graph_archive(path / 'graph.npz')
    with np.load(path / 'train.npz') as archive:
        for index in range(6):
            mask = replay_early_mask(graph, config, archive, index)
            assert mask.dtype == bool and mask.shape == (34,)
            assert mask.any()
            np.testing.assert_array_equal(mask, replay_early_mask(graph, config, archive, index))


@pytest.mark.parametrize('field', ['source_labels', 'infected_masks', 'candidate_masks', 'features',
                                 'source_counts', 'probabilities', 'observation_fractions'])
def test_corruption_stops_replay(temporal_dataset, field):
    from diffusion_sources.temporal_replay import replay_early_mask
    path, config = temporal_dataset
    _, graph = load_graph_archive(path / 'graph.npz')
    with np.load(path / 'train.npz') as archive:
        bad = {key: archive[key].copy() for key in archive.files}
    if field == 'features':
        bad[field][0, 0, 0] = 1 - bad[field][0, 0, 0]
    elif bad[field].ndim == 2:
        bad[field][0, 0] = 1 - bad[field][0, 0]
    else:
        bad[field][0] += 1
    with pytest.raises(ValueError, match='example 0'):
        replay_early_mask(graph, config, bad, 0)


def test_invalid_archive_and_protocol(temporal_dataset):
    from diffusion_sources.temporal_replay import replay_early_mask
    path, config = temporal_dataset
    _, graph = load_graph_archive(path / 'graph.npz')
    with np.load(path / 'train.npz') as archive:
        with pytest.raises(ValueError):
            replay_early_mask(graph, config, {}, 0)
        bad = copy.deepcopy(config)
        bad['observation']['hide_source_count'] = 1
        with pytest.raises(ValueError):
            replay_early_mask(graph, bad, archive, 0)
        with pytest.raises(ValueError):
            replay_early_mask(nx.path_graph([10, 11]), config, archive, 0)


def test_stopped_cascade_not_filtered():
    from diffusion_sources.temporal_replay import replay_early_mask
    from diffusion_sources.diffusion import SourceSampler, simulate_ic
    from diffusion_sources.observations import observe_cascade
    from diffusion_sources.dataset import build_example
    graph = nx.path_graph(5)
    rng = np.random.default_rng(13)
    sources = SourceSampler(graph).sample(1, rng)
    cascade = simulate_ic(graph, sources, 0., 3, rng)
    observation = observe_cascade(graph, cascade, 1., 0, np.random.default_rng(14))
    ex = build_example('path', graph, cascade, observation, simulation_seed=13, observation_seed=14)
    archive = dict(features=np.array([ex.features]), source_labels=np.array([ex.source_labels]),
                   candidate_masks=np.array([ex.candidate_mask]), infected_masks=np.array([ex.infected_mask]),
                   source_counts=np.array([1]), simulation_seeds=np.array([13]), observation_seeds=np.array([14]),
                   probabilities=np.array([0.]), observation_fractions=np.array([1.]))
    config = {'simulation': {'source_counts': [1], 'probabilities': [0.], 'max_steps': 3},
              'observation': {'fractions': [1.]}}
    assert replay_early_mask(graph, config, archive, 0).sum() == 1
