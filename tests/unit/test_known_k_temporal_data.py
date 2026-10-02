"""Two-observation known-k inputs must not expose source identity."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
import yaml

from diffusion_sources.dataset import graph_to_edge_index, load_graph_archive
from diffusion_sources.features import SnapshotFeatureBuilder
from diffusion_sources.known_k_temporal_data import load_known_k_split, make_observation


FEATURE_NAMES = (
    "observed_infected", "log_degree_normalized", "mean_distance_to_observed_normalized",
    "max_distance_to_observed_normalized", "induced_observed_eccentricity_normalized",
)


def test_make_observation_is_source_blind():
    graph = nx.path_graph(4)
    final = np.arange(20, dtype=np.float32).reshape(4, 5)
    final[:, 0] = [1, 1, 0, 0]
    early = np.array([0, 1, 0, 0], dtype=bool)
    candidates = np.array([1, 1, 1, 0], dtype=bool)

    case = make_observation(final, early, candidates, 2, graph_to_edge_index(graph))

    assert case.x.shape == (4, 9)
    torch.testing.assert_close(case.x[:, :5], torch.from_numpy(final))
    assert case.x[:, 5].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert case.x[:, 6:].tolist() == [[0.0, 1.0, 0.0]] * 4
    assert case.observed_mask.tolist() == [True, True, False, False]
    assert case.candidate_mask.tolist() == [True, True, True, False]
    assert not hasattr(case, "source_labels")
    assert not hasattr(case, "infection_times")


@pytest.mark.parametrize("bad_k", [0, 4, 3])
def test_empty_early_and_bad_k(bad_k):
    final = np.zeros((3, 5), dtype=np.float32)
    early = np.zeros(3, dtype=bool)
    candidates = np.array([1, 1, 0], dtype=bool)
    edge = graph_to_edge_index(nx.path_graph(3))

    case = make_observation(final, early, candidates, 2, edge)
    assert case.x[:, 5].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        make_observation(final, early, candidates, bad_k, edge)


def test_split_replays_same_early_mask_and_preserves_example_index(temporal_dataset):
    from diffusion_sources.temporal_replay import replay_early_mask

    data_dir, config = temporal_dataset
    _, graph = load_graph_archive(data_dir / "graph.npz")
    builder = SnapshotFeatureBuilder(graph)

    cases = load_known_k_split(data_dir, "train", graph, builder, FEATURE_NAMES, [0, 2])

    assert [case.example_index for case in cases] == [0, 2]
    assert all(case.x.shape == (34, 9) for case in cases)
    assert cases[0].edge_index.data_ptr() == cases[1].edge_index.data_ptr()
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        for case in cases:
            index = case.example_index
            expected = replay_early_mask(graph, config, archive, index)
            assert case.early_observed_mask.tolist() == expected.tolist()
            assert case.source_labels.sum().item() == case.source_count.item()
            assert not (case.source_labels.bool() & ~case.candidate_mask).any()


@pytest.mark.parametrize("field", ["simulation_seeds", "source_labels"])
def test_archive_replay_rejects_corruption(temporal_dataset, tmp_path, field):
    data_dir, _ = temporal_dataset
    _, graph = load_graph_archive(data_dir / "graph.npz")
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if field == "source_labels":
        arrays[field][0] = 0
    else:
        arrays[field][0] += 1
    bad_dir = tmp_path / "corrupt"
    bad_dir.mkdir()
    (bad_dir / "config.yaml").write_bytes((data_dir / "config.yaml").read_bytes())
    np.savez_compressed(bad_dir / "train.npz", **arrays)

    with pytest.raises(ValueError, match="example 0"):
        load_known_k_split(bad_dir, "train", graph, SnapshotFeatureBuilder(graph), FEATURE_NAMES, [0])
