"""Explicit opening and frozen known-k inference boundaries."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from tests.unit.test_known_k_independent_artifacts import case  # noqa: F401
from tests.unit.test_known_k_independent_generation import _fake_generator


@pytest.fixture
def sealed_case(case, monkeypatch):
    from scripts import known_k_independent_generation as generation
    from scripts.known_k_independent_artifacts import freeze_inputs

    freeze_inputs(case)
    monkeypatch.setattr(generation, "generate_dataset", _fake_generator(case, []))
    generation.generate_and_seal(case)
    return case


def test_open_requires_exact_token(sealed_case):
    from scripts.known_k_independent_inference import open_evaluation, verify_opened

    with pytest.raises(ValueError, match="OPEN_KNOWN_K_INDEPENDENT_HOLDOUT"):
        open_evaluation(sealed_case, "yes")
    assert not (sealed_case.reports / "opened").exists()
    marker = open_evaluation(sealed_case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    assert marker == verify_opened(sealed_case) == open_evaluation(
        sealed_case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    assert marker["evaluation_status"] == "opened"


def test_target_read_requires_open(sealed_case, monkeypatch):
    from scripts import known_k_independent_inference as module

    real_load = np.load
    target_reads = []

    def checked_load(path, *args, **kwargs):
        if Path(path) == sealed_case.data / "independent_holdout.npz":
            target_reads.append(str(path))
            raise AssertionError("target archive was read before opened marker")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)
    with pytest.raises(ValueError, match="opened|open"):
        module.evaluate_seed(sealed_case, 7026, torch.device("cpu"))
    assert target_reads == []


def test_invalid_seed_rejected_before_target_read(sealed_case):
    from scripts.known_k_independent_inference import evaluate_seed

    with pytest.raises(ValueError, match="7026|7027|7028"):
        evaluate_seed(sealed_case, 7000, torch.device("cpu"))


def test_frozen_pairing_has_no_target_in_model_input():
    from scripts.known_k_independent_inference import _score_case
    from diffusion_sources.dataset import graph_to_edge_index

    graph = nx.path_graph(5)
    final = np.zeros((5, 5), dtype=np.float32)
    final[:, 0] = [1, 1, 0, 0, 0]
    candidates = np.ones(5, dtype=bool)
    early = np.asarray([1, 0, 0, 0, 0], dtype=bool)

    class Candidate:
        def __call__(self, data):
            assert data.x.shape == (5, 9)
            assert "source_labels" not in data and "source_count" not in data
            return torch.tensor([5., 1., 0., 0., 0.])

    class Frozen:
        def __call__(self, data):
            assert data.x.shape == (5, 5)
            assert "source_labels" not in data and "source_count" not in data
            assert "early_observed_mask" not in data
            return torch.tensor([0., 5., 4., 0., 0.]), torch.tensor([[0., 8., 0.]])

    row = _score_case(graph, graph_to_edge_index(graph), final, candidates, early,
                      1, frozenset({0}), 0, Candidate(), Frozen(), torch.device("cpu"))
    assert row["snapshot_estimated_k"] == 2
    assert len(row["control_sources"]) == 1
    assert row["candidate_sources"] == [0]
    assert row["snapshot_estimated"]["count_accuracy"] == 0.0
    assert row["candidate"]["f1"] == 1.0


def test_completed_seed_stage_is_read_not_recomputed(sealed_case, monkeypatch):
    from scripts import known_k_independent_inference as module

    module.open_evaluation(sealed_case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    calls = []

    def collect(paths, seed, device):
        calls.append(seed)
        return {"rows": [{"index": i, "k": 1, "candidate_count": 5,
                          "true_sources": [0], "candidate_sources": [0],
                          "control_sources": [0], "early_mask_hash": "a",
                          "candidate_mask_hash": "b", "candidate": {"f1": 1.},
                          "control": {"f1": 1.}} for i in range(1998)],
                "candidate": {"all": {"f1": 1.}}, "control": {"all": {"f1": 1.}}}

    monkeypatch.setattr(module, "_collect_report", collect)
    first = module.evaluate_seed(sealed_case, 7026, torch.device("cpu"))
    assert first == module.evaluate_seed(sealed_case, 7026, torch.device("cpu"))
    assert calls == [7026]


def test_partial_seed_stage_is_not_overwritten(sealed_case):
    from scripts.known_k_independent_inference import evaluate_seed, open_evaluation

    open_evaluation(sealed_case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    (sealed_case.reports / ".seed_7026-partial").mkdir()
    with pytest.raises(ValueError, match="Partial|partial"):
        evaluate_seed(sealed_case, 7026, torch.device("cpu"))
