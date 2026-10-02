"""Paired known-k evaluation and the preregistered pilot gate."""

import networkx as nx
import numpy as np
import pytest
import torch

from diffusion_sources.dataset import graph_to_edge_index
from diffusion_sources.known_k_temporal_data import make_observation
from diffusion_sources.known_k_temporal_eval import evaluate_known_k_pairs, known_k_pilot_gate


def _case(index=0):
    graph = nx.path_graph(3)
    final = np.zeros((3, 10), dtype=np.float32)
    final[:, 0] = [1, 1, 0]
    case = make_observation(
        final, np.array([0, 0, 1]), np.array([1, 1, 1]), 1, graph_to_edge_index(graph)
    )
    case.source_labels = torch.tensor([0.0, 0.0, 1.0])
    case.source_count = torch.tensor(1)
    case.example_index = index
    return graph, case


def test_paired_rows_require_matching_indices_and_known_k_control():
    graph, case = _case(7)
    candidate = {7: torch.tensor([0.0, 0.0, 3.0])}
    baseline = {7: torch.tensor([0.0, 2.0, 1.0])}

    with pytest.raises(ValueError, match="index"):
        evaluate_known_k_pairs([case], graph, candidate, {8: baseline[7]})
    report = evaluate_known_k_pairs([case], graph, candidate, baseline)

    row = report["rows"][0]
    assert row["index"] == 7 and row["k"] == 1
    assert row["snapshot_sources"] == [1]
    assert row["control_sources"] == [2]  # sigmoid(1) + 0.5 exceeds sigmoid(2)
    assert row["candidate_sources"] == [2]
    assert len(row["candidate_sources"]) == row["k"]
    assert report["delta_f1"] == 0.0
    assert report["candidate"]["all"]["f1"] == 1.0
    assert "count_accuracy" not in report["candidate"]["all"]
    assert report["count_accuracy"] is None


def test_duplicate_case_indices_and_bad_targets_fail_before_aggregation():
    graph, case = _case(7)
    logits = {7: torch.tensor([0.0, 0.0, 3.0])}
    with pytest.raises(ValueError, match="index"):
        evaluate_known_k_pairs([case, case], graph, logits, logits)
    case.source_labels = torch.tensor([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="target"):
        evaluate_known_k_pairs([case], graph, logits, logits)


def _passing_report():
    return {
        "delta_f1": 0.02,
        "f1_ci": [0.001, 0.04],
        "delta_exact": 0.0,
        "delta_by_k": {"1": 0.0, "2": 0.0, "3": 0.0},
        "delta_by_candidates": {"1-10": 0.0, "11-20": 0.0, "21-50": 0.0, "51+": 0.0},
    }


def test_gate_accepts_only_all_thresholds_met():
    assert known_k_pilot_gate(_passing_report()) == {"passed": True, "reasons": []}


@pytest.mark.parametrize(
    "field,value",
    [("delta_f1", 0.019), ("f1_ci", [0.0, 0.04]), ("delta_exact", -0.001)],
)
def test_gate_rejects_primary_or_exact_failure(field, value):
    report = _passing_report()
    report[field] = value
    assert not known_k_pilot_gate(report)["passed"]


@pytest.mark.parametrize("group", ["1", "2", "3"])
def test_gate_rejects_each_k_decline(group):
    report = _passing_report()
    report["delta_by_k"][group] = -0.021
    assert not known_k_pilot_gate(report)["passed"]


@pytest.mark.parametrize("group", ["1-10", "11-20", "21-50", "51+"])
def test_gate_rejects_each_candidate_bin_decline(group):
    report = _passing_report()
    report["delta_by_candidates"][group] = -0.021
    assert not known_k_pilot_gate(report)["passed"]
