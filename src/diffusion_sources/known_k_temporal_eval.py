"""Paired known-k model comparison on the same observed cascades."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from .inference import predict_oracle_k
from .metrics import set_metrics, source_radius_hits, source_set_distances
from .temporal_scoring import CandidateScores, correct_sources
from .temporal_statistics import _candidate_bin, paired_bootstrap_ci


def _metrics(graph: nx.Graph, truth: frozenset[int], predicted: frozenset[int]) -> dict:
    # Cardinality is supplied, not inferred: legacy count_accuracy=1 is not a result.
    values = set_metrics(truth, predicted)
    values.pop("count_accuracy")
    values.pop("count_mae")
    return {**values, **source_set_distances(graph, truth, predicted),
            **source_radius_hits(graph, truth, predicted)}


def _aggregate(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"n": 0}
    return {"n": len(rows), **{
        name: float(np.mean([row[label][name] for row in rows]))
        for name in rows[0][label]
    }}


def evaluate_known_k_pairs(
    cases: Sequence[Data],
    graph: nx.Graph,
    candidate_logits_by_index: Mapping[int, torch.Tensor],
    s1b_logits_by_index: Mapping[int, torch.Tensor],
    *,
    beta: float = 0.5,
) -> dict:
    """Evaluate learned ranking against frozen S1b and Temporal-v3 at true k."""
    if beta != 0.5:
        raise ValueError("Known-k pilot fixes Temporal-v3 beta at 0.5")
    if not cases:
        raise ValueError("No paired examples")
    indices = [int(case.example_index) for case in cases]
    if (len(set(indices)) != len(indices)
            or set(indices) != set(candidate_logits_by_index)
            or set(indices) != set(s1b_logits_by_index)):
        raise ValueError("Paired example index mismatch or duplicate index")
    n = graph.number_of_nodes()
    rows: list[dict] = []
    for case in cases:
        index = int(case.example_index)
        if (case.x.shape != (n, 9) or case.source_labels.shape != (n,)
                or case.candidate_mask.shape != (n,) or case.early_observed_mask.shape != (n,)):
            raise ValueError(f"example {index}: invalid paired masks/features")
        k = int(case.source_count.item())
        candidates = case.candidate_mask.bool().cpu()
        truth_mask = case.source_labels.bool().cpu()
        if (k not in (1, 2, 3) or int(truth_mask.sum()) != k
                or bool((truth_mask & ~candidates).any()) or int(candidates.sum()) < k):
            raise ValueError(f"example {index}: invalid target or candidate mask")
        learned_logits = candidate_logits_by_index[index].detach().cpu().flatten()
        frozen_logits = s1b_logits_by_index[index].detach().cpu().flatten()
        if (learned_logits.shape != (n,) or frozen_logits.shape != (n,)
                or not torch.isfinite(learned_logits).all() or not torch.isfinite(frozen_logits).all()):
            raise ValueError(f"example {index}: invalid paired logits")
        candidate_prediction = predict_oracle_k(learned_logits, candidates, k)
        snapshot_prediction = predict_oracle_k(frozen_logits, candidates, k)
        candidate_ids = tuple(torch.nonzero(candidates, as_tuple=False).flatten().tolist())
        frozen_scores = snapshot_prediction.scores.tolist()
        early = case.early_observed_mask.bool().cpu().tolist()
        scoring = CandidateScores(
            candidate_ids=candidate_ids,
            scores=tuple(float(frozen_scores[node]) for node in candidate_ids),
            early_observed=tuple(bool(early[node]) for node in candidate_ids),
            baseline_sources=snapshot_prediction.sources,
            predicted_count=k,
        )
        control_sources = correct_sources(scoring, beta)
        truth = frozenset(torch.nonzero(truth_mask, as_tuple=False).flatten().tolist())
        rows.append({
            "index": index, "k": k, "candidate_count": len(candidate_ids),
            "true_sources": sorted(truth),
            "snapshot_sources": sorted(snapshot_prediction.sources),
            "control_sources": sorted(control_sources),
            "candidate_sources": sorted(candidate_prediction.sources),
            "snapshot": _metrics(graph, truth, snapshot_prediction.sources),
            "control": _metrics(graph, truth, control_sources),
            "candidate": _metrics(graph, truth, candidate_prediction.sources),
        })
    report: dict = {"exploratory": True, "beta": beta, "count_accuracy": None, "rows": rows}
    for label in ("snapshot", "control", "candidate"):
        report[label] = {
            "all": _aggregate(rows, label),
            "by_k": {str(k): _aggregate([r for r in rows if r["k"] == k], label) for k in (1, 2, 3)},
            "by_candidates": {
                group: _aggregate([r for r in rows if _candidate_bin(r["candidate_count"]) == group], label)
                for group in ("1-10", "11-20", "21-50", "51+")
            },
        }
    report["delta_f1"] = report["candidate"]["all"]["f1"] - report["control"]["all"]["f1"]
    report["delta_exact"] = (report["candidate"]["all"]["exact_set_accuracy"]
                             - report["control"]["all"]["exact_set_accuracy"])
    report["delta_by_k"] = {
        str(k): (report["candidate"]["by_k"][str(k)].get("f1", 0)
                 - report["control"]["by_k"][str(k)].get("f1", 0))
        if report["candidate"]["by_k"][str(k)]["n"] else None
        for k in (1, 2, 3)
    }
    report["delta_by_candidates"] = {
        group: (report["candidate"]["by_candidates"][group].get("f1", 0)
                - report["control"]["by_candidates"][group].get("f1", 0))
        if report["candidate"]["by_candidates"][group]["n"] else None
        for group in ("1-10", "11-20", "21-50", "51+")
    }
    report["f1_ci"] = list(paired_bootstrap_ci(
        np.asarray([r["candidate"]["f1"] - r["control"]["f1"] for r in rows]),
        np.asarray([r["k"] for r in rows]),
    ))
    return report


def known_k_pilot_gate(report: dict) -> dict:
    """Apply the fixed exploratory go/no-go conditions without retuning."""
    reasons = []
    if not np.isfinite(report["delta_f1"]) or report["delta_f1"] < 0.02:
        reasons.append("delta F1 < 0.02")
    if not np.isfinite(report["f1_ci"][0]) or report["f1_ci"][0] <= 0:
        reasons.append("paired CI lower <= 0")
    if not np.isfinite(report["delta_exact"]) or report["delta_exact"] < 0:
        reasons.append("exact-set accuracy declined")
    for name, groups in (("k", ("1", "2", "3")),
                         ("candidates", ("1-10", "11-20", "21-50", "51+"))):
        for group in groups:
            value = report[f"delta_by_{name}"].get(group)
            if value is None or not np.isfinite(value) or value < -0.02:
                reasons.append(f"{name}={group} missing or F1 decline > 0.02")
    return {"passed": not reasons, "reasons": reasons}
