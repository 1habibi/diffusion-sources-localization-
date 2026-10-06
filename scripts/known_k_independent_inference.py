"""Explicit opening and frozen, no-training known-k evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from tqdm.auto import tqdm

from diffusion_sources.dataset import graph_to_edge_index, load_graph_archive
from diffusion_sources.inference import predict_joint, predict_oracle_k
from diffusion_sources.known_k_temporal_data import make_observation
from diffusion_sources.known_k_temporal_eval import _aggregate
from diffusion_sources.known_k_temporal_pilot import _config, _load_frozen_s1b
from diffusion_sources.metrics import set_metrics, source_radius_hits, source_set_distances
from diffusion_sources.models import NodeOnlyGCN
from diffusion_sources.temporal_pilot_artifacts import read_stage, sha256_file, write_stage
from diffusion_sources.temporal_replay import FIELDS, replay_early_mask
from diffusion_sources.temporal_scoring import CandidateScores, _readonly_feature_builder, correct_sources
from diffusion_sources.temporal_statistics import _candidate_bin, paired_bootstrap_ci
from scripts.known_k_independent_artifacts import IndependentKnownKPaths, SEEDS
from scripts.known_k_independent_generation import verify_seal

OPEN_TOKEN = "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT"


def _opened_identity(paths: IndependentKnownKPaths) -> dict:
    verify_seal(paths)
    return {"seal": sha256_file(paths.reports / "seal/payload.json"),
            "dataset": sha256_file(paths.data / "independent_holdout.npz")}


def verify_opened(paths: IndependentKnownKPaths) -> dict:
    if not (paths.reports / "opened/complete").is_file():
        raise ValueError("Independent known-k dataset has not been opened")
    result = read_stage(paths.reports, "opened", _opened_identity(paths))[1]
    if result.get("evaluation_status") != "opened" or result.get("confirmation") != OPEN_TOKEN:
        raise ValueError("Invalid known-k opened marker")
    return result


def open_evaluation(paths: IndependentKnownKPaths, confirmation: str) -> dict:
    if confirmation != OPEN_TOKEN:
        raise ValueError(f"Explicit confirmation {OPEN_TOKEN} required")
    identity = _opened_identity(paths)
    if (paths.reports / "opened").exists():
        return verify_opened(paths)
    marker = {"evaluation_status": "opened", "evaluation_role": "independent_confirmation",
              "confirmation": OPEN_TOKEN, "identity": identity,
              "warning": "No retraining, retuning, replacement seed or second dataset."}
    write_stage(paths.reports, "opened", {"identity": identity}, marker)
    print("Known-k independent evaluation OPENED; frozen parameters unchanged.", flush=True)
    return verify_opened(paths)


def _metrics(graph, truth: frozenset[int], sources: frozenset[int], *, known_k: bool) -> dict:
    result = {**set_metrics(truth, sources),
              **source_set_distances(graph, truth, sources),
              **source_radius_hits(graph, truth, sources)}
    if known_k:
        result.pop("count_accuracy")
        result.pop("count_mae")
    return result


def _score_case(graph, edge_index: torch.Tensor, final_features: np.ndarray,
                candidates: np.ndarray, early: np.ndarray, k: int,
                truth: frozenset[int], index: int, candidate_model,
                frozen_model, device: torch.device) -> dict:
    """Feed observable fields to models; truth is consulted only after inference."""
    candidate_input = make_observation(final_features, early, candidates, k, edge_index).to(device)
    snapshot_input = Data(x=candidate_input.x[:, :5], edge_index=candidate_input.edge_index,
                          candidate_mask=candidate_input.candidate_mask,
                          observed_mask=candidate_input.observed_mask)
    with torch.inference_mode():
        candidate_logits = candidate_model(candidate_input).detach().cpu().flatten()
        frozen_logits, count_logits = frozen_model(snapshot_input)
        frozen_logits = frozen_logits.detach().cpu().flatten()
        count_logits = count_logits.detach().cpu()
    candidate_mask = candidate_input.candidate_mask.detach().cpu().bool()
    early_mask = candidate_input.early_observed_mask.detach().cpu().bool()
    if (candidate_logits.shape != (graph.number_of_nodes(),)
            or frozen_logits.shape != candidate_logits.shape
            or not torch.isfinite(candidate_logits).all()
            or not torch.isfinite(frozen_logits).all()
            or not torch.isfinite(count_logits).all()):
        raise ValueError(f"example {index}: invalid frozen logits")
    ids = tuple(torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist())
    if (k not in (1, 2, 3) or len(truth) != k or not truth <= set(ids)):
        raise ValueError(f"example {index}: invalid target/candidate pairing")
    oracle = predict_oracle_k(frozen_logits, candidate_mask, k)
    learned = predict_oracle_k(candidate_logits, candidate_mask, k)
    operational = predict_joint(frozen_logits, count_logits, candidate_mask)
    scores = oracle.scores.tolist()
    scoring = CandidateScores(ids, tuple(float(scores[node]) for node in ids),
                              tuple(bool(early_mask[node]) for node in ids), oracle.sources, k)
    control = correct_sources(scoring, .5)
    if len(control) != k:
        raise ValueError(f"example {index}: known-k control changed cardinality")
    return {
        "index": index, "k": k, "candidate_count": len(ids),
        "true_sources": sorted(truth), "candidate_ids": list(ids),
        "early_mask_hash": hashlib.sha256(early_mask.numpy().tobytes()).hexdigest(),
        "candidate_mask_hash": hashlib.sha256(candidate_mask.numpy().tobytes()).hexdigest(),
        "snapshot_estimated_k": operational.source_count,
        "snapshot_estimated_sources": sorted(operational.sources),
        "snapshot_sources": sorted(oracle.sources),
        "control_sources": sorted(control), "candidate_sources": sorted(learned.sources),
        "snapshot_estimated": _metrics(graph, truth, operational.sources, known_k=False),
        "snapshot": _metrics(graph, truth, oracle.sources, known_k=True),
        "control": _metrics(graph, truth, control, known_k=True),
        "candidate": _metrics(graph, truth, learned.sources, known_k=True),
    }


def _group_metrics(rows: list[dict], label: str) -> dict:
    return {
        "all": _aggregate(rows, label),
        "by_k": {str(k): _aggregate([r for r in rows if r["k"] == k], label)
                 for k in (1, 2, 3)},
        "by_candidates": {
            group: _aggregate([r for r in rows if _candidate_bin(r["candidate_count"]) == group], label)
            for group in ("1-10", "11-20", "21-50", "51+")},
    }


def _collect_report(paths: IndependentKnownKPaths, seed: int, device: torch.device) -> dict:
    verify_opened(paths)  # This must precede every target-array access.
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    _, graph = load_graph_archive(paths.data / "graph.npz")
    edge_index = graph_to_edge_index(graph)
    generation = yaml.safe_load((paths.data / "config.yaml").read_text(encoding="utf-8"))
    s1b_config = yaml.safe_load((paths.s1b / "config.yaml").read_text(encoding="utf-8"))
    names = tuple(s1b_config["data"]["feature_names"])
    if len(names) != 5:
        raise ValueError("Frozen S1b must expose five final-snapshot features")
    builder = _readonly_feature_builder(graph, s1b_config["data"])
    pilot_config = _config(paths.repo / "configs/known_k_temporal_gcn_pilot.yaml")
    candidate_model = NodeOnlyGCN(input_dim=9, hidden_dim=int(pilot_config["model"]["hidden_dim"]),
                                  dropout=float(pilot_config["model"]["dropout"])).to(device)
    checkpoint = (paths.pilot / "pilot/best_model.pt" if seed == 7026
                  else paths.repeats / f"seed_{seed}/best_model.pt")
    candidate_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    candidate_model.eval()
    frozen_model = _load_frozen_s1b(paths.s1b, s1b_config, device)
    with np.load(paths.data / "independent_holdout.npz", allow_pickle=False) as archive:
        arrays = {field: archive[field] for field in FIELDS}
    if any(len(values) != 1998 for values in arrays.values()):
        raise ValueError("Independent known-k archive must contain exactly 1998 cases")
    rows = []
    for index in tqdm(range(1998), desc=f"Known-k independent frozen seed {seed}", unit="cascade"):
        early = replay_early_mask(graph, generation, arrays, index)
        candidates = arrays["candidate_masks"][index].astype(bool)
        final = builder.build(arrays["features"][index, :, 0].astype(bool), names,
                              base_features=arrays["features"][index], candidate_mask=candidates)
        truth = frozenset(np.flatnonzero(arrays["source_labels"][index]).tolist())
        rows.append(_score_case(graph, edge_index, final, candidates, early,
                                int(arrays["source_counts"][index]), truth, index,
                                candidate_model, frozen_model, device))
    deltas = np.asarray([r["candidate"]["f1"] - r["control"]["f1"] for r in rows])
    result = {"seed": seed, "beta": .5, "n": 1998,
              "evaluation_role": "independent_confirmation", "exploratory": False,
              "count_accuracy": None, "rows": rows,
              "early_mask_hash": hashlib.sha256(json.dumps(
                  [(r["index"], r["early_mask_hash"]) for r in rows]).encode()).hexdigest(),
              "candidate_mask_hash": hashlib.sha256(json.dumps(
                  [(r["index"], r["candidate_mask_hash"]) for r in rows]).encode()).hexdigest(),
              "f1_ci": list(paired_bootstrap_ci(deltas, np.asarray([r["k"] for r in rows])))}
    for name in ("snapshot_estimated", "snapshot", "control", "candidate"):
        result[name] = _group_metrics(rows, name)
    result["delta_f1"] = result["candidate"]["all"]["f1"] - result["control"]["all"]["f1"]
    return result


def _seed_identity(paths: IndependentKnownKPaths, seed: int) -> dict:
    if seed not in SEEDS:
        raise ValueError("Requires frozen known-k seed 7026/7027/7028")
    verify_opened(paths)
    return {**_opened_identity(paths),
            "opened": sha256_file(paths.reports / "opened/payload.json"),
            "seed": seed, "beta": .5}


def evaluate_seed(paths: IndependentKnownKPaths, seed: int, device: torch.device) -> dict:
    identity = _seed_identity(paths, seed)
    stage = f"seed_{seed}"
    if (paths.reports / stage).exists():
        return read_stage(paths.reports, stage, identity)[1]
    if list(paths.reports.glob(f".{stage}-*")):
        raise ValueError(f"Partial {stage} stage; no overwrite")
    report = _collect_report(paths, seed, device)
    if _seed_identity(paths, seed) != identity:
        raise ValueError("Frozen identity changed during inference")
    write_stage(paths.reports, stage, {"identity": identity, "device": str(device)}, report)
    print(f"Known-k independent report: {paths.reports / stage}", flush=True)
    return read_stage(paths.reports, stage, identity)[1]
