"""Pre-registered paired summary over 1,998 cascades, not 5,994 pseudo-cases."""

from __future__ import annotations

import math
import statistics

import numpy as np

from diffusion_sources.temporal_pilot_artifacts import read_stage, sha256_file, write_stage
from diffusion_sources.temporal_statistics import paired_bootstrap_ci
from scripts.known_k_independent_artifacts import IndependentKnownKPaths, SEEDS
from scripts.known_k_independent_inference import _seed_identity, validate_seed_rows, verify_opened


def _metric_table(reports: list[dict], label: str) -> dict:
    names = tuple(sorted(reports[0]["rows"][0][label]))
    values = {name: [float(np.mean([r[label][name] for r in report["rows"]]))
                     for report in reports] for name in names}
    return {"mean": {name: float(statistics.mean(series)) for name, series in values.items()},
            "sample_sd": {name: float(statistics.stdev(series)) for name, series in values.items()}}


def _group(rows_by_seed: list[list[dict]], indices: np.ndarray) -> dict:
    count = int(indices.sum())
    names = tuple(sorted(rows_by_seed[0][0]["control"]))
    if not count:
        missing = {name: None for name in names}
        return {"n_per_seed": 0, "control": missing, "candidate": missing.copy(),
                "control_f1": None, "candidate_f1": None, "delta_f1": None}

    def arm(label: str) -> dict:
        return {name: float(np.mean([[row[label][name] for row, include in zip(rows, indices, strict=True)
                                      if include] for rows in rows_by_seed])) for name in names}

    control = arm("control")
    candidate = arm("candidate")
    return {"n_per_seed": count, "control": control, "candidate": candidate,
            "control_f1": control["f1"], "candidate_f1": candidate["f1"],
            "delta_f1": candidate["f1"] - control["f1"]}


def summarize_reports(reports: dict[int, dict]) -> dict:
    if set(reports) != set(SEEDS):
        raise ValueError("Exactly three frozen known-k seeds 7026/7027/7028 required")
    ordered = [reports[seed] for seed in SEEDS]
    reference_rows = ordered[0]["rows"]
    validate_seed_rows(reference_rows)
    for seed, report in zip(SEEDS, ordered, strict=True):
        if (report.get("seed") != seed or report.get("n") != 1998
                or report.get("beta") != .5 or report.get("count_accuracy") is not None
                or report.get("evaluation_role") != "independent_confirmation"
                or report.get("exploratory") is not False):
            raise ValueError("Invalid independent known-k report protocol or seed")
        validate_seed_rows(report["rows"], reference_rows)
        if (report.get("early_mask_hash") != ordered[0].get("early_mask_hash")
                or report.get("candidate_mask_hash") != ordered[0].get("candidate_mask_hash")):
            raise ValueError("Paired early or candidate fingerprints differ")
    true_k = np.asarray([row["k"] for row in reference_rows], dtype=np.int64)
    if {k: int((true_k == k).sum()) for k in (1, 2, 3)} != {1: 666, 2: 666, 3: 666}:
        raise ValueError("Independent known-k strata must contain 666 cases each")
    control = np.asarray([[row["control"]["f1"] for row in report["rows"]]
                          for report in ordered], dtype=np.float64)
    candidate = np.asarray([[row["candidate"]["f1"] for row in report["rows"]]
                            for report in ordered], dtype=np.float64)
    delta = candidate - control
    per_seed = {}
    for position, seed in enumerate(SEEDS):
        mean_delta = float(delta[position].mean())
        if not math.isfinite(mean_delta) or abs(mean_delta - reports[seed].get("delta_f1", mean_delta)) > 1e-12:
            raise ValueError(f"Seed {seed} reported delta disagrees with paired rows")
        per_seed[str(seed)] = {
            "control_f1": float(control[position].mean()),
            "candidate_f1": float(candidate[position].mean()),
            "delta_f1": mean_delta,
            "f1_ci": list(paired_bootstrap_ci(delta[position], true_k,
                                               repetitions=2000, seed=9282026)),
        }
    cascade_mean_delta = delta.mean(axis=0)
    primary_ci = list(paired_bootstrap_ci(cascade_mean_delta, true_k,
                                           repetitions=2000, seed=9282026))
    primary_delta = float(cascade_mean_delta.mean())
    reasons = []
    if primary_delta < .02:
        reasons.append("mean delta F1 < 0.02")
    if primary_ci[0] <= 0:
        reasons.append("paired CI lower <= 0")
    if any(per_seed[str(seed)]["delta_f1"] <= 0 for seed in SEEDS):
        reasons.append("non-positive delta F1 on a seed")
    count = np.asarray([row["candidate_count"] for row in reference_rows], dtype=np.int64)
    metrics = {label: _metric_table(ordered, label) for label in ("control", "candidate")}
    snapshot_estimated = _metric_table(ordered, "snapshot_estimated")
    snapshot_estimated["by_k"] = {
        str(k): {name: float(np.mean([row["snapshot_estimated"][name]
                                      for row in reference_rows if row["k"] == k]))
                 for name in snapshot_estimated["mean"]} for k in (1, 2, 3)}
    snapshot_estimated["by_candidates"] = {
        group: {name: (float(np.mean([reference_rows[i]["snapshot_estimated"][name]
                                      for i in np.flatnonzero(mask)])) if mask.any() else None)
                for name in snapshot_estimated["mean"]}
        for group, mask in {
            "1-10": count <= 10, "11-20": (count > 10) & (count <= 20),
            "21-50": (count > 20) & (count <= 50), "51+": count > 50,
        }.items()}
    return {
        "evaluation_role": "independent_confirmation", "exploratory": False,
        "seeds": list(SEEDS), "n": 1998, "count_accuracy": None, "beta": .5,
        "bootstrap": {"n": 1998, "strata": "true_k", "repetitions": 2000,
                      "seed": 9282026, "unit": "mean three seed deltas per cascade"},
        "primary": {"delta_f1": primary_delta, "ci": primary_ci,
                    "per_seed_delta": {key: row["delta_f1"] for key, row in per_seed.items()},
                    "passed": not reasons, "reasons": reasons},
        "mean": {"control_f1": float(control.mean()), "candidate_f1": float(candidate.mean()),
                 "delta_f1": primary_delta},
        "sample_sd": {"control_f1": float(statistics.stdev([r["control_f1"] for r in per_seed.values()])),
                      "candidate_f1": float(statistics.stdev([r["candidate_f1"] for r in per_seed.values()])),
                      "delta_f1": float(statistics.stdev([r["delta_f1"] for r in per_seed.values()]))},
        "per_seed": per_seed, "metrics": metrics, "snapshot_estimated": snapshot_estimated,
        "by_k": {str(k): _group([r["rows"] for r in ordered], true_k == k) for k in (1, 2, 3)},
        "by_candidates": {
            "1-10": _group([r["rows"] for r in ordered], count <= 10),
            "11-20": _group([r["rows"] for r in ordered], (count > 10) & (count <= 20)),
            "21-50": _group([r["rows"] for r in ordered], (count > 20) & (count <= 50)),
            "51+": _group([r["rows"] for r in ordered], count > 50),
        },
        "interpretation": "New synthetic IC cascades on the same Facebook topology; not transfer. "
                          "CI is conditional on graph, protocol and three frozen checkpoints; "
                          "subgroup results are descriptive. Negative results are retained without retuning.",
    }


def save_summary(paths: IndependentKnownKPaths) -> dict:
    verify_opened(paths)
    reports = {}
    stage_hashes = {}
    for seed in SEEDS:
        stage = f"seed_{seed}"
        reports[seed] = read_stage(paths.reports, stage, _seed_identity(paths, seed))[1]
        stage_hashes[str(seed)] = sha256_file(paths.reports / stage / "manifest.json")
    identity = {"opened": sha256_file(paths.reports / "opened/payload.json"),
                "seed_manifests": stage_hashes}
    if (paths.reports / "summary").exists():
        return read_stage(paths.reports, "summary", identity)[1]
    result = summarize_reports(reports)
    result["seed_manifest_hashes"] = stage_hashes
    write_stage(paths.reports, "summary", {"identity": identity}, result)
    return read_stage(paths.reports, "summary", identity)[1]
