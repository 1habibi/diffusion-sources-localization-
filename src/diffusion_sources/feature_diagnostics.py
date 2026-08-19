"""Diagnose named snapshot features on train/validation candidates only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .dataset import load_graph_archive
from .features import SnapshotFeatureBuilder


def diagnose_snapshot_features(
    config_path: str | Path,
    output_path: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Summarize candidate feature variance without reading labels or test."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_config = config.get("data", {})
    feature_names = [str(name) for name in data_config.get("feature_names", [])]
    if not feature_names:
        raise ValueError("Configuration must define data.feature_names.")
    if any(split not in {"train", "validation"} for split in splits):
        raise ValueError("Feature diagnostics are restricted to train/validation.")
    if bool(config.get("evaluation", {}).get("evaluate_test", True)):
        raise ValueError("Feature diagnostics require evaluation.evaluate_test=false.")

    data_dir = Path(data_config["directory"])
    graph_id, graph = load_graph_archive(data_dir / "graph.npz")
    builder = SnapshotFeatureBuilder(
        graph,
        distance_cache_path=data_config.get("distance_cache"),
        distance_cap=int(data_config.get("distance_cap", 10)),
    )
    split_results = {}
    for split in splits:
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            observed_masks = archive["features"][:, :, 0].astype(bool)
            candidate_masks = archive["candidate_masks"].astype(bool)
            split_results[split] = _summarize_split(
                builder,
                observed_masks,
                candidate_masks,
                feature_names,
            )
    summary = {
        "model_version": str(config.get("model_version", "unknown")),
        "experiment": str(config.get("experiment", "unknown")),
        "graph_id": graph_id,
        "scope": "candidate_nodes_train_validation_only",
        "feature_names": feature_names,
        "splits": split_results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _summarize_split(
    builder: SnapshotFeatureBuilder,
    observed_masks: np.ndarray,
    candidate_masks: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    accumulators = {
        name: {
            "count": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "nonzero": 0,
            "varying_examples": 0,
            "example_mean_sum": 0.0,
            "example_mean_sum_squares": 0.0,
        }
        for name in feature_names
    }
    pairwise_equal = {
        (left, right): True
        for left_index, left in enumerate(feature_names)
        for right in feature_names[left_index + 1 :]
    }
    for observed, candidates in zip(
        observed_masks, candidate_masks, strict=True
    ):
        values = builder.build(
            observed, feature_names, candidate_mask=candidates
        )
        candidate_values = values[candidates]
        if not len(candidate_values):
            raise ValueError("Every diagnostic example must contain candidates.")
        for index, name in enumerate(feature_names):
            column = candidate_values[:, index].astype(np.float64)
            accumulator = accumulators[name]
            accumulator["count"] += len(column)
            accumulator["sum"] += float(column.sum())
            accumulator["sum_squares"] += float(np.square(column).sum())
            accumulator["min"] = min(accumulator["min"], float(column.min()))
            accumulator["max"] = max(accumulator["max"], float(column.max()))
            accumulator["nonzero"] += int(np.count_nonzero(column))
            accumulator["varying_examples"] += int(float(column.min()) != float(column.max()))
            example_mean = float(column.mean())
            accumulator["example_mean_sum"] += example_mean
            accumulator["example_mean_sum_squares"] += example_mean * example_mean
        for left_index, left in enumerate(feature_names):
            for right_index, right in enumerate(
                feature_names[left_index + 1 :], start=left_index + 1
            ):
                if pairwise_equal[(left, right)] and not np.array_equal(
                    candidate_values[:, left_index], candidate_values[:, right_index]
                ):
                    pairwise_equal[(left, right)] = False

    example_count = len(observed_masks)
    result = {"example_count": example_count, "features": {}}
    for name, values in accumulators.items():
        count = int(values["count"])
        mean = values["sum"] / count
        variance = max(values["sum_squares"] / count - mean * mean, 0.0)
        example_mean = values["example_mean_sum"] / example_count
        example_mean_variance = max(
            values["example_mean_sum_squares"] / example_count
            - example_mean * example_mean,
            0.0,
        )
        result["features"][name] = {
            "candidate_count": count,
            "mean": mean,
            "std": variance**0.5,
            "min": values["min"],
            "max": values["max"],
            "nonzero_fraction": values["nonzero"] / count,
            "varying_example_fraction": values["varying_examples"] / example_count,
            "example_mean_std": example_mean_variance**0.5,
        }
    result["exact_duplicate_feature_pairs"] = [
        [left, right]
        for (left, right), equal in pairwise_equal.items()
        if equal
    ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = diagnose_snapshot_features(args.config, args.output)
    compact = {
        split: {
            name: {
                "std": values["std"],
                "varying_example_fraction": values["varying_example_fraction"],
                "example_mean_std": values["example_mean_std"],
            }
            for name, values in details["features"].items()
        }
        for split, details in summary["splits"].items()
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
