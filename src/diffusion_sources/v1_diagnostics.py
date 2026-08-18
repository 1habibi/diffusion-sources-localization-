"""Diagnostics for completed v1 predictions without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("f1", "exact_set_accuracy", "count_accuracy", "count_mae", "symmetric_set_distance")
SEEDS = (7026, 7027, 7028)
METHODS = {
    "joint_no_consistency": ("joint_estimated_k", "joint"),
    "node_only": ("node_thresholded", "node"),
}


def diagnose_v1(
    data_dir: str | Path,
    joint_root: str | Path,
    node_root: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_repeats: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build confusion, bootstrap and candidate-size diagnostics."""
    if bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100.")
    data = np.load(Path(data_dir) / "test.npz", allow_pickle=False)
    candidate_sizes = data["candidate_masks"].sum(axis=1).astype(int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict[str, Any]]] = {}
    for experiment, (method, kind) in METHODS.items():
        root = Path(joint_root if kind == "joint" else node_root)
        records[experiment] = _load_prediction_records(root, method)

    confusion = {
        experiment: _confusion(records_for_method)
        for experiment, records_for_method in records.items()
    }
    bootstrap = {
        experiment: _bootstrap(
            records_for_method,
            bootstrap_repeats=bootstrap_repeats,
            seed=seed + index,
        )
        for index, (experiment, records_for_method) in enumerate(records.items())
    }
    candidate_analysis = {
        experiment: _candidate_size_analysis(records_for_method, candidate_sizes)
        for experiment, records_for_method in records.items()
    }
    summary = {
        "test_examples": int(len(candidate_sizes)),
        "candidate_size": {
            "min": int(candidate_sizes.min()),
            "median": float(np.median(candidate_sizes)),
            "mean": float(candidate_sizes.mean()),
            "max": int(candidate_sizes.max()),
        },
        "confusion": confusion,
        "bootstrap": bootstrap,
        "candidate_analysis": candidate_analysis,
    }
    (output_path / "v1_diagnostics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_confusion_csv(confusion, output_path / "count_confusion.csv")
    _write_bootstrap_csv(bootstrap, output_path / "bootstrap_summary.csv")
    _write_candidate_csv(candidate_analysis, output_path / "candidate_size_analysis.csv")
    _plot_confusion(confusion, output_path / "count_confusion.png")
    _plot_candidate_f1(candidate_analysis, output_path / "candidate_size_f1.png")
    return summary


def _load_prediction_records(root: Path, method: str) -> list[list[dict[str, Any]]]:
    by_seed: list[list[dict[str, Any]]] = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as file:
            rows = [
                {
                    key: (
                        value
                        if key == "method"
                        else int(value)
                        if key in {"example", "k", "predicted_k"}
                        else float(value)
                    )
                    for key, value in row.items()
                }
                for row in csv.DictReader(file)
                if row["method"] == method
            ]
        rows.sort(key=lambda row: row["example"])
        if not rows:
            raise ValueError(f"No rows for method {method} in {path}.")
        by_seed.append(rows)
    expected = [row["example"] for row in by_seed[0]]
    if any([row["example"] for row in rows] != expected for rows in by_seed[1:]):
        raise ValueError(f"Prediction examples are not aligned for {method}.")
    return by_seed


def _confusion(records_by_seed: list[list[dict[str, Any]]]) -> dict[str, Any]:
    matrix = np.zeros((3, 3), dtype=int)
    for rows in records_by_seed:
        for row in rows:
            matrix[int(row["k"]) - 1, int(row["predicted_k"]) - 1] += 1
    return {"labels": [1, 2, 3], "matrix": matrix.tolist()}


def _bootstrap(
    records_by_seed: list[list[dict[str, Any]]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    arrays = {
        metric: np.asarray(
            [[row[metric] for row in rows] for rows in records_by_seed], dtype=float
        )
        for metric in METRICS
    }
    count = arrays["f1"].shape[1]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(bootstrap_repeats, count))
    result = {}
    for metric, values in arrays.items():
        samples = values[:, indices].mean(axis=(1, 2))
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.mean(axis=0).std(ddof=1)),
            "ci95_low": float(np.percentile(samples, 2.5)),
            "ci95_high": float(np.percentile(samples, 97.5)),
        }
    return result


def _candidate_size_analysis(
    records_by_seed: list[list[dict[str, Any]]], candidate_sizes: np.ndarray
) -> list[dict[str, Any]]:
    bins = ((5, 10), (11, 20), (21, 50), (51, int(candidate_sizes.max())))
    rows: list[dict[str, Any]] = []
    for lower, upper in bins:
        selected = [
            row
            for records in records_by_seed
            for row in records
            if lower <= candidate_sizes[row["example"]] <= upper
        ]
        if not selected:
            continue
        rows.append(
            {
                "candidate_bin": f"{lower}-{upper}",
                "lower": lower,
                "upper": upper,
                "examples_with_seeds": len(selected),
                "f1": float(np.mean([row["f1"] for row in selected])),
                "exact_set_accuracy": float(
                    np.mean([row["exact_set_accuracy"] for row in selected])
                ),
                "symmetric_set_distance": float(
                    np.mean([row["symmetric_set_distance"] for row in selected])
                ),
            }
        )
    return rows


def _write_confusion_csv(confusion: dict, path: Path) -> None:
    rows = []
    for experiment, details in confusion.items():
        for true_k, values in zip(details["labels"], details["matrix"], strict=True):
            for predicted_k, count in zip(details["labels"], values, strict=True):
                rows.append(
                    {
                        "experiment": experiment,
                        "true_k": true_k,
                        "predicted_k": predicted_k,
                        "count": count,
                    }
                )
    _write_csv(path, rows)


def _write_bootstrap_csv(bootstrap: dict, path: Path) -> None:
    rows = [
        {"experiment": experiment, "metric": metric, **values}
        for experiment, metrics in bootstrap.items()
        for metric, values in metrics.items()
    ]
    _write_csv(path, rows)


def _write_candidate_csv(candidate_analysis: dict, path: Path) -> None:
    rows = [
        {"experiment": experiment, **row}
        for experiment, analysis in candidate_analysis.items()
        for row in analysis
    ]
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(confusion: dict, path: Path) -> None:
    figure, axes = plt.subplots(1, len(confusion), figsize=(10, 4))
    axes = np.atleast_1d(axes)
    for axis, (experiment, details) in zip(axes, confusion.items(), strict=True):
        image = axis.imshow(details["matrix"], cmap="Blues")
        axis.set_title(experiment)
        axis.set_xlabel("predicted k")
        axis.set_ylabel("true k")
        axis.set_xticks(range(3), [1, 2, 3])
        axis.set_yticks(range(3), [1, 2, 3])
        for row in range(3):
            for column in range(3):
                axis.text(column, row, details["matrix"][row][column], ha="center", va="center")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_candidate_f1(candidate_analysis: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for experiment, rows in candidate_analysis.items():
        axis.plot(
            [row["candidate_bin"] for row in rows],
            [row["f1"] for row in rows],
            marker="o",
            label=experiment,
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("candidate set size")
    axis.set_ylabel("F1")
    axis.set_title("v1 quality by candidate set size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--joint-root", required=True, type=Path)
    parser.add_argument("--node-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args(argv)
    summary = diagnose_v1(
        args.data,
        args.joint_root,
        args.node_root,
        args.output,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(json.dumps(summary["candidate_size"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
