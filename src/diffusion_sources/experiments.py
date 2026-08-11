"""Run reproducible multi-seed experiment series and aggregate their metrics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .node_train_cli import run_node_training
from .train_cli import load_train_config, run_training


AGGREGATED_METRICS = (
    "f1",
    "precision",
    "recall",
    "exact_set_accuracy",
    "count_accuracy",
    "count_mae",
    "symmetric_set_distance",
)

BOUNDED_METRICS = {
    "f1",
    "precision",
    "recall",
    "exact_set_accuracy",
    "count_accuracy",
}


def run_experiment_series(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Run all configured variants for all seeds and aggregate estimated-k metrics."""
    series_path = Path(config_path)
    series = yaml.safe_load(series_path.read_text(encoding="utf-8"))
    if not isinstance(series, dict) or not series.get("seeds") or not series.get("experiments"):
        raise ValueError("Series config requires non-empty seeds and experiments.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    project_root = series_path.resolve().parent.parent
    run_records: list[dict[str, Any]] = []

    for experiment_name, specification in series["experiments"].items():
        prediction_key = str(specification["prediction"])
        kind = str(specification["kind"])
        base_config_path = project_root / specification["config"]
        base_config = load_train_config(base_config_path)
        for seed_value in series["seeds"]:
            seed = int(seed_value)
            run_dir = output_path / "runs" / experiment_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            if reuse_existing and metrics_path.exists():
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                run_config = copy.deepcopy(base_config)
                run_config["training"]["seed"] = seed
                run_config["experiment"] = experiment_name
                result = (
                    run_training(run_config, run_dir)
                    if kind == "joint"
                    else run_node_training(run_config, run_dir)
                )
            values = result["prediction_metrics"][prediction_key]["all"]
            run_records.append(
                {
                    "experiment": experiment_name,
                    "kind": kind,
                    "prediction": prediction_key,
                    "seed": seed,
                    **{metric: float(values[metric]) for metric in AGGREGATED_METRICS},
                }
            )

    aggregates = aggregate_seed_metrics(run_records)
    summary = {
        "seeds": [int(seed) for seed in series["seeds"]],
        "runs": run_records,
        "aggregates": aggregates,
    }
    (output_path / "series_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output_path / "series_runs.csv", run_records)
    aggregate_rows = _flatten_aggregates(aggregates)
    _write_csv(output_path / "series_summary.csv", aggregate_rows)
    _plot_series(aggregates, output_path / "series_comparison.png")
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(series, sort_keys=False), encoding="utf-8"
    )
    return summary


def aggregate_seed_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate mean, sample std and two-sided 95% t interval by experiment."""
    if not records:
        raise ValueError("At least one experiment record is required.")
    result: dict[str, dict[str, dict[str, float]]] = {}
    experiments = sorted({str(record["experiment"]) for record in records})
    for experiment in experiments:
        selected = [record for record in records if record["experiment"] == experiment]
        result[experiment] = {}
        for metric in AGGREGATED_METRICS:
            values = np.asarray([float(record[metric]) for record in selected], dtype=float)
            mean = float(values.mean())
            if len(values) > 1:
                std = float(values.std(ddof=1))
                margin = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
            else:
                std = 0.0
                margin = 0.0
            lower = mean - margin
            upper = mean + margin
            if metric in BOUNDED_METRICS:
                lower = max(0.0, lower)
                upper = min(1.0, upper)
            else:
                lower = max(0.0, lower)
            result[experiment][metric] = {
                "mean": mean,
                "std": std,
                "ci95_low": lower,
                "ci95_high": upper,
            }
    return result


def _flatten_aggregates(aggregates: dict) -> list[dict[str, Any]]:
    return [
        {"experiment": experiment, "metric": metric, **statistics}
        for experiment, metrics in aggregates.items()
        for metric, statistics in metrics.items()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_series(aggregates: dict, output_path: Path) -> Path:
    experiments = list(aggregates)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title, limits in (
        (axes[0], "f1", "Estimated-k macro-F1", (0, 1)),
        (axes[1], "count_accuracy", "Count accuracy", (0, 1)),
        (axes[2], "symmetric_set_distance", "Graph distance", None),
    ):
        means = [aggregates[name][metric]["mean"] for name in experiments]
        errors = [
            aggregates[name][metric]["ci95_high"] - aggregates[name][metric]["mean"]
            for name in experiments
        ]
        axis.bar(experiments, means, yerr=errors, capsize=4, color="#3D7A80")
        axis.set_title(title)
        if limits:
            axis.set_ylim(*limits)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Three-seed experiment series with 95% t intervals")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    result = run_experiment_series(
        args.config, args.output, reuse_existing=args.reuse_existing
    )
    print(json.dumps(result["aggregates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
