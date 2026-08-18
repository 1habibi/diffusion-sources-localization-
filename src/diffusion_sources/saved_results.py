"""Aggregate completed multi-seed runs imported from an external GPU runtime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiments import AGGREGATED_METRICS, aggregate_seed_metrics


REQUIRED_ARTIFACTS = (
    "best_model.pt",
    "last_checkpoint.pt",
    "history.csv",
    "history.json",
    "metrics.json",
    "test_predictions.csv",
    "config.yaml",
)


def aggregate_saved_results(
    config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Validate imported runs and build estimated-k and oracle-k summaries."""
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("seeds") or not config.get("experiments"):
        raise ValueError("Saved-results config requires seeds and experiments.")

    project_root = config_file.resolve().parent.parent
    runs_root = project_root / str(config["runs_root"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    estimated_records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    artifact_checks: list[dict[str, Any]] = []
    for experiment, specification in config["experiments"].items():
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            run_dir = runs_root / str(specification["directory"]) / f"seed_{seed}"
            missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
            artifact_checks.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "complete": not missing,
                    "missing": missing,
                }
            )
            if missing:
                raise FileNotFoundError(f"Incomplete run {run_dir}: missing {missing}")

            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            if int(metrics["seed"]) != seed:
                raise ValueError(f"Seed mismatch in {run_dir / 'metrics.json'}.")
            estimated_records.extend(
                _prediction_records(
                    metrics,
                    experiment,
                    seed,
                    str(specification["estimated_prediction"]),
                )
            )
            oracle_records.extend(
                _prediction_records(
                    metrics,
                    experiment,
                    seed,
                    str(specification["oracle_prediction"]),
                )
            )

    estimated_by_k = _aggregate_by_k(estimated_records)
    oracle_by_k = _aggregate_by_k(oracle_records)
    baselines = _load_baselines(project_root, config.get("baselines"))
    summary = {
        "seeds": [int(seed) for seed in config["seeds"]],
        "artifact_checks": artifact_checks,
        "estimated_runs": estimated_records,
        "estimated_aggregates": estimated_by_k,
        "oracle_runs": oracle_records,
        "oracle_aggregates": oracle_by_k,
        "baselines": baselines,
    }

    (output_path / "saved_results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output_path / "estimated_runs.csv", estimated_records)
    _write_csv(
        output_path / "estimated_summary.csv",
        _flatten_aggregates(estimated_by_k),
    )
    _write_csv(output_path / "oracle_runs.csv", oracle_records)
    _write_csv(
        output_path / "oracle_summary.csv",
        _flatten_aggregates(oracle_by_k),
    )
    _write_csv(
        output_path / "artifact_checks.csv",
        [
            {**record, "missing": ";".join(record["missing"])}
            for record in artifact_checks
        ],
    )
    _plot_estimated_overview(
        estimated_by_k["all"], output_path / "estimated_k_comparison.png"
    )
    _plot_f1_by_k(estimated_by_k, output_path / "estimated_f1_by_k.png")
    _plot_oracle_methods(
        oracle_by_k["all"], baselines, output_path / "oracle_k_methods.png"
    )
    return summary


def _prediction_records(
    metrics: dict[str, Any], experiment: str, seed: int, prediction: str
) -> list[dict[str, Any]]:
    groups = metrics["prediction_metrics"][prediction]
    return [
        {
            "experiment": experiment,
            "prediction": prediction,
            "seed": seed,
            "k": str(group),
            **{metric: float(values[metric]) for metric in AGGREGATED_METRICS},
        }
        for group, values in groups.items()
    ]


def _aggregate_by_k(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(record["k"]) for record in records}, key=lambda value: (value != "all", value))
    return {
        group: aggregate_seed_metrics(
            [record for record in records if str(record["k"]) == group]
        )
        for group in groups
    }


def _load_baselines(project_root: Path, configured_path: Any) -> dict[str, Any]:
    if configured_path is None:
        return {}
    path = project_root / str(configured_path)
    return json.loads(path.read_text(encoding="utf-8"))["methods"]


def _flatten_aggregates(aggregates_by_k: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "k": group,
            "experiment": experiment,
            "metric": metric,
            **statistics,
        }
        for group, aggregates in aggregates_by_k.items()
        for experiment, metrics in aggregates.items()
        for metric, statistics in metrics.items()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_estimated_overview(aggregates: dict[str, Any], path: Path) -> None:
    names = list(aggregates)
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for axis, metric, title, limits in (
        (axes[0], "f1", "Estimated-k F1", (0, 1)),
        (axes[1], "count_accuracy", "Count accuracy", (0, 1)),
        (axes[2], "exact_set_accuracy", "Exact set accuracy", (0, 0.12)),
        (axes[3], "symmetric_set_distance", "Graph distance", None),
    ):
        values = [aggregates[name][metric]["mean"] for name in names]
        errors = [
            aggregates[name][metric]["ci95_high"] - aggregates[name][metric]["mean"]
            for name in names
        ]
        axis.bar(names, values, yerr=errors, capsize=4, color=("#3D7A80", "#E07A5F"))
        axis.set_title(title)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_f1_by_k(aggregates_by_k: dict[str, Any], path: Path) -> None:
    groups = [group for group in ("1", "2", "3") if group in aggregates_by_k]
    names = list(aggregates_by_k[groups[0]])
    positions = range(len(groups))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for index, name in enumerate(names):
        values = [aggregates_by_k[group][name]["f1"]["mean"] for group in groups]
        axis.bar(
            [position + (index - 0.5) * width for position in positions],
            values,
            width=width,
            label=name,
        )
    axis.set_xticks(list(positions), [f"k={group}" for group in groups])
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.set_title("Estimated-k F1 by true source count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_oracle_methods(
    aggregates: dict[str, Any], baselines: dict[str, Any], path: Path
) -> None:
    neural_names = list(aggregates)
    names = [*baselines, *neural_names]
    f1_values = [
        *[baselines[name]["all"]["f1"] for name in baselines],
        *[aggregates[name]["f1"]["mean"] for name in neural_names],
    ]
    distances = [
        *[baselines[name]["all"]["symmetric_set_distance"] for name in baselines],
        *[aggregates[name]["symmetric_set_distance"]["mean"] for name in neural_names],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(names, f1_values, color="#3D7A80")
    axes[0].set_title("Oracle-k F1")
    axes[0].set_ylim(0, 1)
    axes[1].bar(names, distances, color="#E07A5F")
    axes[1].set_title("Oracle-k graph distance")
    for axis in axes:
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = aggregate_saved_results(args.config, args.output)
    print(json.dumps(summary["estimated_aggregates"]["all"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
