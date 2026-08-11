"""Build report-ready figures and summary tables from saved run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_report(
    run_dir: str | Path,
    output_dir: str | Path,
    baseline_dir: str | Path | None = None,
    node_run_dir: str | Path | None = None,
) -> list[Path]:
    """Create deterministic PNG figures and a compact final metrics table."""
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = _read_history(run_path / "history.csv")
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    outputs = [
        _plot_history(
            history,
            output_path / "loss_curves.png",
            ("loss", "source_loss", "count_loss", "consistency_loss"),
            "Loss components",
        ),
        _plot_history(
            history,
            output_path / "quality_curves.png",
            ("macro_f1", "pr_auc", "count_accuracy"),
            "Training and validation quality",
        ),
        _write_final_metrics(metrics, output_path / "final_metrics.csv"),
    ]

    if baseline_dir is not None:
        baseline_path = Path(baseline_dir) / "baseline_metrics.json"
        baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        outputs.append(
            _plot_baselines(
                baseline_metrics, output_path / "baseline_comparison.png"
            )
        )
        outputs.extend(
            _write_all_method_outputs(
                metrics,
                baseline_metrics,
                output_path,
                (
                    json.loads(
                        (Path(node_run_dir) / "metrics.json").read_text(encoding="utf-8")
                    )
                    if node_run_dir is not None
                    else None
                ),
            )
        )
    return outputs


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Training history is empty.")
    return rows


def _plot_history(
    rows: list[dict[str, str]],
    output_path: Path,
    metrics: tuple[str, ...],
    title: str,
) -> Path:
    figure, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.7 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics, strict=True):
        for split, color in (("train", "#234E70"), ("validation", "#E07A5F")):
            selected = [row for row in rows if row["split"] == split]
            axis.plot(
                [int(row["epoch"]) for row in selected],
                [float(row[metric]) for row in selected],
                label=split,
                color=color,
            )
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _write_final_metrics(metrics: dict, output_path: Path) -> Path:
    rows = []
    for split, values in metrics["metrics"].items():
        rows.append({"split": split, **values})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _plot_baselines(metrics: dict, output_path: Path) -> Path:
    methods = sorted(metrics["methods"])
    f1_values = [metrics["methods"][method]["all"]["f1"] for method in methods]
    distances = [
        metrics["methods"][method]["all"]["symmetric_set_distance"]
        for method in methods
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(methods, f1_values, color="#3D7A80")
    axes[0].set_title("Oracle-k macro-F1")
    axes[0].set_ylim(0, 1)
    axes[1].bar(methods, distances, color="#E07A5F")
    axes[1].set_title("Symmetric graph distance")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _write_all_method_outputs(
    run_metrics: dict,
    baseline_metrics: dict,
    output_dir: Path,
    node_metrics: dict | None = None,
) -> list[Path]:
    methods = {
        **baseline_metrics["methods"],
        **run_metrics.get("prediction_metrics", {}),
        **(node_metrics.get("prediction_metrics", {}) if node_metrics else {}),
    }
    if not methods:
        return []
    table_path = output_dir / "all_methods.csv"
    rows = []
    for method, groups in methods.items():
        for group, values in groups.items():
            rows.append({"method": method, "k": group, **values})
    all_fields = ["method", "k"]
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output_dir / "all_methods_comparison.png"
    method_names = sorted(methods)
    f1_values = [methods[method]["all"]["f1"] for method in method_names]
    distances = [
        methods[method]["all"]["symmetric_set_distance"] for method in method_names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(method_names, f1_values, color="#3D7A80")
    axes[0].set_title("All methods: macro-F1")
    axes[0].set_ylim(0, 1)
    axes[1].bar(method_names, distances, color="#E07A5F")
    axes[1].set_title("All methods: symmetric graph distance")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return [table_path, figure_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--node-run", type=Path)
    args = parser.parse_args(argv)
    outputs = build_report(args.run, args.output, args.baselines, args.node_run)
    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
