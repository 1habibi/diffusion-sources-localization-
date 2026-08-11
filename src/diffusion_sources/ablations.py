"""Aggregate mandatory model ablations into report-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ABLATION_SPECS = (
    ("node_only", "node_thresholded", "Without count head"),
    ("without_consistency", "joint_estimated_k", "Without consistency loss"),
    ("joint_full", "joint_estimated_k", "Full joint model"),
)


def build_ablation_report(
    node_run: str | Path,
    without_consistency_run: str | Path,
    joint_run: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Compare the three mandatory estimated-k model variants."""
    run_paths = {
        "node_only": Path(node_run),
        "without_consistency": Path(without_consistency_run),
        "joint_full": Path(joint_run),
    }
    rows = []
    for key, prediction_key, label in ABLATION_SPECS:
        metrics = json.loads(
            (run_paths[key] / "metrics.json").read_text(encoding="utf-8")
        )
        prediction = metrics["prediction_metrics"][prediction_key]["all"]
        rows.append(
            {
                "variant": key,
                "label": label,
                "seed": metrics["seed"],
                "f1": prediction["f1"],
                "precision": prediction["precision"],
                "recall": prediction["recall"],
                "exact_set_accuracy": prediction["exact_set_accuracy"],
                "count_accuracy": prediction["count_accuracy"],
                "count_mae": prediction["count_mae"],
                "symmetric_set_distance": prediction["symmetric_set_distance"],
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path = output_path / "ablation_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_path / "ablation_metrics.json"
    json_path.write_text(json.dumps({"variants": rows}, indent=2), encoding="utf-8")
    figure_path = _plot_ablation(rows, output_path / "ablation_comparison.png")
    return [table_path, json_path, figure_path]


def _plot_ablation(rows: list[dict], output_path: Path) -> Path:
    labels = [row["label"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plots = (
        ("f1", "Macro-F1", (0, 1)),
        ("count_accuracy", "Count accuracy", (0, 1)),
        ("symmetric_set_distance", "Graph distance", None),
    )
    colors = ["#8D99AE", "#E07A5F", "#3D7A80"]
    for axis, (metric, title, limits) in zip(axes, plots, strict=True):
        axis.bar(labels, [row[metric] for row in rows], color=colors)
        axis.set_title(title)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Mandatory ablation study, estimated-k")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-run", required=True, type=Path)
    parser.add_argument("--without-consistency-run", required=True, type=Path)
    parser.add_argument("--joint-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = build_ablation_report(
        args.node_run,
        args.without_consistency_run,
        args.joint_run,
        args.output,
    )
    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
