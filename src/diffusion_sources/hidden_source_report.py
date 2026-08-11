"""Compare the primary visible-source setup with the hidden-source setup."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_hidden_source_report(
    primary_run: str | Path,
    hidden_run: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Build a separate comparison without merging incompatible candidate modes."""
    rows = []
    for mode, label, run_path in (
        ("visible_sources", "All sources visible, M=I_obs", Path(primary_run)),
        ("hidden_source", "One source hidden, M=V", Path(hidden_run)),
    ):
        run = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        values = run["prediction_metrics"]["joint_estimated_k"]["all"]
        rows.append(
            {
                "mode": mode,
                "label": label,
                "seed": run["seed"],
                "f1": values["f1"],
                "precision": values["precision"],
                "recall": values["recall"],
                "exact_set_accuracy": values["exact_set_accuracy"],
                "count_accuracy": values["count_accuracy"],
                "count_mae": values["count_mae"],
                "symmetric_set_distance": values["symmetric_set_distance"],
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path = output_path / "hidden_source_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_path / "hidden_source_metrics.json"
    json_path.write_text(json.dumps({"modes": rows}, indent=2), encoding="utf-8")
    figure_path = _plot(rows, output_path / "hidden_source_comparison.png")
    return [table_path, json_path, figure_path]


def _plot(rows: list[dict], output_path: Path) -> Path:
    labels = [row["label"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric, title, limit in (
        (axes[0], "f1", "Macro-F1", (0, 1)),
        (axes[1], "count_accuracy", "Count accuracy", (0, 1)),
        (axes[2], "symmetric_set_distance", "Graph distance", None),
    ):
        axis.bar(labels, [row[metric] for row in rows], color=["#3D7A80", "#E07A5F"])
        axis.set_title(title)
        if limit:
            axis.set_ylim(*limit)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Visible-source and hidden-source setups (reported separately)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--hidden-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = build_hidden_source_report(
        args.primary_run, args.hidden_run, args.output
    )
    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
