"""Analyze generated cascade splits and emit pilot parameter diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import load_graph_archive


def analyze_dataset(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Summarize graph, cascade size, candidates, k balance, and disk usage."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graph_id, graph = load_graph_archive(data_path / "graph.npz")
    split_names = [
        name for name in ("train", "validation", "test")
        if (data_path / f"{name}.npz").exists()
    ]
    if not split_names:
        raise ValueError("No generated split archives were found.")

    rows: list[dict] = []
    split_summaries: dict[str, dict] = {}
    for split in split_names:
        archive_path = data_path / f"{split}.npz"
        archive = np.load(archive_path, allow_pickle=False)
        infected_sizes = archive["infected_masks"].sum(axis=1)
        candidate_sizes = archive["candidate_masks"].sum(axis=1)
        source_counts = archive["source_counts"]
        probabilities = archive["probabilities"]
        fractions = archive["observation_fractions"]
        for index in range(len(source_counts)):
            rows.append(
                {
                    "split": split,
                    "example": index,
                    "k": int(source_counts[index]),
                    "infected": int(infected_sizes[index]),
                    "candidates": int(candidate_sizes[index]),
                    "infected_fraction": float(infected_sizes[index] / graph.number_of_nodes()),
                    "probability": float(probabilities[index]),
                    "observation_fraction": float(fractions[index]),
                }
            )
        split_summaries[split] = {
            "examples": int(len(source_counts)),
            "k_counts": {
                str(k): int(np.sum(source_counts == k)) for k in (1, 2, 3)
            },
            "infected": _distribution(infected_sizes),
            "candidates": _distribution(candidate_sizes),
            "archive_bytes": archive_path.stat().st_size,
            "probability_counts": _value_counts(probabilities),
            "observation_fraction_counts": _value_counts(fractions),
        }

    summary = {
        "graph": {
            "id": graph_id,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": float(
                2 * graph.number_of_edges()
                / (graph.number_of_nodes() * (graph.number_of_nodes() - 1))
            ),
            "average_degree": float(
                2 * graph.number_of_edges() / graph.number_of_nodes()
            ),
        },
        "splits": split_summaries,
        "total_examples": len(rows),
        "total_archive_bytes": sum(
            (data_path / f"{split}.npz").stat().st_size for split in split_names
        ),
    }
    (output_path / "dataset_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_csv(output_path / "cascade_statistics.csv", rows)
    _plot_distributions(rows, output_path / "cascade_distributions.png")
    return summary


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _value_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(float(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_distributions(rows: list[dict], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist([row["infected"] for row in rows], bins=20, color="#3D7A80")
    axes[0].set_title("True cascade size")
    axes[1].hist([row["candidates"] for row in rows], bins=20, color="#E07A5F")
    axes[1].set_title("Candidate set size")
    k_values = [1, 2, 3]
    axes[2].bar(
        k_values,
        [sum(row["k"] == k for row in rows) for k in k_values],
        color="#8D99AE",
    )
    axes[2].set_xticks(k_values)
    axes[2].set_title("Source count balance")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze_dataset(args.data, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
