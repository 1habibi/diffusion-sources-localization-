"""Evaluate a fixed Joint GCN checkpoint under paired observation conditions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import graph_to_edge_index, load_graph_archive
from .inference import predict_joint
from .metrics import set_metrics, source_set_distances
from .models import JointSourceCountGCN


def build_condition_data(
    graph: nx.Graph,
    infected_mask: np.ndarray,
    source_labels: np.ndarray,
    fraction: float,
    false_positive_fraction: float,
    rng: np.random.Generator,
    feature_indices: list[int],
):
    """Build one paired observation while always retaining true sources."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if false_positive_fraction < 0.0:
        raise ValueError("false_positive_fraction must be non-negative.")
    sources = set(np.flatnonzero(source_labels).tolist())
    infected = set(np.flatnonzero(infected_mask).tolist())
    non_sources = sorted(infected - sources)
    target_count = max(len(sources), round(fraction * len(infected)))
    sample_count = min(target_count - len(sources), len(non_sources))
    sampled = set(
        rng.choice(non_sources, size=sample_count, replace=False).tolist()
        if sample_count
        else []
    )
    observed_true = sources | sampled
    susceptible = sorted(set(graph) - infected)
    false_count = min(
        round(false_positive_fraction * len(observed_true)), len(susceptible)
    )
    false_positive = set(
        rng.choice(susceptible, size=false_count, replace=False).tolist()
        if false_count
        else []
    )
    candidates = observed_true | false_positive
    max_degree = max(dict(graph.degree()).values())
    feature_matrix = np.zeros((graph.number_of_nodes(), 2), dtype=np.float32)
    feature_matrix[list(candidates), 0] = 1.0
    denominator = np.log1p(max_degree) or 1.0
    for node in graph:
        feature_matrix[node, 1] = np.log1p(graph.degree(node)) / denominator

    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(feature_matrix[:, feature_indices]).float(),
        edge_index=graph_to_edge_index(graph),
        candidate_mask=torch.tensor(
            [node in candidates for node in graph], dtype=torch.bool
        ),
        source_labels=torch.from_numpy(source_labels).float(),
        source_count=torch.tensor(len(sources), dtype=torch.long),
        observed_mask=torch.tensor(
            [node in candidates for node in graph], dtype=torch.bool
        ),
    )


def evaluate_robustness(
    data_dir: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    fractions: tuple[float, ...] = (1.0, 0.75, 0.5),
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.1),
    seed: int = 2026,
) -> dict:
    """Evaluate one selected checkpoint on a paired fraction/noise grid."""
    data_path, run_path, output_path = Path(data_dir), Path(run_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _, graph = load_graph_archive(data_path / "graph.npz")
    archive = np.load(data_path / "test.npz", allow_pickle=False)
    if "infected_masks" not in archive:
        raise ValueError("Dataset must be regenerated with infected_masks.")
    run_metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    feature_indices = [int(value) for value in run_metrics["feature_indices"]]
    config = __import__("yaml").safe_load(
        (run_path / "config.yaml").read_text(encoding="utf-8")
    )
    model = JointSourceCountGCN(
        input_dim=len(feature_indices),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(torch.load(run_path / "best_model.pt", map_location="cpu"))
    model.eval()
    rows = []
    with torch.no_grad():
        for fraction in fractions:
            for noise in noise_levels:
                for index in range(len(archive["source_counts"])):
                    data = build_condition_data(
                        graph,
                        archive["infected_masks"][index],
                        archive["source_labels"][index],
                        fraction,
                        noise,
                        np.random.default_rng(
                            seed + index + round(fraction * 1000) * 100 + round(noise * 1000)
                        ),
                        feature_indices,
                    )
                    source_logits, count_logits = model(data)
                    prediction = predict_joint(
                        source_logits, count_logits, data.candidate_mask
                    )
                    true_sources = set(np.flatnonzero(archive["source_labels"][index]))
                    rows.append(
                        {
                            "fraction": fraction,
                            "noise": noise,
                            "example": index,
                            "k": int(archive["source_counts"][index]),
                            **set_metrics(true_sources, prediction.sources),
                            **source_set_distances(graph, true_sources, prediction.sources),
                        }
                    )
    aggregates = _aggregate(rows)
    summary = {"seed": seed, "conditions": aggregates}
    _write_csv(output_path / "robustness_predictions.csv", rows)
    _write_csv(output_path / "robustness_table.csv", list(aggregates.values()))
    (output_path / "robustness_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot(aggregates, output_path / "robustness_heatmaps.png")
    return summary


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["fraction"], row["noise"])].append(row)
    return {
        f"fraction_{fraction}_noise_{noise}": {
            "fraction": fraction,
            "noise": noise,
            "f1": float(np.mean([row["f1"] for row in selected])),
            "count_accuracy": float(
                np.mean([row["count_accuracy"] for row in selected])
            ),
            "count_mae": float(np.mean([row["count_mae"] for row in selected])),
            "symmetric_set_distance": float(
                np.mean([row["symmetric_set_distance"] for row in selected])
            ),
        }
        for (fraction, noise), selected in grouped.items()
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(aggregates: dict[str, dict], output_path: Path) -> None:
    fractions = sorted({row["fraction"] for row in aggregates.values()}, reverse=True)
    noise_levels = sorted({row["noise"] for row in aggregates.values()})
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in (
        (axes[0], "f1", "Macro-F1"),
        (axes[1], "count_accuracy", "Count accuracy"),
        (axes[2], "symmetric_set_distance", "Graph distance"),
    ):
        matrix = np.asarray(
            [
                [
                    next(
                        row[metric]
                        for row in aggregates.values()
                        if row["fraction"] == fraction and row["noise"] == noise
                    )
                    for noise in noise_levels
                ]
                for fraction in fractions
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(noise_levels)), [f"{value:.0%}" for value in noise_levels])
        axis.set_yticks(range(len(fractions)), [f"{value:.0%}" for value in fractions])
        axis.set_xlabel("false-positive noise")
        axis.set_ylabel("observed infected")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    summary = evaluate_robustness(args.data, args.run, args.output, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
