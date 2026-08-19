"""Evaluate a selected Joint GCN checkpoint on a held-out graph topology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import load_graph_archive, load_pyg_split
from .inference import predict_joint, predict_oracle_k
from .metrics import set_metrics, source_radius_hits, source_set_distances
from .models import JointSourceCountGCN
from .train_cli import _aggregate_prediction_rows


def evaluate_transfer(
    external_data_dir: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Evaluate an unchanged checkpoint on a generated external test split."""
    external_path = Path(external_data_dir)
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    external_graph_id, graph = load_graph_archive(external_path / "graph.npz")
    run_metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    feature_indices = [int(value) for value in run_metrics["feature_indices"]]
    examples = load_pyg_split(
        external_path / "test.npz", graph, feature_indices=feature_indices
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
        for index, data in enumerate(tqdm(examples, desc="External graph", unit="cascade")):
            source_logits, count_logits = model(data)
            true_sources = frozenset(
                torch.nonzero(data.source_labels, as_tuple=False).flatten().tolist()
            )
            true_count = int(data.source_count.item())
            predictions = {
                "external_estimated_k": predict_joint(
                    source_logits, count_logits, data.candidate_mask
                ),
                "external_oracle_k": predict_oracle_k(
                    source_logits, data.candidate_mask, true_count
                ),
            }
            for method, prediction in predictions.items():
                rows.append(
                    {
                        "example": index,
                        "k": true_count,
                        "predicted_k": prediction.source_count,
                        "method": method,
                        **set_metrics(true_sources, prediction.sources),
                        **source_set_distances(graph, true_sources, prediction.sources),
                        **source_radius_hits(graph, true_sources, prediction.sources),
                    }
                )

    external_metrics = _aggregate_prediction_rows(rows)
    in_domain = run_metrics["prediction_metrics"]["joint_estimated_k"]["all"]
    comparison = {
        "in_domain": {
            "domain": run_metrics["graph_id"],
            **{key: in_domain[key] for key in _comparison_metrics()},
        },
        "external": {
            "domain": external_graph_id,
            **{
                key: external_metrics["external_estimated_k"]["all"][key]
                for key in _comparison_metrics()
            },
        },
    }
    summary = {
        "train_graph": run_metrics["graph_id"],
        "external_graph": external_graph_id,
        "external_example_count": len(examples),
        "feature_indices": feature_indices,
        "external_metrics": external_metrics,
        "comparison": comparison,
    }
    _write_csv(output_path / "transfer_predictions.csv", rows)
    _write_csv(output_path / "transfer_table.csv", list(comparison.values()))
    (output_path / "transfer_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot(comparison, output_path / "transfer_comparison.png")
    return summary


def _comparison_metrics() -> tuple[str, ...]:
    return "f1", "count_accuracy", "count_mae", "symmetric_set_distance"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(comparison: dict[str, dict], output_path: Path) -> None:
    labels = [comparison[key]["domain"] for key in ("in_domain", "external")]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric, title, limit in (
        (axes[0], "f1", "Estimated-k macro-F1", (0, 1)),
        (axes[1], "count_accuracy", "Count accuracy", (0, 1)),
        (axes[2], "symmetric_set_distance", "Graph distance", None),
    ):
        axis.bar(
            labels,
            [comparison[key][metric] for key in ("in_domain", "external")],
            color=["#3D7A80", "#E07A5F"],
        )
        axis.set_title(title)
        if limit:
            axis.set_ylim(*limit)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Cross-topology transfer without fine-tuning")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = evaluate_transfer(args.external_data, args.run, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
