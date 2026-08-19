"""Re-evaluate frozen v1 checkpoints with hop metrics without rewriting v1 runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .dataset import load_graph_archive, load_pyg_split
from .inference import predict_joint, predict_oracle_k, predict_thresholded
from .metrics import source_radius_hits
from .models import JointSourceCountGCN, NodeOnlyGCN


def evaluate_v1_hops(
    data_dir: str | Path,
    joint_root: str | Path,
    node_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Iterable[int] = (7026, 7027, 7028),
    batch_size: int = 3,
) -> dict[str, Any]:
    """Calculate estimated/oracle-k Hit@1/2-hop for existing v1 checkpoints."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("At least one seed is required.")

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _, graph = load_graph_archive(data_path / "graph.npz")
    rows: list[dict[str, Any]] = []

    for seed in seed_values:
        joint_dir = Path(joint_root) / f"seed_{seed}"
        joint_model, joint_features, _ = _load_joint(joint_dir)
        joint_examples = load_pyg_split(
            data_path / "test.npz", graph, feature_indices=joint_features
        )
        rows.extend(
            _evaluate_joint(
                joint_model, joint_examples, graph, seed=seed, batch_size=batch_size
            )
        )

        node_dir = Path(node_root) / f"seed_{seed}"
        node_model, node_features, node_metrics = _load_node(node_dir)
        node_examples = load_pyg_split(
            data_path / "test.npz", graph, feature_indices=node_features
        )
        rows.extend(
            _evaluate_node(
                node_model,
                node_examples,
                graph,
                threshold=float(node_metrics["threshold"]),
                seed=seed,
                batch_size=batch_size,
            )
        )

    aggregates = _aggregate(rows)
    summary = {
        "model_version": "v1_baseline",
        "evaluation": "checkpoint_re_evaluation_without_training",
        "seeds": list(seed_values),
        "test_examples_per_seed": len(joint_examples),
        "metrics": aggregates,
    }
    _write_csv(output_path / "v1_hop_predictions.csv", rows)
    _write_aggregate_csv(output_path / "v1_hop_metrics.csv", aggregates)
    (output_path / "v1_hop_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _load_joint(run_dir: Path):
    config, metrics = _load_run_metadata(run_dir)
    feature_indices = [int(value) for value in metrics["feature_indices"]]
    model = JointSourceCountGCN(
        input_dim=len(feature_indices),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, feature_indices, metrics


def _load_node(run_dir: Path):
    config, metrics = _load_run_metadata(run_dir)
    feature_indices = [int(value) for value in metrics["feature_indices"]]
    model = NodeOnlyGCN(
        input_dim=len(feature_indices),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, feature_indices, metrics


def _load_run_metadata(run_dir: Path) -> tuple[dict, dict]:
    required = (run_dir / "config.yaml", run_dir / "metrics.json", run_dir / "best_model.pt")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v1 run artifacts: {', '.join(missing)}")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return config, metrics


@torch.no_grad()
def _evaluate_joint(model, examples, graph, *, seed: int, batch_size: int) -> list[dict]:
    rows: list[dict] = []
    example_index = 0
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False)
    for data in tqdm(loader, desc=f"v1 Joint seed {seed}", unit="batch", leave=False):
        source_logits, count_logits = model(data)
        ptr = data.ptr
        for graph_index in range(count_logits.size(0)):
            start, end = int(ptr[graph_index]), int(ptr[graph_index + 1])
            true_sources = frozenset(
                torch.nonzero(data.source_labels[start:end], as_tuple=False).flatten().tolist()
            )
            true_count = int(data.source_count[graph_index])
            predictions = {
                "joint_estimated_k": predict_joint(
                    source_logits[start:end],
                    count_logits[graph_index : graph_index + 1],
                    data.candidate_mask[start:end],
                ),
                "joint_oracle_k": predict_oracle_k(
                    source_logits[start:end], data.candidate_mask[start:end], true_count
                ),
            }
            rows.extend(
                _hop_rows(
                    graph,
                    predictions,
                    true_sources,
                    true_count,
                    example=example_index,
                    seed=seed,
                )
            )
            example_index += 1
    return rows


@torch.no_grad()
def _evaluate_node(
    model, examples, graph, *, threshold: float, seed: int, batch_size: int
) -> list[dict]:
    rows: list[dict] = []
    example_index = 0
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False)
    for data in tqdm(loader, desc=f"v1 Node seed {seed}", unit="batch", leave=False):
        logits = model(data)
        ptr = data.ptr
        for graph_index in range(len(data.source_count.reshape(-1))):
            start, end = int(ptr[graph_index]), int(ptr[graph_index + 1])
            true_sources = frozenset(
                torch.nonzero(data.source_labels[start:end], as_tuple=False).flatten().tolist()
            )
            true_count = int(data.source_count[graph_index])
            predictions = {
                "node_thresholded": predict_thresholded(
                    logits[start:end], data.candidate_mask[start:end], threshold
                ),
                "node_oracle_k": predict_oracle_k(
                    logits[start:end], data.candidate_mask[start:end], true_count
                ),
            }
            rows.extend(
                _hop_rows(
                    graph,
                    predictions,
                    true_sources,
                    true_count,
                    example=example_index,
                    seed=seed,
                )
            )
            example_index += 1
    return rows


def _hop_rows(graph, predictions, true_sources, true_count, *, example, seed):
    return [
        {
            "seed": seed,
            "example": example,
            "k": true_count,
            "predicted_k": prediction.source_count,
            "method": method,
            **source_radius_hits(graph, true_sources, prediction.sources),
        }
        for method, prediction in predictions.items()
    ]


def _aggregate(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], "all")].append(row)
        grouped[(row["method"], str(row["k"]))].append(row)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, group), selected in grouped.items():
        result[method][group] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in ("hit_at_1_hop", "hit_at_2_hop")
        }
    return dict(result)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_csv(path: Path, aggregates: dict) -> None:
    rows = [
        {"method": method, "k": group, **metrics}
        for method, groups in aggregates.items()
        for group, metrics in groups.items()
    ]
    _write_csv(path, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--joint-root", required=True, type=Path)
    parser.add_argument("--node-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args(argv)
    summary = evaluate_v1_hops(
        args.data,
        args.joint_root,
        args.node_root,
        args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
