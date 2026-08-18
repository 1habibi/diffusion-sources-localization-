"""Evaluate an IC-trained checkpoint on paired multi-source SI cascades."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import graph_to_edge_index, load_graph_archive
from .diffusion import simulate_si
from .features import node_features
from .inference import predict_joint
from .metrics import set_metrics, source_set_distances
from .models import JointSourceCountGCN
from .observations import observe_cascade


def evaluate_process_shift(
    data_dir: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict:
    """Compare saved IC test predictions with paired SI simulations."""
    data_path = Path(data_dir)
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _, graph = load_graph_archive(data_path / "graph.npz")
    archive = np.load(data_path / "test.npz", allow_pickle=False)
    run_metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_path / "config.yaml").read_text(encoding="utf-8"))
    feature_indices = [int(value) for value in run_metrics["feature_indices"]]
    model = JointSourceCountGCN(
        input_dim=len(feature_indices),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(torch.load(run_path / "best_model.pt", map_location="cpu"))
    model.eval()

    rows = load_ic_rows(run_path / "test_predictions.csv")
    with torch.no_grad():
        for index in tqdm(
            range(len(archive["source_counts"])),
            desc="SI evaluation",
            unit="cascade",
        ):
            sources = frozenset(np.flatnonzero(archive["source_labels"][index]).tolist())
            cascade = simulate_si(
                graph,
                sources,
                float(archive["probabilities"][index]),
                int(config.get("process_shift", {}).get("max_steps", 3)),
                np.random.default_rng(int(archive["simulation_seeds"][index]) + seed),
            )
            observation = observe_cascade(
                graph,
                cascade,
                float(archive["observation_fractions"][index]),
                0,
                np.random.default_rng(int(archive["observation_seeds"][index]) + seed),
            )
            features = node_features(graph, observation)[:, feature_indices]
            data = Data(
                x=torch.from_numpy(features).float(),
                edge_index=graph_to_edge_index(graph),
                candidate_mask=torch.tensor(
                    [node in observation.candidate_nodes for node in graph], dtype=torch.bool
                ),
                observed_mask=torch.tensor(
                    [node in observation.observed_infected for node in graph], dtype=torch.bool
                ),
            )
            source_logits, count_logits = model(data)
            prediction = predict_joint(source_logits, count_logits, data.candidate_mask)
            rows.append(
                {
                    "process": "SI",
                    "example": index,
                    "k": len(sources),
                    **set_metrics(sources, prediction.sources),
                    **source_set_distances(graph, sources, prediction.sources),
                }
            )

    aggregates = aggregate_process_rows(rows)
    summary = {"seed": seed, "processes": aggregates}
    _write_csv(output_path / "process_shift_predictions.csv", rows)
    _write_csv(output_path / "process_shift_table.csv", list(aggregates.values()))
    (output_path / "process_shift_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot(aggregates, output_path / "process_shift_comparison.png")
    return summary


def load_ic_rows(path: str | Path) -> list[dict]:
    """Load only estimated-k Joint rows from the regular test evaluation."""
    with Path(path).open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    float_fields = (
        "precision", "recall", "f1", "exact_set_accuracy", "count_accuracy",
        "count_mae", "source_to_set_distance", "set_to_source_distance",
        "symmetric_set_distance",
    )
    return [
        {
            "process": "IC",
            "example": int(row["example"]),
            "k": int(row["k"]),
            **{field: float(row[field]) for field in float_fields},
        }
        for row in rows
        if row["method"] == "joint_estimated_k"
    ]


def aggregate_process_rows(rows: list[dict]) -> dict[str, dict]:
    """Aggregate localization and count metrics separately for IC and SI."""
    result = {}
    for process in ("IC", "SI"):
        selected = [row for row in rows if row["process"] == process]
        if not selected:
            raise ValueError(f"No rows for process {process}.")
        result[process] = {
            "process": process,
            "f1": float(np.mean([row["f1"] for row in selected])),
            "count_accuracy": float(np.mean([row["count_accuracy"] for row in selected])),
            "count_mae": float(np.mean([row["count_mae"] for row in selected])),
            "symmetric_set_distance": float(
                np.mean([row["symmetric_set_distance"] for row in selected])
            ),
        }
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(aggregates: dict[str, dict], output_path: Path) -> None:
    processes = ["IC", "SI"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric, title, limit in (
        (axes[0], "f1", "Macro-F1", (0, 1)),
        (axes[1], "count_accuracy", "Count accuracy", (0, 1)),
        (axes[2], "symmetric_set_distance", "Graph distance", None),
    ):
        axis.bar(
            processes,
            [aggregates[process][metric] for process in processes],
            color=["#3D7A80", "#E07A5F"],
        )
        axis.set_title(title)
        if limit:
            axis.set_ylim(*limit)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Process shift: trained on IC, evaluated on IC and SI")
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
    summary = evaluate_process_shift(args.data, args.run, args.output, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
