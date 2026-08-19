"""Evaluate oracle-k classical baselines on a generated dataset split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from .baselines import degree_candidates, multi_jordan, uniform_candidates
from .dataset import load_graph_archive
from .metrics import set_metrics, source_radius_hits, source_set_distances


def evaluate_baselines(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    uniform_repeats: int = 100,
    seed: int = 2026,
) -> dict:
    """Evaluate all oracle-k baselines and save per-example and aggregate data."""
    if uniform_repeats < 1:
        raise ValueError("uniform_repeats must be positive.")
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graph_id, graph = load_graph_archive(data_path / "graph.npz")
    archive = np.load(data_path / f"{split}.npz", allow_pickle=False)
    rows: list[dict[str, float | int | str]] = []

    for index in range(len(archive["source_counts"])):
        source_count = int(archive["source_counts"][index])
        true_sources = frozenset(
            np.flatnonzero(archive["source_labels"][index]).tolist()
        )
        candidates = frozenset(
            np.flatnonzero(archive["candidate_masks"][index]).tolist()
        )
        observed = frozenset(
            np.flatnonzero(archive["features"][index, :, 0]).tolist()
        )

        deterministic = {
            "degree": degree_candidates(graph, candidates, source_count),
            "multi_jordan": multi_jordan(
                graph, observed, candidates, source_count
            ),
        }
        for method, prediction in deterministic.items():
            rows.append(
                _evaluation_row(
                    index, source_count, method, true_sources, prediction, graph
                )
            )

        uniform_rows = []
        for repeat in range(uniform_repeats):
            prediction = uniform_candidates(
                candidates,
                source_count,
                np.random.default_rng(seed + index * uniform_repeats + repeat),
            )
            uniform_rows.append(
                _evaluation_row(
                    index, source_count, "uniform", true_sources, prediction, graph
                )
            )
        rows.append(_mean_row(uniform_rows))

    aggregates = _aggregate_rows(rows)
    summary = {
        "graph_id": graph_id,
        "split": split,
        "example_count": len(archive["source_counts"]),
        "uniform_repeats": uniform_repeats,
        "seed": seed,
        "methods": aggregates,
    }
    (output_path / "baseline_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_rows(output_path / "baseline_predictions.csv", rows)
    _write_aggregate_table(output_path / "baseline_table.csv", aggregates)
    return summary


def _evaluation_row(index, source_count, method, true_sources, prediction, graph):
    return {
        "example": index,
        "k": source_count,
        "method": method,
        **set_metrics(true_sources, prediction),
        **source_set_distances(graph, true_sources, prediction),
        **source_radius_hits(graph, true_sources, prediction),
    }


def _mean_row(rows: list[dict]) -> dict:
    result = {key: rows[0][key] for key in ("example", "k", "method")}
    for key in rows[0].keys() - result.keys():
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def _aggregate_rows(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[tuple[str, int | str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], "all")].append(row)
        grouped[(row["method"], row["k"])].append(row)
    result: dict[str, dict] = defaultdict(dict)
    metric_names = [
        "precision",
        "recall",
        "f1",
        "exact_set_accuracy",
        "symmetric_set_distance",
        "hit_at_1_hop",
        "hit_at_2_hop",
    ]
    for (method, group), group_rows in grouped.items():
        result[method][str(group)] = {
            metric: float(np.mean([row[metric] for row in group_rows]))
            for metric in metric_names
        }
    return dict(result)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_table(path: Path, aggregates: dict) -> None:
    rows = []
    for method, groups in aggregates.items():
        for group, metrics in groups.items():
            rows.append({"method": method, "k": group, **metrics})
    _write_rows(path, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--uniform-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    result = evaluate_baselines(
        args.data,
        args.output,
        split=args.split,
        uniform_repeats=args.uniform_repeats,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
