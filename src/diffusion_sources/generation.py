"""Configuration-driven generation and storage of pilot diffusion datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from .dataset import CascadeExample, build_example
from .diffusion import sample_sources, simulate_ic
from .graphs import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    karate_graph,
    load_edge_list,
)
from .observations import observe_cascade


@dataclass(frozen=True)
class GenerationSummary:
    graph_id: str
    requested: dict[str, int]
    accepted: dict[str, int]
    attempts: dict[str, int]
    output_dir: str


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML generation configuration and validate required sections."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for section in ("graph", "simulation", "observation", "dataset"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}.")
    return config


def graph_from_config(config: dict[str, Any]) -> tuple[str, nx.Graph]:
    """Construct one supported pilot graph from its configuration."""
    kind = config.get("kind")
    graph_id = str(config.get("id", kind))
    if kind == "karate":
        return graph_id, karate_graph()
    if kind == "erdos_renyi":
        return graph_id, erdos_renyi_graph(
            int(config["nodes"]), float(config["probability"]), config.get("seed")
        )
    if kind == "barabasi_albert":
        return graph_id, barabasi_albert_graph(
            int(config["nodes"]), int(config["attachments"]), config.get("seed")
        )
    if kind == "edge_list":
        graph, _ = load_edge_list(config["path"])
        return graph_id, graph
    raise ValueError(f"Unsupported graph kind: {kind!r}.")


def generate_dataset(config: dict[str, Any], output_dir: str | Path) -> GenerationSummary:
    """Generate balanced split files and save graph topology once."""
    graph_id, graph = graph_from_config(config["graph"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _save_graph(graph_id, graph, output_path)

    dataset_config = config["dataset"]
    base_seed = int(dataset_config.get("seed", 0))
    split_sizes = {
        split: int(size) for split, size in dataset_config["splits"].items()
    }
    max_attempt_factor = int(dataset_config.get("max_attempt_factor", 50))
    accepted: dict[str, int] = {}
    attempts: dict[str, int] = {}

    for split_index, (split, requested_size) in enumerate(split_sizes.items()):
        examples, split_attempts = _generate_split(
            graph_id,
            graph,
            config,
            split,
            requested_size,
            base_seed + split_index * 1_000_000,
            max_attempt_factor,
        )
        _save_examples(split, examples, output_path)
        accepted[split] = len(examples)
        attempts[split] = split_attempts

    summary = GenerationSummary(
        graph_id=graph_id,
        requested=split_sizes,
        accepted=accepted,
        attempts=attempts,
        output_dir=str(output_path),
    )
    (output_path / "generation_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return summary


def _generate_split(
    graph_id: str,
    graph: nx.Graph,
    config: dict[str, Any],
    split: str,
    requested_size: int,
    split_seed: int,
    max_attempt_factor: int,
) -> tuple[list[CascadeExample], int]:
    simulation = config["simulation"]
    observation = config["observation"]
    dataset = config["dataset"]
    probabilities = [float(value) for value in simulation["probabilities"]]
    fractions = [float(value) for value in observation["fractions"]]
    hide_source_count = int(observation.get("hide_source_count", 0))
    source_counts = [int(value) for value in simulation.get("source_counts", [1, 2, 3])]
    max_steps = int(simulation["max_steps"])
    min_candidates = int(dataset.get("min_candidates", 5))
    max_infected_fraction = float(dataset.get("max_infected_fraction", 0.5))
    distance_ranges = simulation.get(
        "distance_ranges", [{"min": 0, "max": None}]
    )

    examples: list[CascadeExample] = []
    attempts = 0
    max_attempts = max(requested_size * max_attempt_factor, 1)
    while len(examples) < requested_size and attempts < max_attempts:
        example_index = len(examples)
        simulation_seed = split_seed + attempts * 2
        observation_seed = simulation_seed + 1
        simulation_rng = np.random.default_rng(simulation_seed)
        observation_rng = np.random.default_rng(observation_seed)
        source_count = source_counts[example_index % len(source_counts)]
        distance_range = distance_ranges[example_index % len(distance_ranges)]
        attempts += 1

        try:
            sources = sample_sources(
                graph,
                source_count,
                simulation_rng,
                min_distance=int(distance_range.get("min", 0)),
                max_distance=(
                    int(distance_range["max"])
                    if distance_range.get("max") is not None
                    else None
                ),
            )
        except RuntimeError:
            continue
        cascade = simulate_ic(
            graph,
            sources,
            probabilities[attempts % len(probabilities)],
            max_steps,
            simulation_rng,
        )
        if len(cascade.infected) > max_infected_fraction * graph.number_of_nodes():
            continue

        observed = observe_cascade(
            graph,
            cascade,
            fractions[attempts % len(fractions)],
            int(observation.get("false_positive_count", 0)),
            observation_rng,
            hide_source_count=min(hide_source_count, source_count),
        )
        if not observed.observed_infected:
            continue
        if len(observed.candidate_nodes) < max(min_candidates, 2 * source_count):
            continue
        examples.append(
            build_example(
                graph_id,
                graph,
                cascade,
                observed,
                simulation_seed=simulation_seed,
                observation_seed=observation_seed,
            )
        )

    if len(examples) != requested_size:
        raise RuntimeError(
            f"Generated {len(examples)}/{requested_size} examples for {split} "
            f"after {attempts} attempts. Relax pilot filters or diffusion parameters."
        )
    return examples, attempts


def _save_graph(graph_id: str, graph: nx.Graph, output_dir: Path) -> None:
    edges = np.asarray(sorted(graph.edges()), dtype=np.int64).reshape(-1, 2)
    np.savez_compressed(
        output_dir / "graph.npz",
        graph_id=np.asarray(graph_id),
        node_count=np.asarray(graph.number_of_nodes(), dtype=np.int64),
        edges=edges,
    )


def _save_examples(split: str, examples: list[CascadeExample], output_dir: Path) -> None:
    np.savez_compressed(
        output_dir / f"{split}.npz",
        features=np.stack([example.features for example in examples]),
        candidate_masks=np.stack([example.candidate_mask for example in examples]),
        source_labels=np.stack([example.source_labels for example in examples]),
        infected_masks=np.stack([example.infected_mask for example in examples]),
        source_counts=np.asarray([example.source_count for example in examples]),
        hidden_source_masks=np.stack(
            [
                np.asarray(
                    [node in example.observation["hidden_sources"] for node in range(len(example.source_labels))],
                    dtype=bool,
                )
                for example in examples
            ]
        ),
        simulation_seeds=np.asarray(
            [example.simulation["simulation_seed"] for example in examples]
        ),
        observation_seeds=np.asarray(
            [example.observation["observation_seed"] for example in examples]
        ),
        probabilities=np.asarray(
            [example.simulation["probability"] for example in examples]
        ),
        observation_fractions=np.asarray(
            [example.observation["observation_fraction"] for example in examples]
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = generate_dataset(load_config(args.config), args.output)
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
