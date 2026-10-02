"""Training entrypoint for one known-k two-observation GCN pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import argparse
import numpy as np
import torch
import yaml

from .dataset import load_graph_archive
from .features import SnapshotFeatureBuilder
from .known_k_temporal_data import load_known_k_split
from .models import NodeOnlyGCN
from .train_cli import calculate_pos_weight, set_seed
from .training import fit_node_model, save_training_result


@dataclass(frozen=True)
class KnownKPaths:
    data_dir: Path
    frozen_s1b_dir: Path
    output_dir: Path
    config_path: Path


def _config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "model": {"architecture": "node_only", "input_dim": 14, "hidden_dim": 64, "dropout": 0.2},
        "training": {"seed": 7026, "learning_rate": 0.001, "batch_size": 3,
                     "max_epochs": 100, "patience": 10},
    }
    if (not isinstance(config, dict)
            or any(config.get(section) != value for section, value in expected.items())
            or config.get("evaluation", {}).get("evaluate_test") is not False
            or config.get("evaluation", {}).get("control_beta") != 0.5
            or len(config.get("data", {}).get("feature_names", [])) != 10):
        raise ValueError("pilot config differs from the fixed protocol")
    return config


def run_stage(
    stage: Literal["freeze", "smoke", "pilot"],
    paths: KnownKPaths,
    device: torch.device,
    *,
    resume: bool = False,
) -> dict:
    """Run a small CPU smoke or the fixed full-data GPU training stage."""
    if stage not in ("smoke", "pilot"):
        raise ValueError("freeze stage is added with artifact guards")
    config = _config(paths.config_path)
    if stage == "pilot" and device.type != "cuda":
        raise ValueError("GPU required for the full pilot")
    if stage == "pilot" and not torch.cuda.is_available():
        raise ValueError("GPU unavailable")
    if resume:
        raise ValueError("resume requires frozen artifact guards")
    _, graph = load_graph_archive(paths.data_dir / "graph.npz")
    builder = SnapshotFeatureBuilder(graph, distance_cap=int(config["data"].get("distance_cap", 10)))
    names = tuple(config["data"]["feature_names"])
    with np.load(paths.data_dir / "train.npz", allow_pickle=False) as archive:
        n_train = len(archive["source_counts"])
    with np.load(paths.data_dir / "validation.npz", allow_pickle=False) as archive:
        n_validation = len(archive["source_counts"])
    if stage == "smoke":
        n_train, n_validation = min(6, n_train), min(6, n_validation)
    print(f"[{stage}] loading {n_train} train / {n_validation} validation cascades", flush=True)
    train = load_known_k_split(paths.data_dir, "train", graph, builder, names, range(n_train))
    validation = load_known_k_split(paths.data_dir, "validation", graph, builder, names,
                                    range(n_validation))
    set_seed(config["training"]["seed"])
    model = NodeOnlyGCN(input_dim=14, hidden_dim=64, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output = paths.output_dir / stage
    output.mkdir(parents=True, exist_ok=True)
    result = fit_node_model(
        model, train, validation, optimizer,
        max_epochs=1 if stage == "smoke" else config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        pos_weight=calculate_pos_weight(train).to(device),
        batch_size=config["training"]["batch_size"],
        checkpoint_path=output / "last_checkpoint.pt",
    )
    save_training_result(result, output)
    return {"stage": stage, "training_performed": True, "quality_gate": None,
            "n_train": n_train, "n_validation": n_validation,
            "best_epoch": result.best_epoch,
            "best_validation_oracle_f1": result.validation_history[result.best_epoch - 1].macro_f1,
            "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Known-k temporal GCN pilot")
    parser.add_argument("stage", choices=("freeze", "smoke", "pilot"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-s1b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    paths = KnownKPaths(args.data_dir, args.frozen_s1b_dir, args.output_dir, args.config)
    print(run_stage(args.stage, paths, torch.device(args.device), resume=args.resume), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
