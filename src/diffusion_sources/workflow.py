"""Run the reproducible project workflow from generation to report artifacts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .ablations import build_ablation_report
from .evaluate_cli import evaluate_baselines
from .generation import generate_dataset, load_config
from .hidden_source_report import build_hidden_source_report
from .node_train_cli import run_node_training
from .process_shift import evaluate_process_shift
from .reporting import build_report
from .robustness import evaluate_robustness
from .train_cli import load_train_config, run_training


def run_workflow(config_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Execute enabled workflow stages and write one artifact manifest."""
    workflow_path = Path(config_path).resolve()
    project_root = workflow_path.parent.parent
    config = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "configs" not in config:
        raise ValueError("Workflow config must contain a configs mapping.")

    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = _workflow_paths(output_path)
    overrides = config.get("training_overrides", {})
    model_overrides = config.get("model_overrides", {})
    artifacts: dict[str, Any] = {}

    generation_config = load_config(project_root / config["configs"]["generation"])
    artifacts["generation"] = _summary(
        generate_dataset(generation_config, paths["data"])
    )

    joint_config = _training_config(
        project_root / config["configs"]["joint"],
        paths["data"],
        overrides,
        model_overrides,
    )
    node_config = _training_config(
        project_root / config["configs"]["node"],
        paths["data"],
        overrides,
        model_overrides,
    )
    no_consistency_config = _training_config(
        project_root / config["configs"]["without_consistency"],
        paths["data"],
        overrides,
        model_overrides,
    )
    artifacts["joint"] = run_training(joint_config, paths["joint"])
    artifacts["node"] = run_node_training(node_config, paths["node"])
    artifacts["without_consistency"] = run_training(
        no_consistency_config, paths["without_consistency"]
    )

    artifacts["baselines"] = evaluate_baselines(
        paths["data"],
        paths["baselines"],
        split="test",
        uniform_repeats=int(config.get("uniform_repeats", 100)),
        seed=int(config.get("seed", 2026)),
    )
    artifacts["report"] = [
        str(path)
        for path in build_report(
            paths["joint"], paths["report"], paths["baselines"], paths["node"]
        )
    ]
    artifacts["ablations"] = [
        str(path)
        for path in build_ablation_report(
            paths["node"],
            paths["without_consistency"],
            paths["joint"],
            paths["ablations"],
        )
    ]

    if config.get("run_robustness", True):
        artifacts["robustness"] = evaluate_robustness(
            paths["data"],
            paths["joint"],
            paths["robustness"],
            fractions=tuple(float(value) for value in config.get("fractions", [1, 0.75, 0.5])),
            noise_levels=tuple(float(value) for value in config.get("noise_levels", [0, 0.05, 0.1])),
            seed=int(config.get("seed", 2026)),
        )
    if config.get("run_process_shift", True):
        artifacts["process_shift"] = evaluate_process_shift(
            paths["data"],
            paths["joint"],
            paths["process_shift"],
            seed=int(config.get("seed", 2026)),
        )

    hidden_generation_name = config["configs"].get("hidden_generation")
    hidden_training_name = config["configs"].get("hidden_training")
    if hidden_generation_name and hidden_training_name:
        hidden_generation = load_config(project_root / hidden_generation_name)
        artifacts["hidden_generation"] = _summary(
            generate_dataset(hidden_generation, paths["hidden_data"])
        )
        hidden_training = _training_config(
            project_root / hidden_training_name,
            paths["hidden_data"],
            overrides,
            model_overrides,
        )
        artifacts["hidden_run"] = run_training(hidden_training, paths["hidden_run"])
        artifacts["hidden_report"] = [
            str(path)
            for path in build_hidden_source_report(
                paths["joint"], paths["hidden_run"], paths["hidden_report"]
            )
        ]

    manifest = {
        "workflow_config": str(workflow_path),
        "output_root": str(output_path.resolve()),
        "artifacts": artifacts,
    }
    (output_path / "workflow_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_path / "workflow_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return manifest


def _training_config(
    path: Path,
    data_dir: Path,
    training_overrides: dict,
    model_overrides: dict,
) -> dict:
    config = copy.deepcopy(load_train_config(path))
    config["data"]["directory"] = str(data_dir)
    config["training"].update(training_overrides)
    config["model"].update(model_overrides)
    return config


def _workflow_paths(root: Path) -> dict[str, Path]:
    names = (
        "data", "joint", "node", "without_consistency", "baselines", "report",
        "ablations", "robustness", "process_shift", "hidden_data", "hidden_run",
        "hidden_report",
    )
    return {name: root / name for name in names}


def _summary(value) -> dict:
    return {
        "graph_id": value.graph_id,
        "requested": value.requested,
        "accepted": value.accepted,
        "attempts": value.attempts,
        "output_dir": value.output_dir,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = run_workflow(args.config, args.output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
