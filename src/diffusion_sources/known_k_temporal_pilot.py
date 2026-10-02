"""Training entrypoint for one known-k two-observation GCN pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import argparse
import json
import numpy as np
import torch
import yaml

from .dataset import load_graph_archive
from .features import SnapshotFeatureBuilder
from .known_k_temporal_data import load_known_k_split
from .known_k_temporal_eval import evaluate_known_k_pairs, known_k_pilot_gate
from .models import JointSourceCountGCN, NodeOnlyGCN
from .train_cli import calculate_pos_weight, set_seed
from .temporal_pilot_artifacts import read_stage, sha256_file, write_stage
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
        "model": {"architecture": "node_only", "input_dim": 9, "hidden_dim": 64, "dropout": 0.2},
        "training": {"seed": 7026, "learning_rate": 0.001, "batch_size": 3,
                     "max_epochs": 100, "patience": 10},
    }
    if (not isinstance(config, dict)
            or any(config.get(section) != value for section, value in expected.items())
            or config.get("evaluation", {}).get("evaluate_test") is not False
            or config.get("evaluation", {}).get("control_beta") != 0.5
            or len(config.get("data", {}).get("feature_names", [])) != 5):
        raise ValueError("pilot config differs from the fixed protocol")
    return config


def _load_frozen_s1b(frozen_dir: Path, config: dict, device: torch.device) -> JointSourceCountGCN:
    mc = config["model"]
    model = JointSourceCountGCN(
        input_dim=5, hidden_dim=int(mc["hidden_dim"]), dropout=float(mc["dropout"]),
        source_head_mode=mc.get("source_head_mode", "local"),
        global_feature_dim=int(mc.get("global_feature_dim", 0)),
        backbone_mode=mc.get("backbone_mode", "plain_2"),
        source_head_strategy=mc.get("source_head_strategy", "shared"),
        shortlist_mode=mc.get("shortlist_mode", "disabled"),
    )
    try:
        model.load_state_dict(torch.load(frozen_dir / "best_model.pt",
                                         map_location="cpu", weights_only=True))
    except Exception as exc:
        raise ValueError("frozen S1b checkpoint is incompatible with its config") from exc
    return model.to(device).eval()


def _input_identity(paths: KnownKPaths, config: dict) -> dict:
    data = Path(paths.data_dir).resolve()
    frozen = Path(paths.frozen_s1b_dir).resolve()
    output = Path(paths.output_dir).resolve()
    if (output == data or output.is_relative_to(data) or data.is_relative_to(output)
            or output == frozen or output.is_relative_to(frozen) or frozen.is_relative_to(output)):
        raise ValueError("output root overlaps protected inputs")
    s1b_config = yaml.safe_load((frozen / "config.yaml").read_text(encoding="utf-8"))
    if (s1b_config["data"]["feature_names"] != config["data"]["feature_names"]
            or s1b_config["model"]["input_dim"] != 5
            or s1b_config["model"].get("source_head_strategy", "shared") != "shared"):
        raise ValueError("frozen S1b config does not match the known-k control")
    _load_frozen_s1b(frozen, s1b_config, torch.device("cpu"))
    inputs = {
        "graph.npz": data / "graph.npz",
        "train.npz": data / "train.npz",
        "validation.npz": data / "validation.npz",
        "config.yaml": data / "config.yaml",
        "pilot_config.yaml": Path(paths.config_path).resolve(),
        "s1b_config.yaml": frozen / "config.yaml",
        "s1b_best_model.pt": frozen / "best_model.pt",
    }
    return {"output_root": str(output),
            "input_hashes": {name: sha256_file(path) for name, path in inputs.items()}}


def _complete_binary_stage(output: Path, stage: str, progress: Path,
                           identity: dict, payload: dict) -> dict:
    (progress / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    hashes = {path.name: sha256_file(path) for path in progress.iterdir() if path.is_file()}
    manifest = {"schema_version": 1, "identity": identity, "file_hashes": hashes}
    (progress / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (progress / "complete").write_text("complete\n", encoding="utf-8")
    progress.rename(output / stage)
    return read_stage(output, stage, identity)[1]


def _validate_resume_progress(progress: Path, identity: dict) -> None:
    saved = json.loads((progress / "input_identity.json").read_text(encoding="utf-8"))
    if saved != identity:
        raise ValueError("partial pilot identity mismatch; choose a new output root")
    checkpoint = progress / "last_checkpoint.pt"
    if not checkpoint.is_file() or checkpoint.with_suffix(".pt.tmp").exists():
        raise ValueError("partial pilot has no complete last-checkpoint")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("partial pilot checkpoint is corrupt") from exc
    if not isinstance(state, dict) or not {"epoch", "model_state_dict", "optimizer_state_dict"} <= set(state):
        raise ValueError("partial pilot checkpoint is corrupt")


def _evaluate_pilot(model: NodeOnlyGCN, cases: list, graph,
                    frozen_s1b_dir: Path, device: torch.device) -> dict:
    """Run both frozen and candidate heads on the identical validation cases."""
    frozen_config = yaml.safe_load((frozen_s1b_dir / "config.yaml").read_text(encoding="utf-8"))
    frozen = _load_frozen_s1b(frozen_s1b_dir, frozen_config, device)
    model.eval()
    candidate_logits = {}
    s1b_logits = {}
    with torch.inference_mode():
        for case in cases:
            current = case.clone().to(device)
            frozen_input = current.clone()
            frozen_input.x = current.x[:, :5]
            candidate_logits[case.example_index] = model(current).detach().cpu()
            s1b_logits[case.example_index] = frozen(frozen_input)[0].detach().cpu()
    return evaluate_known_k_pairs(cases, graph, candidate_logits, s1b_logits, beta=0.5)


def run_stage(
    stage: Literal["freeze", "smoke", "pilot"],
    paths: KnownKPaths,
    device: torch.device,
    *,
    resume: bool = False,
) -> dict:
    """Run append-only freeze, small CPU smoke or fixed full-data pilot."""
    if stage not in ("freeze", "smoke", "pilot"):
        raise ValueError("unknown known-k stage")
    config = _config(paths.config_path)
    identity = _input_identity(paths, config)
    output_root = Path(paths.output_dir)
    if stage == "freeze":
        if (output_root / "freeze").exists():
            return read_stage(output_root, "freeze", identity)[1]
        payload = {"status": "frozen", "input_hashes": identity["input_hashes"]}
        write_stage(output_root, "freeze", {"identity": identity}, payload)
        return read_stage(output_root, "freeze", identity)[1]
    read_stage(output_root, "freeze", identity)
    if stage == "pilot":
        read_stage(output_root, "smoke", identity)
    if stage == "pilot" and device.type != "cuda":
        raise ValueError("GPU required for the full pilot")
    if stage == "pilot" and not torch.cuda.is_available():
        raise ValueError("GPU unavailable")
    if resume and stage != "pilot":
        raise ValueError("resume is only allowed for an incomplete pilot")
    final = output_root / stage
    if final.exists():
        if resume:
            raise ValueError("completed stage cannot be resumed")
        return read_stage(output_root, stage, identity)[1]
    progress = output_root / f".{stage}-progress"
    if progress.exists():
        if not resume:
            raise ValueError(f"partial {stage} stage exists; explicit resume or new output root required")
        _validate_resume_progress(progress, identity)
    else:
        if resume:
            raise ValueError("no partial pilot checkpoint to resume")
        progress.mkdir(parents=True)
        (progress / "input_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    model = NodeOnlyGCN(input_dim=9, hidden_dim=64, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    result = fit_node_model(
        model, train, validation, optimizer,
        max_epochs=1 if stage == "smoke" else config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        pos_weight=calculate_pos_weight(train).to(device),
        batch_size=config["training"]["batch_size"],
        checkpoint_path=progress / "last_checkpoint.pt",
        resume_from=progress / "last_checkpoint.pt" if resume else None,
    )
    save_training_result(result, progress)
    payload = {"stage": stage, "training_performed": True, "quality_gate": None,
            "n_train": n_train, "n_validation": n_validation,
            "best_epoch": result.best_epoch,
            "best_validation_oracle_f1": result.validation_history[result.best_epoch - 1].macro_f1,
            "output": str(final)}
    if stage == "pilot":
        paired = _evaluate_pilot(model, validation, graph, Path(paths.frozen_s1b_dir), device)
        payload["paired_report"] = paired
        payload["quality_gate"] = known_k_pilot_gate(paired)
        payload["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return _complete_binary_stage(output_root, stage, progress, identity, payload)


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
