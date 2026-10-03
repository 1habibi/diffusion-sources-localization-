"""Fixed, append-only repeats of the frozen known-k Temporal-GCN pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import json
import time
import numpy as np
import torch
import yaml

from diffusion_sources.dataset import load_graph_archive
from diffusion_sources.features import SnapshotFeatureBuilder
from diffusion_sources.known_k_temporal_data import load_known_k_split
from diffusion_sources.known_k_temporal_pilot import (
    KnownKPaths, _complete_binary_stage, _config, _evaluate_pilot, _input_identity,
)
from diffusion_sources.models import NodeOnlyGCN
from diffusion_sources.temporal_pilot_artifacts import read_stage, sha256_file, write_stage
from diffusion_sources.train_cli import calculate_pos_weight, set_seed
from diffusion_sources.training import fit_node_model, save_training_result


@dataclass(frozen=True)
class RepeatPaths:
    pilot: KnownKPaths
    output_dir: Path
    protocol_path: Path
    notebook_path: Path


def _overlaps(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _protocol(path: Path) -> dict:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"seeds": [7027, 7028], "stop_on_nonpositive_delta": True,
                "pilot_config": "configs/known_k_temporal_gcn_pilot.yaml"}
    if protocol != expected:
        raise ValueError("repeat protocol differs from fixed seeds, stop rule or pilot config")
    return protocol


def preflight(paths: RepeatPaths) -> dict:
    """Read and authenticate the old pilot without creating repeat output."""
    output = Path(paths.output_dir).resolve()
    for root in (paths.pilot.data_dir, paths.pilot.frozen_s1b_dir, paths.pilot.output_dir):
        if _overlaps(output, Path(root)):
            raise ValueError("repeat output overlaps protected pilot/data/S1b root")
    _protocol(paths.protocol_path)
    if not str(Path(paths.pilot.config_path).as_posix()).endswith(
            "configs/known_k_temporal_gcn_pilot.yaml"):
        raise ValueError("pilot config path differs from fixed protocol")
    old_identity = _input_identity(paths.pilot, _config(paths.pilot.config_path))
    old_root = Path(paths.pilot.output_dir)
    for stage in ("freeze", "smoke"):
        read_stage(old_root, stage, old_identity)
    pilot_manifest, pilot_payload = read_stage(old_root, "pilot", old_identity)
    required = {"input_identity.json", "last_checkpoint.pt", "last_checkpoint.pt.sha256",
                "history.json", "history.csv", "best_model.pt", "payload.json"}
    if not required <= set(pilot_manifest["file_hashes"]):
        raise ValueError("pilot manifest omits required checkpoint or history hashes")
    if pilot_payload.get("quality_gate", {}).get("passed") is not True:
        raise ValueError("pilot quality gate did not pass")
    source_dir = Path(__file__).resolve().parents[1] / "src" / "diffusion_sources"
    identity = {
        "old_pilot_identity": old_identity,
        "old_stage_manifest_hashes": {
            stage: sha256_file(old_root / stage / "manifest.json")
            for stage in ("freeze", "smoke", "pilot")
        },
        "old_pilot_checkpoint_hash": sha256_file(old_root / "pilot" / "best_model.pt"),
        "old_pilot_payload_hash": sha256_file(old_root / "pilot" / "payload.json"),
        "package_hashes": {path.name: sha256_file(path) for path in sorted(source_dir.glob("*.py"))},
        "runner_hash": sha256_file(Path(__file__)),
        "protocol_hash": sha256_file(paths.protocol_path),
        "notebook_hash": sha256_file(paths.notebook_path),
        "output_root": str(output),
    }
    if identity["package_hashes"] != old_identity["code_hashes"]:
        raise ValueError("package code hashes differ from pilot")
    return {"pilot": pilot_payload, "identity": identity}


def _assert_same_control(reference: dict, candidate: dict) -> None:
    """Require the exact same ordered examples and frozen control predictions."""
    old_rows, new_rows = reference.get("rows"), candidate.get("rows")
    if not isinstance(old_rows, list) or not isinstance(new_rows, list) or len(old_rows) != len(new_rows):
        raise ValueError("control row count differs from the 7026 pilot")
    fields = ("index", "k", "candidate_count", "true_sources", "snapshot_sources",
              "control_sources", "snapshot", "control")
    indices = [row.get("index") for row in old_rows]
    if len(set(indices)) != len(indices):
        raise ValueError("control reference contains duplicate indices")
    for old, new in zip(old_rows, new_rows):
        if any(old.get(field) != new.get(field) for field in fields):
            raise ValueError(f"control/index drift at example {old.get('index')}")


def _validate_repeat_resume(progress: Path, identity: dict, seed: int) -> None:
    """Verify the checkpoint digest and metadata before the trainer unpickles it."""
    try:
        saved = json.loads((progress / "input_identity.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("partial repeat identity is missing or corrupt") from exc
    if saved != identity:
        raise ValueError("partial repeat identity mismatch; choose a new output root")
    checkpoint = progress / "last_checkpoint.pt"
    sidecar = progress / "last_checkpoint.pt.sha256"
    if (not checkpoint.is_file() or not sidecar.is_file()
            or checkpoint.with_suffix(".pt.tmp").exists()
            or sidecar.with_suffix(".sha256.tmp").exists()):
        raise ValueError("partial repeat has no complete checkpoint and SHA-256 sidecar")
    if sidecar.read_text(encoding="ascii").strip() != sha256_file(checkpoint):
        raise ValueError("partial repeat checkpoint SHA-256 mismatch")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError("partial repeat checkpoint is corrupt") from exc
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "best_epoch",
                "best_score", "best_state_dict", "stale_epochs", "train_history",
                "validation_history", "python_random_state", "numpy_random_state",
                "torch_random_state", "cuda_random_states", "checkpoint_metadata"}
    if (not isinstance(state, dict) or not required <= set(state)
            or state["checkpoint_metadata"] != {"repeat_identity": identity, "seed": seed}):
        raise ValueError("partial repeat seed/metadata/identity mismatch")
    epoch = state["epoch"]
    if (not isinstance(epoch, int) or not 1 <= epoch <= 100
            or len(state["train_history"]) != epoch
            or len(state["validation_history"]) != epoch
            or not isinstance(state["model_state_dict"], dict)
            or not state["model_state_dict"]
            or not isinstance(state["optimizer_state_dict"], dict)
            or not state["optimizer_state_dict"]):
        raise ValueError("partial repeat checkpoint is corrupt")


def _finish_interrupted_finalization(root: Path, stage: str, progress: Path,
                                     identity: dict) -> dict | None:
    manifest_path = progress / "manifest.json"
    marker = progress / "complete"
    if not manifest_path.exists() and not marker.exists():
        return None
    if not manifest_path.is_file() or not marker.is_file():
        raise ValueError("interrupted finalization is incomplete; manual inspection required")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = manifest["file_hashes"]
        required = {"input_identity.json", "last_checkpoint.pt", "last_checkpoint.pt.sha256",
                    "history.json", "history.csv", "best_model.pt", "payload.json"}
        if (manifest.get("identity") != identity or manifest.get("schema_version") != 1
                or not required <= set(hashes)
                or any(Path(name).name != name or sha256_file(progress / name) != digest
                       for name, digest in hashes.items())):
            raise ValueError("interrupted finalization manifest/hash mismatch")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("interrupted finalization is corrupt") from exc
    progress.rename(root / stage)
    return read_stage(root, stage, identity)[1]


def _run_seed(paths: RepeatPaths, seed: int, device: torch.device, identity: dict,
              *, resume: bool, require_full: bool = True) -> dict:
    """Run one fixed training seed; CPU/small archives are permitted only for mechanics tests."""
    if seed not in (7027, 7028):
        raise ValueError("only fixed repeat seeds 7027 and 7028 are allowed")
    root = Path(paths.output_dir)
    stage = f"seed_{seed}"
    final = root / stage
    if final.exists():
        if resume:
            raise ValueError("completed seed cannot be resumed")
        return read_stage(root, stage, identity)[1]
    progress = root / f".{stage}-progress"
    if progress.exists():
        if not resume:
            raise ValueError("partial seed stage exists; explicit resume or new output root required")
        _validate_repeat_resume(progress, identity, seed)
        finalized = _finish_interrupted_finalization(root, stage, progress, identity)
        if finalized is not None:
            return finalized
    else:
        if resume:
            raise ValueError("no partial repeat checkpoint to resume")
        progress.mkdir(parents=True)
        (progress / "input_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    config = _config(paths.pilot.config_path)
    data_dir = Path(paths.pilot.data_dir)
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        n_train = len(archive["source_counts"])
    with np.load(data_dir / "validation.npz", allow_pickle=False) as archive:
        n_validation = len(archive["source_counts"])
    if require_full and (n_train, n_validation) != (9990, 1998):
        raise ValueError("full repeat requires exactly 9990 train and 1998 validation cascades")
    _, graph = load_graph_archive(data_dir / "graph.npz")
    builder = SnapshotFeatureBuilder(graph, distance_cap=int(config["data"].get("distance_cap", 10)))
    names = tuple(config["data"]["feature_names"])
    print(f"[{stage}] loading {n_train} train / {n_validation} validation cascades", flush=True)
    train = load_known_k_split(data_dir, "train", graph, builder, names, range(n_train))
    validation = load_known_k_split(data_dir, "validation", graph, builder, names,
                                    range(n_validation))
    set_seed(seed)
    model = NodeOnlyGCN(input_dim=9, hidden_dim=64, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    started = time.perf_counter()
    result = fit_node_model(
        model, train, validation, optimizer,
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        pos_weight=calculate_pos_weight(train).to(device),
        batch_size=config["training"]["batch_size"],
        checkpoint_path=progress / "last_checkpoint.pt",
        resume_from=progress / "last_checkpoint.pt" if resume else None,
        checkpoint_metadata={"repeat_identity": identity, "seed": seed},
    )
    elapsed = time.perf_counter() - started
    save_training_result(result, progress)
    paired = _evaluate_pilot(model, validation, graph, Path(paths.pilot.frozen_s1b_dir), device)
    old = read_stage(paths.pilot.output_dir, "pilot", identity["old_pilot_identity"])[1]
    _assert_same_control(old["paired_report"], paired)
    payload = {
        "stage": stage, "seed": seed, "exploratory": True,
        "n_train": n_train, "n_validation": n_validation,
        "best_epoch": result.best_epoch, "fit_elapsed_seconds": elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "paired_report": paired, "delta_f1": paired["delta_f1"],
        "count_accuracy": None, "output": str(final),
    }
    return _complete_binary_stage(root, stage, progress, identity, payload)


def run_repeat_stage(
    stage: Literal["freeze", "seed_7027", "seed_7028", "summary"],
    paths: RepeatPaths,
    device: torch.device,
    *,
    resume: bool = False,
) -> dict:
    if stage not in ("freeze", "seed_7027", "seed_7028", "summary"):
        raise ValueError("unknown repeat stage")
    verified = preflight(paths)
    identity = verified["identity"]
    root = Path(paths.output_dir)
    if stage == "freeze":
        if resume:
            raise ValueError("freeze cannot be resumed")
        if (root / "freeze").exists():
            return read_stage(root, "freeze", identity)[1]
        payload = {"status": "frozen", "exploratory": True,
                   "reference_seed": 7026, "repeat_seeds": [7027, 7028],
                   "old_pilot_gate": verified["pilot"]["quality_gate"]}
        write_stage(root, "freeze", {"identity": identity}, payload)
        return read_stage(root, "freeze", identity)[1]
    read_stage(root, "freeze", identity)
    if stage.startswith("seed_"):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("GPU required for full fixed repeats")
        return _run_seed(paths, int(stage.split("_")[1]), device, identity, resume=resume)
    raise NotImplementedError("summary will be implemented after seed tests")
