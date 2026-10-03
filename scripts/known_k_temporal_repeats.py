"""Fixed, append-only repeats of the frozen known-k Temporal-GCN pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml

from diffusion_sources.known_k_temporal_pilot import KnownKPaths, _config, _input_identity
from diffusion_sources.temporal_pilot_artifacts import read_stage, sha256_file, write_stage


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
    raise NotImplementedError(f"{stage} will be implemented after freeze tests")
