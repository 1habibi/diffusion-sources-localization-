"""Read-only restoration of the locally archived, frozen S1b demo model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
import yaml

from .dataset import load_graph_archive
from .features import SnapshotFeatureBuilder
from .models import JointSourceCountGCN


FEATURE_NAMES = (
    "observed_infected",
    "log_degree_normalized",
    "mean_distance_to_observed_normalized",
    "max_distance_to_observed_normalized",
    "induced_observed_eccentricity_normalized",
)


@dataclass(frozen=True)
class DemoResources:
    graph: nx.Graph
    model: JointSourceCountGCN
    builder: SnapshotFeatureBuilder
    generation_config: dict
    model_config: dict


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Backup path escapes its root: {relative}")
    return path


def load_temporal_resources(backup_root: Path) -> DemoResources:
    """Validate protected backup inputs and restore inference on CPU only."""
    root = Path(backup_root).resolve()
    mapping_path = root / "local_paths.json"
    manifest_path = root / "backup_manifest.json"
    manifest = _read_json(manifest_path)
    try:
        local_info = manifest["local_mapping"]
        files = manifest["files"]
        protocol = manifest["protocol"]
        if (not isinstance(local_info, dict) or not isinstance(local_info["sha256"], str)
                or not isinstance(files, list) or not isinstance(protocol, dict)):
            raise ValueError("wrong field types")
        protected = {}
        for row in files:
            if (not isinstance(row, dict) or not isinstance(row["path"], str)
                    or not isinstance(row["bytes"], int) or not isinstance(row["sha256"], str)):
                raise ValueError("invalid file record")
            protected[row["path"]] = row
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid backup manifest: {manifest_path}: {exc}") from exc
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    if local_info["sha256"] != _file_hash(mapping_path):
        raise ValueError(f"SHA256 mismatch: {mapping_path}")
    mapping = _read_json(mapping_path)
    try:
        reference = mapping["reference_data"]
        frozen_run = mapping["runs"]["7026"]
        if not isinstance(reference, str) or not reference or not isinstance(frozen_run, str) or not frozen_run:
            raise ValueError("invalid required paths")
        required = (
            reference + "/graph.npz",
            reference + "/config.yaml",
            frozen_run + "/config.yaml",
            frozen_run + "/best_model.pt",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid local paths: {mapping_path}: {exc}") from exc
    for relative in required:
        path = _inside(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = protected.get(relative)
        if entry is None or path.stat().st_size != entry["bytes"] or _file_hash(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")

    if protocol.get("model") != "S1b snapshot GCN + temporal correction" or protocol.get("beta") != 0.5 or protocol.get("t1") != 1:
        raise ValueError(f"Frozen backup protocol mismatch: {manifest_path}")

    graph_id, graph = load_graph_archive(_inside(root, required[0]))
    if graph_id != "ego_facebook" or graph.number_of_nodes() != 4039 or graph.number_of_edges() != 88234:
        raise ValueError("Frozen Facebook graph topology mismatch")
    if sorted(graph.nodes) != list(range(4039)):
        raise ValueError("Frozen Facebook graph node IDs mismatch")

    generation = yaml.safe_load(_inside(root, required[1]).read_text(encoding="utf-8"))
    config = yaml.safe_load(_inside(root, required[2]).read_text(encoding="utf-8"))
    mc = config["model"]
    names = config["data"]["feature_names"]
    if (config["training"]["seed"] != 7026 or tuple(names) != FEATURE_NAMES
            or mc.get("architecture") != "v1_joint" or mc.get("input_dim") != 5
            or mc.get("backbone_mode") != "plain_2"
            or mc.get("source_head_mode") != "local"
            or mc.get("source_head_strategy") != "shared"
            or mc.get("shortlist_mode") != "disabled"
            or mc.get("global_feature_dim") != 0):
        raise ValueError("Frozen S1b model configuration mismatch")
    if generation.get("graph", {}).get("id") != graph_id:
        raise ValueError("Generation config graph ID mismatch")

    model = JointSourceCountGCN(
        input_dim=int(mc["input_dim"]),
        hidden_dim=int(mc["hidden_dim"]),
        dropout=float(mc["dropout"]),
        backbone_mode=mc["backbone_mode"],
        source_head_mode=mc["source_head_mode"],
        source_head_strategy=mc["source_head_strategy"],
        shortlist_mode=mc["shortlist_mode"],
        global_feature_dim=int(mc["global_feature_dim"]),
    )
    state = torch.load(_inside(root, required[3]), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    builder = SnapshotFeatureBuilder(graph, distance_cap=int(config["data"]["distance_cap"]))
    return DemoResources(graph, model, builder, generation, config)
