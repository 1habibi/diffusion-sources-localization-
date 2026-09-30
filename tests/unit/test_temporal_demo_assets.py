"""The UI must restore only the authenticated local frozen S1b model."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from diffusion_sources.temporal_demo_assets import load_temporal_resources


BACKUP = Path(__file__).resolve().parents[2] / "reports/backups/temporal_v3_20260929"
GRAPH = "data/reference/graph.npz"
GENERATION = "data/reference/config.yaml"
CONFIG = "runs/s1b/seed_7026/config.yaml"
WEIGHTS = "runs/s1b/seed_7026/best_model.pt"


def _small_backup(tmp_path: Path, *, weights: bool = True) -> Path:
    for relative in ("local_paths.json", "backup_manifest.json", GRAPH, GENERATION, CONFIG):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BACKUP / relative, target)
    if weights:
        target = tmp_path / WEIGHTS
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BACKUP / WEIGHTS, target)
    return tmp_path


def _reauthorize_fixture_file(root: Path, relative: str) -> None:
    manifest_path = root / "backup_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = root / relative
    for entry in manifest["files"]:
        if entry["path"] == relative:
            entry["bytes"] = actual.stat().st_size
            entry["sha256"] = hashlib.sha256(actual.read_bytes()).hexdigest()
            break
    else:
        raise AssertionError(f"Missing manifest entry: {relative}")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_frozen_s1b_cpu() -> None:
    resources = load_temporal_resources(BACKUP)
    assert resources.graph.number_of_nodes() == 4039
    assert resources.graph.number_of_edges() == 88234
    assert not resources.model.training
    assert sum(p.numel() for p in resources.model.parameters()) == 21252
    assert len(resources.model_config["data"]["feature_names"]) == 5
    assert resources.builder.distance_cache_path is None


def test_missing_checkpoint_names_path(tmp_path: Path) -> None:
    root = _small_backup(tmp_path, weights=False)
    with pytest.raises(FileNotFoundError, match="best_model.pt"):
        load_temporal_resources(root)


def test_manifest_hash_mismatch_rejected(tmp_path: Path) -> None:
    root = _small_backup(tmp_path)
    with (root / CONFIG).open("a", encoding="utf-8") as stream:
        stream.write("\n# unauthorized edit\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_temporal_resources(root)


@pytest.mark.parametrize("bad_input", ["model", "graph"])
def test_wrong_graph_or_model_config_rejected(tmp_path: Path, bad_input: str) -> None:
    root = _small_backup(tmp_path)
    if bad_input == "model":
        path = root / CONFIG
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["model"]["input_dim"] = 4
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        _reauthorize_fixture_file(root, CONFIG)
    else:
        path = root / GRAPH
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["edges"] = arrays["edges"][:-1]
        np.savez_compressed(path, **arrays)
        _reauthorize_fixture_file(root, GRAPH)
    with pytest.raises(ValueError):
        load_temporal_resources(root)
