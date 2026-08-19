from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from diffusion_sources.holdout import seal_snapshot_holdout


def _write_split(path: Path, simulation_seeds: list[int], observation_seeds: list[int]) -> None:
    count = len(simulation_seeds)
    np.savez_compressed(
        path,
        simulation_seeds=np.asarray(simulation_seeds),
        observation_seeds=np.asarray(observation_seeds),
        source_labels=np.zeros((count, 3), dtype=np.float32),
        source_counts=np.ones(count, dtype=np.int64),
    )


def _prepare_reference(path: Path) -> None:
    path.mkdir()
    for index, split in enumerate(("train", "validation", "test")):
        _write_split(path / f"{split}.npz", [10 + index], [20 + index])


def _prepare_holdout(path: Path, simulation_seeds: list[int]) -> None:
    path.mkdir()
    (path / "graph.npz").write_bytes(b"graph")
    (path / "generation_summary.json").write_text("{}", encoding="utf-8")
    config = {"dataset": {"seed": 100, "splits": {"final_holdout": len(simulation_seeds)}}}
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_split(
        path / "final_holdout.npz",
        simulation_seeds,
        [seed + 1000 for seed in simulation_seeds],
    )


def test_seal_snapshot_holdout_writes_integrity_manifest(tmp_path):
    reference = tmp_path / "reference"
    holdout = tmp_path / "holdout"
    _prepare_reference(reference)
    _prepare_holdout(holdout, [100, 102])

    manifest = seal_snapshot_holdout(holdout, reference)

    assert manifest["evaluation_status"] == "sealed_unopened"
    assert manifest["example_count"] == 2
    assert manifest["seed_isolation"]["disjoint_from_reference"] is True
    saved = json.loads((holdout / "holdout_manifest.json").read_text(encoding="utf-8"))
    assert saved["artifacts"]["final_holdout.npz"]["sha256"]
    assert saved["policy"]["target_aggregates_computed"] is False


def test_seal_snapshot_holdout_rejects_reference_seed_overlap(tmp_path):
    reference = tmp_path / "reference"
    holdout = tmp_path / "holdout"
    _prepare_reference(reference)
    _prepare_holdout(holdout, [10])

    with pytest.raises(ValueError, match="overlap"):
        seal_snapshot_holdout(holdout, reference)
