"""Seal a generated final holdout without computing target-dependent metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SEED_KEYS = ("simulation_seeds", "observation_seeds")
REQUIRED_ARCHIVE_KEYS = (*SEED_KEYS, "source_labels", "source_counts")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_snapshot_holdout(
    holdout_dir: str | Path,
    reference_dir: str | Path,
    *,
    split: str = "final_holdout",
    manifest_name: str = "holdout_manifest.json",
) -> dict[str, Any]:
    """Validate seed isolation and write a non-evaluative holdout manifest."""
    holdout_path = Path(holdout_dir)
    reference_path = Path(reference_dir)
    archive_path = holdout_path / f"{split}.npz"
    required_files = (
        holdout_path / "graph.npz",
        archive_path,
        holdout_path / "config.yaml",
        holdout_path / "generation_summary.json",
    )
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing holdout artifacts: {', '.join(missing)}")

    config = yaml.safe_load((holdout_path / "config.yaml").read_text(encoding="utf-8"))
    configured_splits = config.get("dataset", {}).get("splits", {})
    if set(configured_splits) != {split}:
        raise ValueError(f"Holdout config must contain only the {split!r} split.")

    with np.load(archive_path, allow_pickle=False) as archive:
        missing_keys = [key for key in REQUIRED_ARCHIVE_KEYS if key not in archive]
        if missing_keys:
            raise ValueError(f"Holdout archive is missing keys: {missing_keys}")
        example_count = len(archive["source_counts"])
        holdout_seeds = {
            key: np.asarray(archive[key], dtype=np.int64) for key in SEED_KEYS
        }
        if any(len(values) != example_count for values in holdout_seeds.values()):
            raise ValueError("Holdout seed arrays are not aligned with source_counts.")
        if any(len(np.unique(values)) != len(values) for values in holdout_seeds.values()):
            raise ValueError("Holdout seed arrays must be unique within the split.")

    reference_seed_sets = _load_reference_seed_sets(reference_path)
    overlaps = {
        key: sorted(set(values.tolist()) & reference_seed_sets[key])
        for key, values in holdout_seeds.items()
    }
    if any(overlaps.values()):
        counts = {key: len(values) for key, values in overlaps.items()}
        raise ValueError(f"Holdout seeds overlap reference splits: {counts}")

    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in required_files
    }
    manifest = {
        "schema_version": 1,
        "version": "snapshot-v2",
        "role": "confirmatory_final_holdout",
        "evaluation_status": "sealed_unopened",
        "split": split,
        "example_count": example_count,
        "dataset_seed": int(config["dataset"]["seed"]),
        "reference_dataset": str(reference_path),
        "seed_isolation": {
            "disjoint_from_reference": True,
            **{
                f"{key}_sha256": hashlib.sha256(values.tobytes()).hexdigest()
                for key, values in holdout_seeds.items()
            },
        },
        "policy": {
            "target_aggregates_computed": False,
            "allowed_before_freeze": ["file_integrity", "schema", "seed_isolation"],
            "forbidden_before_freeze": [
                "target_metrics",
                "feature_normalization",
                "hyperparameter_tuning",
                "checkpoint_selection",
            ],
        },
        "artifacts": artifacts,
    }
    manifest_path = holdout_path / manifest_name
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return manifest


def _load_reference_seed_sets(reference_dir: Path) -> dict[str, set[int]]:
    result = {key: set() for key in SEED_KEYS}
    archives = [reference_dir / f"{split}.npz" for split in ("train", "validation", "test")]
    missing = [str(path) for path in archives if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference splits: {', '.join(missing)}")
    for path in archives:
        with np.load(path, allow_pickle=False) as archive:
            for key in SEED_KEYS:
                if key not in archive:
                    raise ValueError(f"Reference archive {path} is missing {key}.")
                result[key].update(np.asarray(archive[key], dtype=np.int64).tolist())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--split", default="final_holdout")
    args = parser.parse_args(argv)
    manifest = seal_snapshot_holdout(
        args.holdout, args.reference, split=args.split
    )
    print(
        json.dumps(
            {
                "evaluation_status": manifest["evaluation_status"],
                "example_count": manifest["example_count"],
                "manifest": str(args.holdout / "holdout_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
