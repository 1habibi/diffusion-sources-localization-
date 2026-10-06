"""Authenticated, append-only inputs for a new known-k independent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diffusion_sources.generation import graph_from_config
from diffusion_sources.known_k_temporal_pilot import KnownKPaths
from diffusion_sources.temporal_independent_artifacts import (
    _graph_topology, _runtime_versions, assert_disjoint_paths, load_yaml,
    metadata_seed_union, topology,
)
from diffusion_sources.temporal_independent_generation import SIMULATION, OBSERVATION, DATASET
from diffusion_sources.temporal_pilot_artifacts import read_stage, sha256_file, write_stage
from scripts.known_k_temporal_repeats import RepeatPaths, preflight as repeat_preflight

SEEDS = (7026, 7027, 7028)
ATTEMPT_WINDOW = (5007026, 5406625)
DATA_FILES = ("graph.npz", "independent_holdout.npz", "config.yaml", "generation_summary.json")
OWN_SCRIPTS = (
    "known_k_independent_artifacts.py", "known_k_independent_generation.py",
    "known_k_independent_inference.py", "known_k_independent_summary.py",
)


@dataclass(frozen=True)
class IndependentKnownKPaths:
    repo: Path
    reference: Path
    snapshot_holdout: Path
    prior_independent: Path
    pilot: Path
    repeats: Path
    s1b: Path
    data: Path
    reports: Path
    generation_config: Path
    notebook: Path


def _required_metadata(paths: IndependentKnownKPaths) -> list[Path]:
    return [*(paths.reference / f"{split}.npz" for split in ("train", "validation", "test")),
            paths.snapshot_holdout / "final_holdout.npz",
            paths.prior_independent / "independent_holdout.npz"]


def _generation_identity(paths: IndependentKnownKPaths) -> tuple[dict, Path, dict]:
    cfg = load_yaml(paths.generation_config)
    old = load_yaml(paths.reference / "config.yaml")
    expected_data = {**DATASET, "seed": ATTEMPT_WINDOW[0]}
    if (set(cfg) != {"graph", "simulation", "observation", "dataset"}
            or cfg["simulation"] != SIMULATION
            or {**cfg["observation"], "hide_source_count": cfg["observation"].get("hide_source_count", 0)} != OBSERVATION
            or cfg["dataset"] != expected_data
            or cfg["graph"].get("id") != "ego_facebook" or cfg["graph"].get("kind") != "edge_list"):
        raise ValueError("Known-k independent generation protocol mismatch")
    if (old["simulation"] != cfg["simulation"]
            or old["observation"] != cfg["observation"]
            or any(old["dataset"].get(key) != cfg["dataset"][key] for key in
                   ("min_candidates", "max_infected_fraction", "max_attempt_factor", "distance_cache_size"))):
        raise ValueError("Reference generation protocol mismatch")
    raw = Path(cfg["graph"]["path"])
    raw = raw if raw.is_absolute() else paths.repo / raw
    graph_id, graph = graph_from_config({**cfg["graph"], "path": str(raw)})
    graph_identity = topology(paths.reference / "graph.npz")
    if _graph_topology(graph_id, graph) != graph_identity:
        raise ValueError("Raw and reference graph topology differ")
    return cfg, raw, graph_identity


def preflight(paths: IndependentKnownKPaths) -> dict:
    """Authenticate prior experiments without touching target values or output roots."""
    assert_disjoint_paths([paths.data, paths.reports],
                          [paths.repo, paths.reference, paths.snapshot_holdout,
                           paths.prior_independent, paths.pilot, paths.repeats, paths.s1b])
    archives = _required_metadata(paths)
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(archive)
    prior_seeds = metadata_seed_union(archives)
    if any(ATTEMPT_WINDOW[0] <= seed <= ATTEMPT_WINDOW[1] for seed in prior_seeds):
        raise ValueError("Independent attempt seed window overlaps prior metadata")
    _, raw, graph_identity = _generation_identity(paths)
    pilot_paths = KnownKPaths(paths.reference, paths.s1b, paths.pilot,
                              paths.repo / "configs/known_k_temporal_gcn_pilot.yaml")
    repeat_paths = RepeatPaths(pilot_paths, paths.repeats,
                               paths.repo / "configs/known_k_temporal_gcn_repeats.yaml",
                               paths.repo / "notebooks/colab_known_k_temporal_gcn_repeats.ipynb")
    authenticated = repeat_preflight(repeat_paths)
    repeat_identity = authenticated["identity"]
    if authenticated["pilot"].get("quality_gate", {}).get("passed") is not True:
        raise ValueError("Known-k pilot gate did not pass")
    package_dir = Path(__file__).resolve().parents[1] / "src/diffusion_sources"
    package_hashes = {p.name: sha256_file(p) for p in sorted(package_dir.glob("*.py"))}
    if repeat_identity.get("package_hashes") != package_hashes:
        raise ValueError("Known-k pilot package source hash mismatch")
    for stage in ("freeze", "seed_7027", "seed_7028"):
        read_stage(paths.repeats, stage, repeat_identity)
    _, old_summary = read_stage(paths.repeats, "summary", repeat_identity)
    if (old_summary.get("status") != "completed"
            or old_summary.get("positive_delta_each_repeat") is not True
            or set(old_summary.get("repeat_seeds", {})) != {"7027", "7028"}):
        raise ValueError("Known-k repeat summary is not completed and positive")
    expected_repeat_manifests = {
        str(seed): sha256_file(paths.repeats / f"seed_{seed}/manifest.json")
        for seed in (7027, 7028)
    }
    if old_summary.get("seed_manifest_hashes") != expected_repeat_manifests:
        raise ValueError("Known-k summary does not reference the current repeat manifests")
    checkpoints = {"7026": sha256_file(paths.pilot / "pilot/best_model.pt"),
                   **{str(seed): sha256_file(paths.repeats / f"seed_{seed}/best_model.pt")
                      for seed in (7027, 7028)}}
    source_paths = [paths.repo / "scripts" / name for name in OWN_SCRIPTS]
    source_paths.append(paths.notebook)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = {
        "historical_repeats": repeat_identity,
        "historical_manifests": {stage: sha256_file(paths.repeats / stage / "manifest.json")
                                 for stage in ("freeze", "seed_7027", "seed_7028", "summary")},
        "historical_summary": sha256_file(paths.repeats / "summary/payload.json"),
        "checkpoints": checkpoints,
        "s1b": {name: sha256_file(paths.s1b / name) for name in ("best_model.pt", "config.yaml")},
        "reference_archives": {str(p.resolve()): sha256_file(p) for p in archives},
        "reference_graph": sha256_file(paths.reference / "graph.npz"),
        "raw_graph": sha256_file(raw), "topology": graph_identity,
        "generation_config": sha256_file(paths.generation_config),
        "evaluation_code": {str(p.resolve()): sha256_file(p) for p in source_paths},
        "package_hashes": package_hashes, "versions": _runtime_versions(),
        "paths": {"data": str(paths.data.resolve()), "reports": str(paths.reports.resolve())},
        "protocol": {"dataset_seed": ATTEMPT_WINDOW[0], "attempt_window": list(ATTEMPT_WINDOW),
                     "n": 1998, "seeds": list(SEEDS), "beta": .5, "t1": 1,
                     "bootstrap": {"repetitions": 2000, "seed": 9282026},
                     "primary_min_delta": .02},
    }
    return {"identity": identity, "summary": old_summary}


def stage_identity(paths: IndependentKnownKPaths, stage: str) -> dict:
    if stage == "freeze":
        return preflight(paths)["identity"]
    verify_freeze(paths)
    if stage == "seal":
        return {"freeze": sha256_file(paths.reports / "freeze/payload.json"),
                "files": {name: sha256_file(paths.data / name) for name in DATA_FILES}}
    raise ValueError(f"Unsupported stage identity: {stage}")


def verify_freeze(paths: IndependentKnownKPaths) -> dict:
    return read_stage(paths.reports, "freeze", stage_identity(paths, "freeze"))[1]


def freeze_inputs(paths: IndependentKnownKPaths) -> dict:
    identity = stage_identity(paths, "freeze")
    if (paths.reports / "freeze").exists():
        return read_stage(paths.reports, "freeze", identity)[1]
    payload = {"role": "parameter_artifact_freeze", "status": "frozen", "identity": identity}
    write_stage(paths.reports, "freeze", {"identity": identity}, payload)
    return read_stage(paths.reports, "freeze", identity)[1]
