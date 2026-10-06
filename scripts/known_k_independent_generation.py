"""Generate and seal the predetermined independent set without inspecting targets."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from diffusion_sources.generation import generate_dataset
from diffusion_sources.temporal_independent_artifacts import load_yaml, metadata_seed_union, topology
from diffusion_sources.temporal_independent_generation import (
    DATASET, OBSERVATION, SIMULATION, _schema,
)
from diffusion_sources.temporal_pilot_artifacts import read_stage, write_stage
from scripts.known_k_independent_artifacts import (
    ATTEMPT_WINDOW, IndependentKnownKPaths, _required_metadata, stage_identity,
    verify_freeze,
)


def validate_generation_config(config: dict, repo: Path) -> dict:
    """Accept only the old IC protocol with the one predeclared seed change."""
    cfg = copy.deepcopy(config)
    expected_dataset = {**DATASET, "seed": ATTEMPT_WINDOW[0]}
    observation = {**cfg.get("observation", {})}
    observation.setdefault("hide_source_count", 0)
    if (set(cfg) != {"graph", "simulation", "observation", "dataset"}
            or cfg.get("simulation") != SIMULATION or observation != OBSERVATION
            or cfg.get("dataset") != expected_dataset
            or set(cfg.get("graph", {})) != {"id", "kind", "path"}
            or cfg["graph"]["id"] != "ego_facebook"
            or cfg["graph"]["kind"] != "edge_list"):
        raise ValueError("Known-k independent generation config differs from the frozen protocol")
    raw = Path(cfg["graph"]["path"])
    cfg["graph"]["path"] = str((raw if raw.is_absolute() else Path(repo) / raw).resolve())
    return cfg


def verify_seal(paths: IndependentKnownKPaths) -> dict:
    if not (paths.reports / "seal/complete").is_file():
        raise ValueError("Independent known-k dataset is not sealed")
    result = read_stage(paths.reports, "seal", stage_identity(paths, "seal"))[1]
    if (result.get("evaluation_status") != "sealed_unopened"
            or result.get("target_metrics_computed") is not False
            or result.get("dataset_seed") != ATTEMPT_WINDOW[0]
            or result.get("n") != 1998):
        raise ValueError("Invalid known-k seal payload")
    return result


def generate_and_seal(paths: IndependentKnownKPaths) -> dict:
    """One attempt only; a partial result is a stop, never an invitation to retry."""
    frozen = verify_freeze(paths)
    if (paths.reports / "seal").exists():
        return verify_seal(paths)
    cfg = validate_generation_config(load_yaml(paths.generation_config), paths.repo)
    references = metadata_seed_union(_required_metadata(paths))
    if any(ATTEMPT_WINDOW[0] <= seed <= ATTEMPT_WINDOW[1] for seed in references):
        raise ValueError("Reference seed overlap with known-k independent attempt window")
    if (paths.data.exists()
            or list(paths.data.parent.glob(f".{paths.data.name}-*"))
            or list(paths.reports.glob(".seal-*"))):
        raise ValueError("Existing unsealed or partial independent generation; no overwrite")
    paths.data.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{paths.data.name}-", dir=paths.data.parent))
    print("Generating fixed known-k independent dataset once; targets remain unopened.", flush=True)
    generate_dataset(cfg, temporary)
    if topology(temporary / "graph.npz") != frozen["identity"]["topology"]:
        raise ValueError("Generated graph topology or node mapping mismatch")
    if load_yaml(temporary / "config.yaml") != cfg:
        raise ValueError("Generated config mismatch")
    archive_path = temporary / "independent_holdout.npz"
    _schema(archive_path, frozen["identity"]["topology"]["nodes"])
    seeds = metadata_seed_union([archive_path])
    if len(seeds) != 3996 or seeds & references:
        raise ValueError("Independent seeds are duplicate or overlap references")
    with np.load(archive_path, allow_pickle=False) as archive:
        simulation = archive["simulation_seeds"]
        observation = archive["observation_seeds"]
    if (np.any(simulation < ATTEMPT_WINDOW[0])
            or np.any(observation > ATTEMPT_WINDOW[1])
            or np.any((simulation - ATTEMPT_WINDOW[0]) % 2)
            or not np.array_equal(observation, simulation + 1)
            or not np.all(np.diff(simulation) > 0)):
        raise ValueError("Independent seeds do not follow the frozen attempt sequence")
    verify_freeze(paths)  # Detect input/code mutation during generation.
    temporary.rename(paths.data)
    payload = {
        "evaluation_status": "sealed_unopened", "n": 1998,
        "dataset_seed": ATTEMPT_WINDOW[0], "target_metrics_computed": False,
        "seed_isolation": True,
        "seed_hashes": {"simulation": hashlib.sha256(simulation.tobytes()).hexdigest(),
                        "observation": hashlib.sha256(observation.tobytes()).hexdigest()},
    }
    write_stage(paths.reports, "seal", {"identity": stage_identity(paths, "seal")}, payload)
    print(f"Sealed: {paths.reports / 'seal/manifest.json'}. STOP before opening.", flush=True)
    return verify_seal(paths)
