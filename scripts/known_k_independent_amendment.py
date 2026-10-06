"""Audited post-open correction for unstable auxiliary S1b GPU ties.

The original sealed dataset, frozen models, primary control, and gate stay fixed.
Only the assertion that repeated auxiliary S1b node IDs must be bit-identical
is removed. Raw per-seed predictions remain available for inspection.
"""

from __future__ import annotations

from pathlib import Path

import torch

from diffusion_sources.temporal_pilot_artifacts import read_stage, sha256_file, write_stage
from scripts.known_k_independent_artifacts import SEEDS
from scripts.known_k_independent_artifacts import IndependentKnownKPaths
from scripts.known_k_independent_inference import (
    _collect_report, _seed_identity, validate_seed_rows, verify_opened,
)
from scripts.known_k_independent_summary import summarize_reports


CORE_FIELDS = (
    "index", "k", "candidate_count", "true_sources", "candidate_ids",
    "early_mask_hash", "candidate_mask_hash", "control_sources", "control",
    "snapshot_estimated_k", "snapshot_estimated", "snapshot",
)
AUXILIARY_FIELDS = (
    "snapshot_estimated_sources", "snapshot_sources",
)
AMENDMENT_DIRECTORY = "post_open_amendment_v1"


def pairing_audit(reference_rows: list[dict], current_rows: list[dict]) -> dict:
    """Require identical paired cases/control; audit auxiliary snapshot variation."""
    if len(reference_rows) != 1998 or len(current_rows) != 1998:
        raise ValueError("paired core input or control length mismatch")
    counts = {name: 0 for name in AUXILIARY_FIELDS}
    first_indices = []
    for index, (reference, current) in enumerate(zip(reference_rows, current_rows, strict=True)):
        if any(reference.get(name) != current.get(name) for name in CORE_FIELDS):
            raise ValueError(f"paired core input or control mismatch at row {index}")
        changed = False
        for name in AUXILIARY_FIELDS:
            if reference.get(name) != current.get(name):
                counts[name] += 1
                changed = True
        if changed and len(first_indices) < 20:
            first_indices.append(index)
    validate_seed_rows(reference_rows)
    validate_seed_rows(current_rows)
    return {"core_pairing_exact": True, "auxiliary_mismatch_counts": counts,
            "first_auxiliary_mismatch_indices": first_indices}


def summarize_amended_reports(reports: dict[int, dict]) -> dict:
    """Run the original gate with one canonical auxiliary S1b pass (seed 7026)."""
    if set(reports) != set(SEEDS):
        raise ValueError("Exactly the original three frozen seeds are required")
    reference_rows = reports[7026]["rows"]
    audits = {str(seed): pairing_audit(reference_rows, reports[seed]["rows"])
              for seed in SEEDS}
    normalized = {}
    for seed in SEEDS:
        normalized[seed] = {
            **reports[seed],
            "rows": [
                {**row, **{name: reference_rows[index][name] for name in AUXILIARY_FIELDS}}
                for index, row in enumerate(reports[seed]["rows"])
            ],
        }
    result = summarize_reports(normalized)
    result["evaluation_role"] = "independent_confirmation_post_open_amendment"
    result["amendment"] = {
        "post_open": True,
        "core_pairing_exact": True,
        "auxiliary_snapshot_reference_seed": 7026,
        "auxiliary_differences": audits,
        "reason": "GPU numerical ties in the auxiliary frozen S1b ranking; no model, data, control, or gate change",
    }
    result["interpretation"] += " Technical pairing correction was made after the dataset was opened."
    return result


def _root(paths: IndependentKnownKPaths) -> Path:
    return paths.reports / AMENDMENT_DIRECTORY


def _original_7026(paths: IndependentKnownKPaths) -> dict:
    return read_stage(paths.reports, "seed_7026", _seed_identity(paths, 7026))[1]


def _freeze_identity(paths: IndependentKnownKPaths, notebook: Path) -> dict:
    verify_opened(paths)
    original = _original_7026(paths)
    validate_seed_rows(original["rows"])
    if original.get("seed") != 7026 or original.get("evaluation_role") != "independent_confirmation":
        raise ValueError("Original 7026 report is not the frozen independent result")
    sources = (paths.repo / "scripts/known_k_independent_amendment.py", Path(notebook))
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    return {
        "original_stages": {stage: sha256_file(paths.reports / stage / "manifest.json")
                            for stage in ("freeze", "seal", "opened", "seed_7026")},
        "original_7026_payload": sha256_file(paths.reports / "seed_7026/payload.json"),
        "dataset": sha256_file(paths.data / "independent_holdout.npz"),
        "amendment_code": {str(source.resolve()): sha256_file(source) for source in sources},
        "protocol": {
            "version": 1, "post_open": True, "new_data": False, "training": False,
            "reranking_or_retuning": False, "primary_gate_unchanged": True,
            "canonical_auxiliary_seed": 7026,
            "core_fields": list(CORE_FIELDS), "auxiliary_fields": list(AUXILIARY_FIELDS),
        },
    }


def verify_amendment_freeze(paths: IndependentKnownKPaths, notebook: Path) -> dict:
    identity = _freeze_identity(paths, notebook)
    return read_stage(_root(paths), "freeze", identity)[1]


def freeze_amendment(paths: IndependentKnownKPaths, notebook: Path) -> dict:
    identity = _freeze_identity(paths, notebook)
    root = _root(paths)
    if (root / "freeze").exists():
        return read_stage(root, "freeze", identity)[1]
    if root.exists() and any(root.iterdir()):
        raise ValueError("Partial amendment root exists; do not overwrite")
    payload = {"status": "post_open_amendment_frozen", "identity": identity,
               "disclosure": "Corrects an auxiliary S1b GPU tie after opening; original stages remain untouched."}
    write_stage(root, "freeze", {"identity": identity}, payload)
    return read_stage(root, "freeze", identity)[1]


def _amended_seed_identity(paths: IndependentKnownKPaths, notebook: Path, seed: int) -> dict:
    if seed not in (7027, 7028):
        raise ValueError("Amendment only evaluates frozen seed 7027 or 7028")
    verify_amendment_freeze(paths, notebook)
    return {"amendment_freeze": sha256_file(_root(paths) / "freeze/manifest.json"),
            "original_7026": sha256_file(paths.reports / "seed_7026/manifest.json"),
            "seed": seed}


def run_amended_seed(paths: IndependentKnownKPaths, notebook: Path, seed: int,
                     device: torch.device) -> dict:
    identity = _amended_seed_identity(paths, notebook, seed)
    root = _root(paths)
    stage = f"seed_{seed}"
    if (root / stage).exists():
        saved = read_stage(root, stage, identity)[1]
        pairing_audit(_original_7026(paths)["rows"], saved["rows"])
        return saved
    if list(root.glob(f".{stage}-*")):
        raise ValueError(f"Partial amended {stage} stage; do not overwrite")
    if seed == 7028:
        previous = read_stage(root, "seed_7027", _amended_seed_identity(paths, notebook, 7027))[1]
        pairing_audit(_original_7026(paths)["rows"], previous["rows"])
    report = _collect_report(paths, seed, device)
    if (report.get("seed") != seed or report.get("n") != 1998
            or report.get("count_accuracy") is not None
            or report.get("evaluation_role") != "independent_confirmation"
            or report.get("exploratory") is not False):
        raise ValueError("Frozen report does not match independent known-k protocol")
    audit = pairing_audit(_original_7026(paths)["rows"], report["rows"])
    report["amendment"] = {"post_open": True, **audit}
    if _amended_seed_identity(paths, notebook, seed) != identity:
        raise ValueError("Amendment identity changed during frozen inference")
    write_stage(root, stage, {"identity": identity, "device": str(device)}, report)
    return read_stage(root, stage, identity)[1]


def save_amended_summary(paths: IndependentKnownKPaths, notebook: Path) -> dict:
    verify_amendment_freeze(paths, notebook)
    root = _root(paths)
    reports = {7026: _original_7026(paths)}
    stage_hashes = {"7026_original": sha256_file(paths.reports / "seed_7026/manifest.json")}
    for seed in (7027, 7028):
        stage = f"seed_{seed}"
        reports[seed] = read_stage(root, stage, _amended_seed_identity(paths, notebook, seed))[1]
        stage_hashes[str(seed)] = sha256_file(root / stage / "manifest.json")
    identity = {"amendment_freeze": sha256_file(root / "freeze/manifest.json"),
                "seed_manifests": stage_hashes}
    if (root / "summary").exists():
        return read_stage(root, "summary", identity)[1]
    result = summarize_amended_reports(reports)
    result["amendment"]["seed_manifest_hashes"] = stage_hashes
    write_stage(root, "summary", {"identity": identity}, result)
    return read_stage(root, "summary", identity)[1]
