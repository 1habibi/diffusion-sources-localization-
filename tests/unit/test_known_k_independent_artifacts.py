"""Safety checks for the new, not-yet-opened known-k evaluation."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from diffusion_sources.temporal_pilot_artifacts import write_stage, sha256_file
from tests.fixtures.temporal_independent import make_case


@pytest.fixture
def case(tmp_path, monkeypatch):
    from scripts import known_k_independent_artifacts as module

    old = make_case(tmp_path)
    repo = old.repo
    scripts = repo / "scripts"
    scripts.mkdir()
    for name in ("known_k_independent_artifacts.py", "known_k_independent_generation.py",
                 "known_k_independent_inference.py", "known_k_independent_summary.py"):
        (scripts / name).write_text(name, encoding="utf-8")
    notebook = repo / "notebooks" / "colab_known_k_temporal_gcn_independent.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}", encoding="utf-8")
    config = yaml.safe_load(old.generation_config.read_text(encoding="utf-8"))
    config["dataset"]["seed"] = 5007026
    old.generation_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    prior = tmp_path / "prior_independent"
    prior.mkdir()
    np.savez(prior / "independent_holdout.npz", simulation_seeds=[4007026], observation_seeds=[4007027])
    s1b = tmp_path / "s1b"
    s1b.mkdir()
    (s1b / "best_model.pt").write_bytes(b"frozen S1b")
    (s1b / "config.yaml").write_text("training: {seed: 7026}\n", encoding="utf-8")
    pilot = tmp_path / "known_pilot"
    repeats = tmp_path / "known_repeats"
    old_identity = {"package_hashes": {p.name: sha256_file(p) for p in
                    Path(module.__file__).resolve().parents[1].joinpath("src", "diffusion_sources").glob("*.py")},
                    "old_stage_manifest_hashes": {"freeze": "a", "smoke": "b", "pilot": "c"}}
    monkeypatch.setattr(module, "repeat_preflight", lambda paths: {
        "identity": old_identity, "pilot": {"quality_gate": {"passed": True}}})
    write_stage(repeats, "freeze", {"identity": old_identity}, {"status": "frozen"})
    for seed in (7027, 7028):
        stage = repeats / f"seed_{seed}"
        stage.mkdir()
        (stage / "best_model.pt").write_bytes(f"model {seed}".encode())
        (stage / "payload.json").write_text(json.dumps({"seed": seed, "delta_f1": .05}), encoding="utf-8")
        (stage / "manifest.json").write_text(json.dumps({"schema_version": 1, "identity": old_identity,
            "file_hashes": {"best_model.pt": sha256_file(stage / "best_model.pt"),
                            "payload.json": sha256_file(stage / "payload.json")}}), encoding="utf-8")
        (stage / "complete").write_text("complete\n", encoding="utf-8")
    pilot_stage = pilot / "pilot"
    pilot_stage.mkdir(parents=True)
    (pilot_stage / "best_model.pt").write_bytes(b"pilot model")
    write_stage(repeats, "summary", {"identity": old_identity},
                {"status": "completed", "positive_delta_each_repeat": True,
                 "repeat_seeds": {"7027": {}, "7028": {}},
                 "seed_manifest_hashes": {str(seed): sha256_file(repeats / f"seed_{seed}/manifest.json")
                                          for seed in (7027, 7028)}})
    return module.IndependentKnownKPaths(repo=repo, reference=old.reference,
        snapshot_holdout=old.snapshot_holdout, prior_independent=prior, pilot=pilot,
        repeats=repeats, s1b=s1b, data=tmp_path / "new_data",
        reports=tmp_path / "new_reports", generation_config=old.generation_config,
        notebook=notebook)


def test_missing_snapshot_metadata_stops(case):
    from scripts.known_k_independent_artifacts import preflight

    (case.snapshot_holdout / "final_holdout.npz").unlink()
    with pytest.raises(FileNotFoundError, match="final_holdout.npz"):
        preflight(case)
    assert not case.reports.exists()


def test_output_overlap_has_no_writes(case):
    from scripts.known_k_independent_artifacts import preflight

    bad = replace(case, reports=case.reference / "nested")
    with pytest.raises(ValueError, match="overlap"):
        preflight(bad)
    assert not bad.reports.exists()


def test_tampered_frozen_input_rejected(case):
    from scripts.known_k_independent_artifacts import freeze_inputs, verify_freeze

    freeze_inputs(case)
    (case.repeats / "seed_7028" / "best_model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash|mismatch"):
        verify_freeze(case)


@pytest.mark.parametrize("target", ["s1b_config", "repeat_manifest", "evaluation_code",
                                    "prior_independent", "reference_graph"])
def test_frozen_identity_detects_changed_input(case, target):
    from scripts.known_k_independent_artifacts import freeze_inputs, verify_freeze

    freeze_inputs(case)
    files = {
        "s1b_config": case.s1b / "config.yaml",
        "repeat_manifest": case.repeats / "seed_7027" / "manifest.json",
        "evaluation_code": case.repo / "scripts/known_k_independent_generation.py",
        "prior_independent": case.prior_independent / "independent_holdout.npz",
        "reference_graph": case.reference / "graph.npz",
    }
    with files[target].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises((ValueError, OSError)):
        verify_freeze(case)


def test_resigned_but_failed_historical_summary_rejected(case):
    from scripts.known_k_independent_artifacts import preflight

    stage = case.repeats / "summary"
    payload_file = stage / "payload.json"
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    payload["status"] = "stopped_after_7027"
    payload_file.write_text(json.dumps(payload), encoding="utf-8")
    manifest_file = stage / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["file_hashes"]["payload.json"] = sha256_file(payload_file)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        preflight(case)


def test_resigned_summary_must_reference_exact_repeat_manifests(case):
    from scripts.known_k_independent_artifacts import preflight

    stage = case.repeats / "summary"
    payload_file = stage / "payload.json"
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    payload["seed_manifest_hashes"]["7028"] = "0" * 64
    payload_file.write_text(json.dumps(payload), encoding="utf-8")
    manifest_file = stage / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["file_hashes"]["payload.json"] = sha256_file(payload_file)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="summary.*manifest"):
        preflight(case)


def test_freeze_is_idempotent_and_authenticates_inputs(case):
    from scripts.known_k_independent_artifacts import freeze_inputs, verify_freeze

    first = freeze_inputs(case)
    assert first == freeze_inputs(case) == verify_freeze(case)
    identity = first["identity"]
    assert set(identity["checkpoints"]) == {"7026", "7027", "7028"}
    assert identity["s1b"]["best_model.pt"] == sha256_file(case.s1b / "best_model.pt")
    assert identity["protocol"]["dataset_seed"] == 5007026
    assert identity["protocol"]["bootstrap"] == {"repetitions": 2000, "seed": 9282026}
    assert not case.data.exists()
