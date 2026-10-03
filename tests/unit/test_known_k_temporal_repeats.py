"""Regression tests for the append-only known-k repeat protocol."""

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_sources.known_k_temporal_pilot import KnownKPaths, _input_identity, _config, run_stage
from diffusion_sources.models import JointSourceCountGCN
from diffusion_sources.temporal_pilot_artifacts import sha256_file


def _fixture_paths(temporal_dataset, tmp_path):
    from scripts.known_k_temporal_repeats import RepeatPaths

    data_dir, _ = temporal_dataset
    frozen = tmp_path / "frozen_s1b"
    frozen.mkdir()
    config = yaml.safe_load(Path("configs/snapshot_v2/s1b_distance_position.yaml").read_text())
    config["data"]["feature_names"] = [
        "observed_infected", "log_degree_normalized", "mean_distance_to_observed_normalized",
        "max_distance_to_observed_normalized", "induced_observed_eccentricity_normalized",
    ]
    config["model"]["input_dim"] = 5
    (frozen / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    torch.save(JointSourceCountGCN(input_dim=5, hidden_dim=64, dropout=0.2).state_dict(),
               frozen / "best_model.pt")
    pilot = KnownKPaths(data_dir, frozen, tmp_path / "old_pilot", Path("configs/known_k_temporal_gcn_pilot.yaml"))
    run_stage("freeze", pilot, torch.device("cpu"))
    run_stage("smoke", pilot, torch.device("cpu"))
    identity = _input_identity(pilot, _config(pilot.config_path))
    stage = pilot.output_dir / "pilot"
    stage.mkdir()
    smoke = pilot.output_dir / "smoke"
    files = ["input_identity.json", "last_checkpoint.pt", "last_checkpoint.pt.sha256",
             "history.json", "history.csv", "best_model.pt"]
    for name in files:
        shutil.copy2(smoke / name, stage / name)
    payload = {"quality_gate": {"passed": True, "reasons": []},
               "paired_report": {"delta_f1": 0.03, "rows": []}}
    (stage / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    files.append("payload.json")
    manifest = {"schema_version": 1, "identity": identity,
                "file_hashes": {name: sha256_file(stage / name) for name in files}}
    (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (stage / "complete").write_text("complete\n", encoding="utf-8")
    protocol = tmp_path / "repeats.yaml"
    protocol.write_text(yaml.safe_dump({"seeds": [7027, 7028],
                                        "stop_on_nonpositive_delta": True,
                                        "pilot_config": "configs/known_k_temporal_gcn_pilot.yaml"}),
                        encoding="utf-8")
    notebook = tmp_path / "repeats.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    return RepeatPaths(pilot, tmp_path / "new_repeats", protocol, notebook)


def _reference_report(paths):
    from diffusion_sources.dataset import load_graph_archive
    from diffusion_sources.features import SnapshotFeatureBuilder
    from diffusion_sources.known_k_temporal_data import load_known_k_split
    from diffusion_sources.known_k_temporal_pilot import _evaluate_pilot
    from diffusion_sources.models import NodeOnlyGCN

    config = _config(paths.pilot.config_path)
    _, graph = load_graph_archive(paths.pilot.data_dir / "graph.npz")
    builder = SnapshotFeatureBuilder(graph, distance_cap=int(config["data"].get("distance_cap", 10)))
    cases = load_known_k_split(paths.pilot.data_dir, "validation", graph, builder,
                               tuple(config["data"]["feature_names"]), range(3))
    report = _evaluate_pilot(NodeOnlyGCN(input_dim=9, hidden_dim=64, dropout=0.2), cases,
                             graph, paths.pilot.frozen_s1b_dir, torch.device("cpu"))
    stage = paths.pilot.output_dir / "pilot"
    payload_path = stage / "payload.json"
    payload = json.loads(payload_path.read_text())
    payload["paired_report"] = report
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_hashes"]["payload.json"] = sha256_file(payload_path)
    manifest_path.write_text(json.dumps(manifest))
    return report


def test_preflight_authenticates_pilot_and_does_not_write(temporal_dataset, tmp_path):
    from scripts.known_k_temporal_repeats import preflight

    paths = _fixture_paths(temporal_dataset, tmp_path)
    result = preflight(paths)
    assert result["pilot"]["quality_gate"]["passed"] is True
    assert result["identity"]["old_pilot_identity"]["output_root"] == str(paths.pilot.output_dir.resolve())
    assert not paths.output_dir.exists()

    checkpoint = paths.pilot.output_dir / "pilot" / "best_model.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash|pilot"):
        preflight(paths)
    assert not paths.output_dir.exists()


def test_preflight_rejects_missing_complete_and_failed_gate(temporal_dataset, tmp_path):
    from scripts.known_k_temporal_repeats import preflight

    paths = _fixture_paths(temporal_dataset, tmp_path)
    (paths.pilot.output_dir / "pilot" / "complete").unlink()
    with pytest.raises(ValueError, match="incomplete|missing"):
        preflight(paths)
    (paths.pilot.output_dir / "pilot" / "complete").write_text("complete\n")
    stage = paths.pilot.output_dir / "pilot"
    payload_path = stage / "payload.json"
    payload = json.loads(payload_path.read_text())
    payload["quality_gate"]["passed"] = False
    payload_path.write_text(json.dumps(payload))
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_hashes"]["payload.json"] = sha256_file(payload_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gate"):
        preflight(paths)


def test_preflight_rejects_protected_output_overlap(temporal_dataset, tmp_path):
    from dataclasses import replace
    from scripts.known_k_temporal_repeats import preflight

    paths = _fixture_paths(temporal_dataset, tmp_path)
    for root in (paths.pilot.output_dir, paths.pilot.data_dir, paths.pilot.frozen_s1b_dir):
        for output in (root, root / "nested", root.parent):
            with pytest.raises(ValueError, match="overlap"):
                preflight(replace(paths, output_dir=output))


def test_freeze_rejects_changed_runner_protocol_or_notebook(temporal_dataset, tmp_path, monkeypatch):
    from scripts import known_k_temporal_repeats as repeats

    paths = _fixture_paths(temporal_dataset, tmp_path)
    first = repeats.run_repeat_stage("freeze", paths, torch.device("cpu"))
    assert repeats.run_repeat_stage("freeze", paths, torch.device("cpu")) == first
    paths.notebook_path.write_text("{\"changed\":true}", encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        repeats.run_repeat_stage("freeze", paths, torch.device("cpu"))


def test_cpu_seed_mechanics_uses_fixed_data_and_paired_control(temporal_dataset, tmp_path):
    from scripts.known_k_temporal_repeats import _run_seed, preflight, run_repeat_stage

    paths = _fixture_paths(temporal_dataset, tmp_path)
    reference = _reference_report(paths)
    run_repeat_stage("freeze", paths, torch.device("cpu"))
    identity = preflight(paths)["identity"]
    payload = _run_seed(paths, 7027, torch.device("cpu"), identity, resume=False,
                        require_full=False)
    assert payload["seed"] == 7027
    assert payload["n_train"] == 6 and payload["n_validation"] == 3
    assert payload["paired_report"]["count_accuracy"] is None
    assert payload["paired_report"]["beta"] == 0.5
    assert [row["index"] for row in payload["paired_report"]["rows"]] == [
        row["index"] for row in reference["rows"]]
    assert (paths.output_dir / "seed_7027" / "best_model.pt").is_file()
    assert torch.load(paths.output_dir / "seed_7027" / "best_model.pt",
                      weights_only=True, map_location="cpu")


def test_control_comparison_rejects_reordering_and_drift():
    from scripts.known_k_temporal_repeats import _assert_same_control

    rows = [
        {"index": 0, "k": 1, "candidate_count": 3, "true_sources": [1],
         "snapshot_sources": [2], "control_sources": [1],
         "snapshot": {"f1": 0.0}, "control": {"f1": 1.0}},
        {"index": 1, "k": 2, "candidate_count": 4, "true_sources": [1, 2],
         "snapshot_sources": [2, 3], "control_sources": [1, 3],
         "snapshot": {"f1": 0.5}, "control": {"f1": 0.5}},
    ]
    _assert_same_control({"rows": rows}, {"rows": rows})
    changed = [dict(row) for row in reversed(rows)]
    with pytest.raises(ValueError, match="control|index"):
        _assert_same_control({"rows": rows}, {"rows": changed})
    changed = [dict(row) for row in rows]
    changed[0]["control_sources"] = [2]
    with pytest.raises(ValueError, match="control"):
        _assert_same_control({"rows": rows}, {"rows": changed})
    changed = [dict(row) for row in rows]
    changed[0]["candidate_count"] = 4
    with pytest.raises(ValueError, match="control"):
        _assert_same_control({"rows": rows}, {"rows": changed})


def test_resume_rejects_wrong_seed_and_stale_sha_before_unpickling(temporal_dataset, tmp_path):
    from scripts.known_k_temporal_repeats import _validate_repeat_resume, preflight

    paths = _fixture_paths(temporal_dataset, tmp_path)
    identity = preflight(paths)["identity"]
    progress = paths.output_dir / ".seed_7027-progress"
    progress.mkdir(parents=True)
    (progress / "input_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    checkpoint = progress / "last_checkpoint.pt"
    state = torch.load(paths.pilot.output_dir / "smoke" / "last_checkpoint.pt",
                       weights_only=False, map_location="cpu")
    state["checkpoint_metadata"] = {"repeat_identity": identity, "seed": 7028}
    torch.save(state, checkpoint)
    (progress / "last_checkpoint.pt.sha256").write_text(sha256_file(checkpoint))
    with pytest.raises(ValueError, match="seed|metadata"):
        _validate_repeat_resume(progress, identity, 7027)
    state["checkpoint_metadata"]["seed"] = 7027
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_repeat_resume(progress, identity, 7027)
    (progress / "last_checkpoint.pt.sha256").write_text(sha256_file(checkpoint))
    _validate_repeat_resume(progress, identity, 7027)


def test_interrupted_finalization_recovers_only_verified_files(temporal_dataset, tmp_path):
    from scripts.known_k_temporal_repeats import (
        _finish_interrupted_finalization, _validate_repeat_resume, preflight,
    )

    paths = _fixture_paths(temporal_dataset, tmp_path)
    identity = preflight(paths)["identity"]
    root = paths.output_dir
    progress = root / ".seed_7027-progress"
    progress.mkdir(parents=True)
    source = paths.pilot.output_dir / "smoke"
    files = ["history.json", "history.csv", "best_model.pt"]
    for name in files:
        shutil.copy2(source / name, progress / name)
    state = torch.load(source / "last_checkpoint.pt", weights_only=False, map_location="cpu")
    state["checkpoint_metadata"] = {"repeat_identity": identity, "seed": 7027}
    torch.save(state, progress / "last_checkpoint.pt")
    (progress / "last_checkpoint.pt.sha256").write_text(
        sha256_file(progress / "last_checkpoint.pt"), encoding="ascii")
    (progress / "input_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (progress / "payload.json").write_text(json.dumps({"seed": 7027}), encoding="utf-8")
    files += ["last_checkpoint.pt", "last_checkpoint.pt.sha256", "input_identity.json", "payload.json"]
    manifest = {"schema_version": 1, "identity": identity,
                "file_hashes": {name: sha256_file(progress / name) for name in files}}
    (progress / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (progress / "complete").write_text("complete\n", encoding="utf-8")
    _validate_repeat_resume(progress, identity, 7027)
    assert _finish_interrupted_finalization(root, "seed_7027", progress, identity) == {"seed": 7027}
    assert not progress.exists()
    assert (root / "seed_7027" / "complete").is_file()
