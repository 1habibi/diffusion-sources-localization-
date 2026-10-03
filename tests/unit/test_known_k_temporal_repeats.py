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

