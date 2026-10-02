"""Guarded training entrypoint for the known-k two-observation pilot."""

from pathlib import Path
import json

import pytest
import torch
import yaml

from diffusion_sources.dataset import load_graph_archive
from diffusion_sources.features import SnapshotFeatureBuilder
from diffusion_sources.known_k_temporal_data import load_known_k_split
from diffusion_sources.known_k_temporal_pilot import (
    KnownKPaths, _evaluate_pilot, _validate_resume_progress, main, run_stage,
)
from diffusion_sources.models import JointSourceCountGCN, NodeOnlyGCN
from diffusion_sources.train_cli import set_seed


CONFIG_PATH = Path("configs/known_k_temporal_gcn_pilot.yaml")
FROZEN_FEATURES = [
    "observed_infected", "log_degree_normalized", "mean_distance_to_observed_normalized",
    "max_distance_to_observed_normalized", "induced_observed_eccentricity_normalized",
]


def _paths(data_dir, output_dir, config_path=CONFIG_PATH):
    frozen = output_dir / "frozen_s1b"
    frozen.mkdir(exist_ok=True)
    frozen_config = yaml.safe_load(
        Path("configs/snapshot_v2/s1b_distance_position.yaml").read_text(encoding="utf-8")
    )
    frozen_config["data"]["feature_names"] = FROZEN_FEATURES
    frozen_config["model"]["input_dim"] = 5
    (frozen / "config.yaml").write_text(yaml.safe_dump(frozen_config), encoding="utf-8")
    torch.save(JointSourceCountGCN(input_dim=5, hidden_dim=64, dropout=0.2).state_dict(),
               frozen / "best_model.pt")
    return KnownKPaths(data_dir, frozen, output_dir / "results", config_path)


def test_pilot_configuration_is_single_fixed_experiment():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["model"] == {"architecture": "node_only", "input_dim": 9,
                               "hidden_dim": 64, "dropout": 0.2}
    assert config["training"] == {"seed": 7026, "learning_rate": 0.001,
                                  "batch_size": 3, "max_epochs": 100, "patience": 10}
    assert config["data"]["feature_names"] == FROZEN_FEATURES
    assert config["evaluation"]["evaluate_test"] is False
    assert config["evaluation"]["control_beta"] == 0.5


def test_smoke_updates_weights_and_saves_best_checkpoint(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    set_seed(7026)
    initial = NodeOnlyGCN(input_dim=9, hidden_dim=64, dropout=0.2)
    initial_weights = {name: value.detach().clone() for name, value in initial.state_dict().items()}

    run_stage("freeze", paths, torch.device("cpu"))
    report = run_stage("smoke", paths, torch.device("cpu"))

    assert report["training_performed"] is True
    assert report["quality_gate"] is None
    assert report["n_train"] == 6 and report["n_validation"] == 3
    best = paths.output_dir / "smoke" / "best_model.pt"
    assert best.is_file()
    selected = torch.load(best, map_location="cpu", weights_only=True)
    assert any(not torch.equal(selected[name], initial_weights[name]) for name in initial_weights)
    assert not (paths.output_dir / "test").exists()
    assert not list(paths.output_dir.rglob("*test*predictions*"))


def test_pilot_rejects_cpu_and_changed_hyperparameters_before_training(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    run_stage("freeze", paths, torch.device("cpu"))
    run_stage("smoke", paths, torch.device("cpu"))
    with pytest.raises(ValueError, match="GPU"):
        run_stage("pilot", paths, torch.device("cpu"))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["training"]["max_epochs"] = 101
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config"):
        run_stage("pilot", _paths(data_dir, tmp_path, changed), torch.device("cuda"))
    assert not (paths.output_dir / "pilot").exists()


def test_cli_smoke_prints_output_and_creates_checkpoint(temporal_dataset, tmp_path, capsys):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    run_stage("freeze", paths, torch.device("cpu"))
    code = main(["smoke", "--data-dir", str(paths.data_dir),
                 "--frozen-s1b-dir", str(paths.frozen_s1b_dir),
                 "--output-dir", str(paths.output_dir),
                 "--config", str(paths.config_path), "--device", "cpu"])
    assert code == 0
    assert (paths.output_dir / "smoke" / "best_model.pt").exists()
    assert "training_performed" in capsys.readouterr().out


def test_freeze_hashes_only_allowed_inputs_and_rejects_changes(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    (data_dir / "test.npz").write_bytes(b"forbidden test data must not be opened")

    frozen = run_stage("freeze", paths, torch.device("cpu"))

    hashes = frozen["input_hashes"]
    assert {"graph.npz", "train.npz", "validation.npz", "config.yaml",
            "pilot_config.yaml", "s1b_config.yaml", "s1b_best_model.pt"} <= set(hashes)
    assert "test.npz" not in hashes
    assert run_stage("freeze", paths, torch.device("cpu")) == frozen
    with (data_dir / "validation.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="identity"):
        run_stage("smoke", paths, torch.device("cpu"))


def test_smoke_requires_freeze_and_rejects_corrupt_saved_checkpoint(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    with pytest.raises(ValueError, match="freeze"):
        run_stage("smoke", paths, torch.device("cpu"))
    run_stage("freeze", paths, torch.device("cpu"))
    first = run_stage("smoke", paths, torch.device("cpu"))
    assert run_stage("smoke", paths, torch.device("cpu")) == first
    best = paths.output_dir / "smoke" / "best_model.pt"
    with best.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        run_stage("smoke", paths, torch.device("cpu"))


def test_incomplete_smoke_does_not_overwrite_previous_attempt(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    run_stage("freeze", paths, torch.device("cpu"))
    progress = paths.output_dir / ".smoke-progress"
    progress.mkdir()
    (progress / "user_note.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="partial|incomplete"):
        run_stage("smoke", paths, torch.device("cpu"))
    assert (progress / "user_note.txt").read_text(encoding="utf-8") == "preserve"


def test_resume_rejects_changed_identity_and_corrupt_checkpoint(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    run_stage("freeze", paths, torch.device("cpu"))
    run_stage("smoke", paths, torch.device("cpu"))
    identity = json.loads((paths.output_dir / "freeze" / "manifest.json").read_text(encoding="utf-8"))["identity"]
    progress = paths.output_dir / ".pilot-progress"
    progress.mkdir()
    (progress / "input_identity.json").write_text(json.dumps({"wrong": True}), encoding="utf-8")
    (progress / "last_checkpoint.pt").write_bytes(
        (paths.output_dir / "smoke" / "last_checkpoint.pt").read_bytes()
    )
    with pytest.raises(ValueError, match="identity"):
        _validate_resume_progress(progress, identity)
    (progress / "input_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (progress / "last_checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        _validate_resume_progress(progress, identity)


def test_paired_validation_uses_same_cases_and_known_k(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    _, graph = load_graph_archive(data_dir / "graph.npz")
    names = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["data"]["feature_names"]
    cases = load_known_k_split(data_dir, "validation", graph, SnapshotFeatureBuilder(graph), names, [0, 1, 2])
    candidate = NodeOnlyGCN(input_dim=9, hidden_dim=64, dropout=0.2)

    report = _evaluate_pilot(candidate, cases, graph, paths.frozen_s1b_dir, torch.device("cpu"))

    assert [row["index"] for row in report["rows"]] == [0, 1, 2]
    assert all(len(row["candidate_sources"]) == row["k"] for row in report["rows"])
    assert all(len(row["control_sources"]) == row["k"] for row in report["rows"])
    assert report["count_accuracy"] is None
    assert report["exploratory"] is True


def test_freeze_rejects_output_inside_data_or_changed_s1b_features(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    overlap = KnownKPaths(data_dir, paths.frozen_s1b_dir, data_dir / "new_results", CONFIG_PATH)
    with pytest.raises(ValueError, match="overlaps"):
        run_stage("freeze", overlap, torch.device("cpu"))
    frozen_config = yaml.safe_load((paths.frozen_s1b_dir / "config.yaml").read_text(encoding="utf-8"))
    frozen_config["data"]["feature_names"][0] = "different_feature"
    (paths.frozen_s1b_dir / "config.yaml").write_text(yaml.safe_dump(frozen_config), encoding="utf-8")
    with pytest.raises(ValueError, match="S1b config"):
        run_stage("freeze", paths, torch.device("cpu"))


def test_freeze_rejects_checkpoint_incompatible_with_its_config(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    torch.save(JointSourceCountGCN(input_dim=10, hidden_dim=64, dropout=0.2).state_dict(),
               paths.frozen_s1b_dir / "best_model.pt")
    with pytest.raises(ValueError, match="frozen S1b checkpoint"):
        run_stage("freeze", paths, torch.device("cpu"))
    assert not (paths.output_dir / "freeze").exists()
