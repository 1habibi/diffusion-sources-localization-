"""Guarded training entrypoint for the known-k two-observation pilot."""

from pathlib import Path

import pytest
import torch
import yaml

from diffusion_sources.known_k_temporal_pilot import KnownKPaths, main, run_stage
from diffusion_sources.models import NodeOnlyGCN
from diffusion_sources.train_cli import set_seed


CONFIG_PATH = Path("configs/known_k_temporal_gcn_pilot.yaml")


def _paths(data_dir, output_dir, config_path=CONFIG_PATH):
    return KnownKPaths(data_dir, output_dir / "frozen_s1b", output_dir / "results", config_path)


def test_pilot_configuration_is_single_fixed_experiment():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["model"] == {"architecture": "node_only", "input_dim": 14,
                               "hidden_dim": 64, "dropout": 0.2}
    assert config["training"] == {"seed": 7026, "learning_rate": 0.001,
                                  "batch_size": 3, "max_epochs": 100, "patience": 10}
    assert len(config["data"]["feature_names"]) == 10
    assert config["evaluation"]["evaluate_test"] is False
    assert config["evaluation"]["control_beta"] == 0.5


def test_smoke_updates_weights_and_saves_best_checkpoint(temporal_dataset, tmp_path):
    data_dir, _ = temporal_dataset
    paths = _paths(data_dir, tmp_path)
    set_seed(7026)
    initial = NodeOnlyGCN(input_dim=14, hidden_dim=64, dropout=0.2)
    initial_weights = {name: value.detach().clone() for name, value in initial.state_dict().items()}

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
    code = main(["smoke", "--data-dir", str(paths.data_dir),
                 "--frozen-s1b-dir", str(paths.frozen_s1b_dir),
                 "--output-dir", str(paths.output_dir),
                 "--config", str(paths.config_path), "--device", "cpu"])
    assert code == 0
    assert (paths.output_dir / "smoke" / "best_model.pt").exists()
    assert "training_performed" in capsys.readouterr().out
