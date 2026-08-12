from __future__ import annotations

from types import SimpleNamespace

import yaml

from diffusion_sources.workflow import run_workflow


def test_workflow_writes_manifest_with_mocked_stages(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    generation = {
        "graph": {}, "simulation": {}, "observation": {}, "dataset": {}
    }
    training = {"data": {}, "model": {}, "training": {}, "loss": {}}
    (config_dir / "generation.yaml").write_text(yaml.safe_dump(generation), encoding="utf-8")
    for name in ("joint", "node", "no_consistency"):
        (config_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(training), encoding="utf-8"
        )
    workflow_path = config_dir / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "run_robustness": False,
                "run_process_shift": False,
                "configs": {
                    "generation": "configs/generation.yaml",
                    "joint": "configs/joint.yaml",
                    "node": "configs/node.yaml",
                    "without_consistency": "configs/no_consistency.yaml",
                },
            }
        ),
        encoding="utf-8",
    )
    fake_generation = SimpleNamespace(
        graph_id="test", requested={"test": 1}, accepted={"test": 1},
        attempts={"test": 1}, rejections={"test": {}},
        duration_seconds={"test": 0.1}, output_dir="data"
    )
    fake_metrics = {
        "prediction_metrics": {
            "joint_estimated_k": {"all": {"f1": 0.5}},
            "node_thresholded": {"all": {"f1": 0.5}},
        }
    }
    monkeypatch.setattr("diffusion_sources.workflow.generate_dataset", lambda *args: fake_generation)
    monkeypatch.setattr("diffusion_sources.workflow.run_training", lambda *args: fake_metrics)
    monkeypatch.setattr("diffusion_sources.workflow.run_node_training", lambda *args: fake_metrics)
    monkeypatch.setattr("diffusion_sources.workflow.evaluate_baselines", lambda *args, **kwargs: {})
    monkeypatch.setattr("diffusion_sources.workflow.build_report", lambda *args: [])
    monkeypatch.setattr("diffusion_sources.workflow.build_ablation_report", lambda *args: [])

    result = run_workflow(workflow_path, tmp_path / "output")

    assert result["artifacts"]["generation"]["graph_id"] == "test"
    assert (tmp_path / "output" / "workflow_manifest.json").exists()
