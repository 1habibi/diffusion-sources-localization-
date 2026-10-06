"""One-shot, target-blind generation for the known-k independent set."""

from pathlib import Path
import json
import shutil

import numpy as np
import pytest
import yaml

from tests.unit.test_known_k_independent_artifacts import case  # noqa: F401


def test_only_seed_changes_from_prior_independent_config():
    old = yaml.safe_load(Path("configs/facebook_temporal_v3_independent_holdout.yaml").read_text())
    new = yaml.safe_load(Path("configs/facebook_known_k_temporal_gcn_independent_holdout.yaml").read_text())
    assert new["dataset"]["seed"] == 5007026
    new["dataset"]["seed"] = old["dataset"]["seed"]
    assert new == old


def test_attempt_window_collision_stops_before_freeze(case):
    from scripts.known_k_independent_artifacts import preflight

    for seed in (5007026, 5406625):
        np.savez(case.prior_independent / "independent_holdout.npz",
                 simulation_seeds=[seed], observation_seeds=[seed + 1])
        with pytest.raises(ValueError, match="seed window"):
            preflight(case)


def _fake_generator(case, calls):
    def generate(config, output):
        calls.append(config["dataset"]["seed"])
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case.reference / "graph.npz", out / "graph.npz")
        (out / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (out / "generation_summary.json").write_text(json.dumps({"accepted": 1998}), encoding="utf-8")
        n, nodes = 1998, 10
        np.savez_compressed(out / "independent_holdout.npz",
            features=np.zeros((n, nodes, 2), dtype=np.float32),
            candidate_masks=np.ones((n, nodes), dtype=bool),
            source_labels=np.zeros((n, nodes), dtype=bool),
            infected_masks=np.ones((n, nodes), dtype=bool),
            source_counts=np.ones(n, dtype=np.int64),
            simulation_seeds=5007026 + 2 * np.arange(n, dtype=np.int64),
            observation_seeds=5007027 + 2 * np.arange(n, dtype=np.int64),
            probabilities=np.full(n, .01),
            observation_fractions=np.ones(n))
    return generate


def test_seal_is_target_blind_and_idempotent(case, monkeypatch):
    from scripts import known_k_independent_generation as module
    from scripts.known_k_independent_artifacts import freeze_inputs

    freeze_inputs(case)
    calls = []
    monkeypatch.setattr(module, "generate_dataset", _fake_generator(case, calls))
    real_load = np.load

    class CheckedArchive:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def __getitem__(self, name):
            assert name in ("simulation_seeds", "observation_seeds")
            return self.inner[name]

    def checked_load(file, *args, **kwargs):
        loaded = real_load(file, *args, **kwargs)
        return CheckedArchive(loaded) if str(file).endswith("independent_holdout.npz") else loaded

    monkeypatch.setattr(np, "load", checked_load)
    first = module.generate_and_seal(case)
    assert first["evaluation_status"] == "sealed_unopened"
    assert first["target_metrics_computed"] is False
    assert first == module.generate_and_seal(case) == module.verify_seal(case)
    assert calls == [5007026]


@pytest.mark.parametrize("existing", ["data", "partial"])
def test_existing_unsealed_output_is_not_replaced(case, monkeypatch, existing):
    from scripts import known_k_independent_generation as module
    from scripts.known_k_independent_artifacts import freeze_inputs

    freeze_inputs(case)
    calls = []
    monkeypatch.setattr(module, "generate_dataset", _fake_generator(case, calls))
    case.data.parent.mkdir(parents=True, exist_ok=True)
    target = case.data if existing == "data" else case.data.parent / f".{case.data.name}-partial"
    target.mkdir()
    with pytest.raises(ValueError, match="partial|unsealed|existing"):
        module.generate_and_seal(case)
    assert calls == []


@pytest.mark.parametrize("fault", ["duplicate_seed", "topology", "config", "changed_freeze"])
def test_fault_during_generation_leaves_no_seal(case, monkeypatch, fault):
    from scripts import known_k_independent_generation as module
    from scripts.known_k_independent_artifacts import freeze_inputs

    freeze_inputs(case)
    calls = []
    ordinary = _fake_generator(case, calls)

    def broken(config, output):
        ordinary(config, output)
        out = Path(output)
        if fault == "duplicate_seed":
            archive = out / "independent_holdout.npz"
            with np.load(archive) as saved:
                values = {name: saved[name] for name in saved.files}
            values["simulation_seeds"][1] = values["simulation_seeds"][0]
            np.savez_compressed(archive, **values)
        elif fault == "topology":
            np.savez(out / "graph.npz", graph_id="ego_facebook", node_count=10, edges=[(0, 9)])
        elif fault == "config":
            (out / "config.yaml").write_text("wrong: true", encoding="utf-8")
        else:
            case.generation_config.write_text("changed: true", encoding="utf-8")

    monkeypatch.setattr(module, "generate_dataset", broken)
    with pytest.raises((ValueError, KeyError)):
        module.generate_and_seal(case)
    assert not (case.reports / "seal").exists()
    assert not case.data.exists()
