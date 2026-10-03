"""Behavioral safety checks for the staged repeat Colab notebook."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


NOTEBOOK = Path("notebooks/colab_known_k_temporal_gcn_repeats.ipynb")
EXPECTED = ["setup", "paths", "preflight", "freeze", "approve_7027", "seed_7027",
            "gate_7027", "approve_7028", "seed_7028", "summary"]


def _cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"}
    assert list(cells) == EXPECTED
    for name, source in cells.items():
        compile(source, f"{NOTEBOOK}:{name}", "exec")
    return cells


def test_notebook_stages_and_setup_are_nontraining():
    cells = _cells()
    for name in ("setup", "paths", "preflight", "freeze"):
        assert "'seed_7027'" not in cells[name]
        assert "'seed_7028'" not in cells[name]
    assert "known-k-temporal-gcn-repeats" in cells["setup"]
    assert "known_k_temporal_gcn_repeats/v1" in cells["paths"]
    assert "diffusion-sources-known-k-repeats" in cells["setup"]
    assert "test.npz" not in "".join(cells.values())
    assert "independent_holdout" not in "".join(cells.values())


def test_seed_7027_requires_one_time_typed_approval(tmp_path):
    cells = _cells()
    output = tmp_path / "repeat"
    (output / "freeze").mkdir(parents=True)
    (output / "freeze" / "complete").write_text("complete")
    calls = []
    env = {"PATHS": SimpleNamespace(output_dir=output), "DEVICE": "cuda",
           "torch": SimpleNamespace(device=lambda value: value),
           "run_repeat_stage": lambda *args: calls.append(args) or
               {"seed": 7027, "best_epoch": 6,
                "paired_report": {"delta_f1": 0.03, "f1_ci": [0.01, 0.05],
                                  "control": {"all": {"f1": 0.55}},
                                  "candidate": {"all": {"f1": 0.58}}}},
           "input": lambda prompt: "WRONG"}
    with pytest.raises(RuntimeError, match="подтвержден|подтверждение"):
        exec(cells["approve_7027"], env)
    with pytest.raises(RuntimeError, match="approve_7027"):
        exec(cells["seed_7027"], env)
    assert not calls
    env["input"] = lambda prompt: "RUN_KNOWN_K_REPEAT_7027"
    exec(cells["approve_7027"], env)
    exec(cells["seed_7027"], env)
    assert [call[0] for call in calls] == ["seed_7027"]
    with pytest.raises(RuntimeError, match="approve_7027"):
        exec(cells["seed_7027"], env)


def test_seed_7028_is_blocked_after_nonpositive_7027(tmp_path):
    cells = _cells()
    output = tmp_path / "repeat"
    output.mkdir()
    calls = []
    env = {"PATHS": SimpleNamespace(output_dir=output), "DEVICE": "cuda",
           "torch": SimpleNamespace(device=lambda value: value),
           "run_repeat_stage": lambda *args: calls.append(args),
           "read_stage": lambda *args: ({}, {"seed": 7027, "delta_f1": -0.01}),
           "preflight": lambda paths: {"identity": {}},
           "input": lambda prompt: "RUN_KNOWN_K_REPEAT_7028"}
    exec(cells["gate_7027"], env)
    assert env["CAN_RUN_7028"] is False
    with pytest.raises(RuntimeError, match="7027|запрещён"):
        exec(cells["approve_7028"], env)
    with pytest.raises(RuntimeError, match="подтвержден|подтверждение|7027"):
        exec(cells["seed_7028"], env)
    assert not calls
