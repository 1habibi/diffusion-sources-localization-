"""Colab notebook stages must be explicit and stop before costly training."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


NOTEBOOK = Path(__file__).parents[2] / "notebooks/colab_known_k_temporal_gcn_pilot.ipynb"


def _cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"}
    for name, source in code.items():
        compile(source, f"notebook:{name}", "exec")
    return code


def test_setup_local_only_imports_project_and_does_not_run_stages(capsys):
    source = _cells()
    assert list(source) == ["setup", "paths", "freeze", "smoke", "approve", "pilot"]
    saved_path = list(sys.path)
    ns = {}
    try:
        exec(source["setup"], ns)
    finally:
        sys.path[:] = saved_path
    assert Path(ns["PILOT"].__file__).resolve().is_relative_to(ns["PROJECT"].resolve())
    assert ns["APPROVED_PILOT"] is False
    assert "Setup не запускал freeze, smoke или обучение" in capsys.readouterr().out


def test_freeze_smoke_and_pilot_are_separate_with_explicit_approval(tmp_path, capsys):
    source = _cells()
    calls = []
    paths = SimpleNamespace(output_dir=tmp_path)
    pilot = SimpleNamespace(run_stage=lambda stage, p, device: calls.append((stage, device.type))
                            or {"stage": stage, "quality_gate": None})
    ns = {"PILOT": pilot, "PATHS": paths, "torch": torch, "DEVICE": "cuda",
          "json": json, "APPROVED_PILOT": False, "input": lambda _: "wrong"}

    exec(source["freeze"], ns)
    exec(source["smoke"], ns)
    assert calls == [("freeze", "cpu"), ("smoke", "cpu")]
    with pytest.raises(RuntimeError, match="подтверж"):
        exec(source["pilot"], ns)
    assert len(calls) == 2
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke/complete").write_text("complete", encoding="utf-8")
    with pytest.raises(RuntimeError, match="подтверж"):
        exec(source["approve"], ns)
    assert len(calls) == 2
    ns["input"] = lambda _: "RUN_KNOWN_K_TEMPORAL_PILOT"
    exec(source["approve"], ns)
    exec(source["pilot"], ns)
    assert calls == [("freeze", "cpu"), ("smoke", "cpu"), ("pilot", "cuda")]
    assert ns["APPROVED_PILOT"] is False
    assert "STOP" in capsys.readouterr().out


def test_approval_refuses_missing_smoke_or_cpu(tmp_path):
    source = _cells()
    ns = {"PATHS": SimpleNamespace(output_dir=tmp_path), "DEVICE": "cuda",
          "APPROVED_PILOT": False, "input": lambda _: "RUN_KNOWN_K_TEMPORAL_PILOT"}
    with pytest.raises(RuntimeError, match="smoke"):
        exec(source["approve"], ns)
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke/complete").write_text("complete", encoding="utf-8")
    ns["DEVICE"] = "cpu"
    with pytest.raises(RuntimeError, match="GPU"):
        exec(source["approve"], ns)
