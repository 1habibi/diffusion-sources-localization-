"""Notebook execution boundaries for the sealed known-k holdout."""

import json
from pathlib import Path

import pytest
import torch

from tests.unit.test_known_k_independent_artifacts import case  # noqa: F401
from tests.unit.test_known_k_independent_generation import _fake_generator
from tests.unit.test_known_k_independent_summary import _report


NOTEBOOK = Path("notebooks/colab_known_k_temporal_gcn_independent.ipynb")


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell):
    return "".join(cell["source"])


def test_notebook_stage_order_and_separate_outputs():
    cells = _cells()
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    for cell in code:
        compile(_source(cell), f"{NOTEBOOK}:{cell['id']}", "exec")
    ids = [cell["id"] for cell in code]
    assert ids == ["setup", "paths", "preflight", "freeze", "stop_after_freeze",
                   "seal", "stop_before_open", "open", "seed_7026",
                   "stop_after_7026", "seed_7027", "stop_after_7027",
                   "seed_7028", "stop_after_7028", "summary"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    combined = {cell["id"]: _source(cell) for cell in code}
    assert "FROZEN =" in combined["freeze"]
    assert "SEALED =" in combined["seal"]
    assert "REPORT_7026 =" in combined["seed_7026"]
    assert "REPORT_7027 =" in combined["seed_7027"]
    assert "REPORT_7028 =" in combined["seed_7028"]
    assert "SUMMARY =" in combined["summary"]


def test_run_all_stops_before_open_and_no_training():
    code = {cell["id"]: _source(cell) for cell in _cells() if cell["cell_type"] == "code"}
    with pytest.raises(RuntimeError, match="STOP"):
        exec(code["stop_before_open"], {})
    with pytest.raises(RuntimeError, match="STOP"):
        exec(code["stop_after_freeze"], {})
    for seed in (7026, 7027, 7028):
        with pytest.raises(RuntimeError, match="STOP"):
            exec(code[f"stop_after_{seed}"], {})
    assert "open_evaluation" not in code["seal"]
    assert "generate_and_seal" not in code["open"]
    assert "run_stage(" not in "\n".join(code.values())
    assert "fit_node_model" not in "\n".join(code.values())
    assert "evaluate_test" not in "\n".join(code.values())


def test_setup_requires_reviewed_revision_before_import():
    code = {cell["id"]: _source(cell) for cell in _cells() if cell["cell_type"] == "code"}
    setup = code["setup"]
    assert "PINNED_REVISION = input(" in setup
    assert "re.fullmatch(r'[0-9a-f]{40}', PINNED_REVISION)" in setup
    assert "if REVISION != PINNED_REVISION:" in setup
    assert setup.index("if REVISION != PINNED_REVISION:") < setup.index("from scripts import known_k_independent_artifacts")
    assert "'pull'" not in setup


def test_missing_drive_path_stops_in_paths_cell(tmp_path):
    code = {cell["id"]: _source(cell) for cell in _cells() if cell["cell_type"] == "code"}
    from scripts.known_k_independent_artifacts import IndependentKnownKPaths

    # Run only the real path cell, with a missing Drive root. No seal callable is supplied.
    scope = {"Path": Path, "PROJECT": tmp_path, "IndependentKnownKPaths": IndependentKnownKPaths,
             "ROOT_OVERRIDE": tmp_path / "missing-drive"}
    with pytest.raises(FileNotFoundError):
        exec(code["paths"], scope)
    assert not (tmp_path / "missing-drive").exists()


def test_synthetic_guarded_lifecycle(case, monkeypatch):
    """Exercise real stage boundaries with synthetic files, never a Drive dataset."""
    from scripts import known_k_independent_generation as generation
    from scripts import known_k_independent_inference as inference
    from scripts.known_k_independent_artifacts import freeze_inputs, verify_freeze
    from scripts.known_k_independent_summary import save_summary

    assert not case.reports.exists()
    frozen = freeze_inputs(case)
    assert frozen == verify_freeze(case)
    generated = []
    monkeypatch.setattr(generation, "generate_dataset", _fake_generator(case, generated))
    sealed = generation.generate_and_seal(case)
    assert generated == [5007026]
    assert sealed["evaluation_status"] == "sealed_unopened"
    assert sealed["target_metrics_computed"] is False
    with pytest.raises(ValueError, match="confirmation"):
        inference.open_evaluation(case, "wrong")
    assert not (case.reports / "opened").exists()
    opened = inference.open_evaluation(case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    assert opened["evaluation_status"] == "opened"

    def synthetic_report(paths, seed, device):
        assert paths == case and device.type == "cpu"
        return _report(seed, {7026: 80, 7027: 120, 7028: 140}[seed])

    monkeypatch.setattr(inference, "_collect_report", synthetic_report)
    reports = [inference.evaluate_seed(case, seed, torch.device("cpu"))
               for seed in (7026, 7027, 7028)]
    assert [report["seed"] for report in reports] == [7026, 7027, 7028]
    summary = save_summary(case)
    assert summary == save_summary(case)
    assert summary["evaluation_role"] == "independent_confirmation"
    with (case.reports / "seed_7027/payload.json").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash|mismatch"):
        save_summary(case)
