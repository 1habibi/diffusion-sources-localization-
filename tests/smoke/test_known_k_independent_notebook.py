"""Notebook execution boundaries for the sealed known-k holdout."""

import json
from pathlib import Path

import pytest


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


def test_missing_drive_path_stops_in_paths_cell(tmp_path):
    code = {cell["id"]: _source(cell) for cell in _cells() if cell["cell_type"] == "code"}
    from scripts.known_k_independent_artifacts import IndependentKnownKPaths

    # Run only the real path cell, with a missing Drive root. No seal callable is supplied.
    scope = {"Path": Path, "PROJECT": tmp_path, "IndependentKnownKPaths": IndependentKnownKPaths,
             "ROOT_OVERRIDE": tmp_path / "missing-drive"}
    with pytest.raises(FileNotFoundError):
        exec(code["paths"], scope)
    assert not (tmp_path / "missing-drive").exists()
