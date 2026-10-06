"""The correction notebook must not regenerate or silently reopen the dataset."""

import json
from pathlib import Path

import pytest


NOTEBOOK = Path("notebooks/colab_known_k_independent_amendment.ipynb")


def _code():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code = {cell["id"]: "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"}
    for name, source in code.items():
        compile(source, f"{NOTEBOOK}:{name}", "exec")
    return code


def test_notebook_has_separate_manual_stages_and_no_generation():
    code = _code()
    assert list(code) == ["setup", "paths", "preflight", "freeze", "stop_after_freeze",
                          "seed_7027", "stop_after_7027", "seed_7028", "stop_after_7028",
                          "summary"]
    combined = "\n".join(code.values())
    assert "generate_and_seal" not in combined
    assert "open_evaluation(" not in combined
    assert "fit_node_model" not in combined
    assert "PINNED_REVISION = input(" in code["setup"]
    assert "if REVISION != PINNED_REVISION:" in code["setup"]
    for name in ("stop_after_freeze", "stop_after_7027", "stop_after_7028"):
        with pytest.raises(RuntimeError, match="STOP"):
            exec(code[name], {})


def test_missing_original_drive_artifact_stops_before_any_amendment_stage(tmp_path):
    from scripts.known_k_independent_artifacts import IndependentKnownKPaths

    code = _code()
    scope = {"Path": Path, "PROJECT": tmp_path,
             "IndependentKnownKPaths": IndependentKnownKPaths,
             "ROOT_OVERRIDE": tmp_path / "missing-drive"}
    with pytest.raises(FileNotFoundError):
        exec(code["paths"], scope)
    assert not (tmp_path / "missing-drive").exists()
