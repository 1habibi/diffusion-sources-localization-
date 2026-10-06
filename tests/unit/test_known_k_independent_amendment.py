"""Post-open correction for a numerically tied auxiliary S1b ranking."""

import copy

import pytest
import torch

from diffusion_sources.temporal_pilot_artifacts import sha256_file, write_stage
from tests.unit.test_known_k_independent_artifacts import case  # noqa: F401
from tests.unit.test_known_k_independent_inference import sealed_case  # noqa: F401
from tests.unit.test_known_k_independent_summary import _report, _three


def _snapshot_tie(rows):
    """At k=3, these symmetric choices have the same descriptive metrics."""
    rows[1440]["snapshot_estimated_sources"] = [2]
    rows[1440]["snapshot_sources"] = [1, 2, 9]


def test_pairing_accepts_only_auxiliary_snapshot_rank_tie():
    from scripts.known_k_independent_amendment import pairing_audit

    reference = _report(7026, 80)["rows"]
    current = copy.deepcopy(reference)
    _snapshot_tie(current)
    audit = pairing_audit(reference, current)
    assert audit["core_pairing_exact"] is True
    assert audit["auxiliary_mismatch_counts"] == {
        "snapshot_estimated_k": 0,
        "snapshot_estimated_sources": 1,
        "snapshot_sources": 1,
        "snapshot_estimated": 0,
        "snapshot": 0,
    }
    assert audit["first_auxiliary_mismatch_indices"] == [1440]


@pytest.mark.parametrize("field, changed", [
    ("early_mask_hash", "f" * 64),
    ("candidate_mask_hash", "f" * 64),
    ("true_sources", [0, 1, 9]),
    ("control_sources", [0, 2, 9]),
    ("control", {"f1": -1}),
])
def test_pairing_rejects_changed_core_input_or_control(field, changed):
    from scripts.known_k_independent_amendment import pairing_audit

    reference = _report(7026, 80)["rows"]
    current = copy.deepcopy(reference)
    current[1440][field] = changed
    with pytest.raises(ValueError, match="paired|control|input"):
        pairing_audit(reference, current)


def test_amended_summary_keeps_primary_gate_and_uses_one_canonical_s1b():
    from scripts.known_k_independent_amendment import summarize_amended_reports
    from scripts.known_k_independent_summary import summarize_reports

    baseline_reports = _three()
    current_reports = copy.deepcopy(baseline_reports)
    _snapshot_tie(current_reports[7027]["rows"])
    original = summarize_reports(baseline_reports)
    amended = summarize_amended_reports(current_reports)
    assert amended["primary"] == original["primary"]
    assert amended["mean"] == original["mean"]
    assert amended["snapshot_estimated"] == original["snapshot_estimated"]
    assert amended["evaluation_role"] == "independent_confirmation_post_open_amendment"
    assert amended["amendment"]["auxiliary_differences"]["7027"]["first_auxiliary_mismatch_indices"] == [1440]


@pytest.fixture
def amendment_case(sealed_case):
    from scripts.known_k_independent_inference import _seed_identity, open_evaluation

    open_evaluation(sealed_case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    write_stage(sealed_case.reports, "seed_7026",
                {"identity": _seed_identity(sealed_case, 7026)}, _report(7026, 80))
    script = sealed_case.repo / "scripts/known_k_independent_amendment.py"
    script.write_text("amendment code v1", encoding="utf-8")
    notebook = sealed_case.repo / "notebooks/colab_known_k_independent_amendment.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    return sealed_case, notebook


def test_amendment_freeze_is_separate_and_authenticates_original_artifacts(amendment_case):
    from scripts.known_k_independent_amendment import freeze_amendment, verify_amendment_freeze

    paths, notebook = amendment_case
    original_dataset = sha256_file(paths.data / "independent_holdout.npz")
    original_7026 = sha256_file(paths.reports / "seed_7026/payload.json")
    frozen = freeze_amendment(paths, notebook)
    assert frozen == verify_amendment_freeze(paths, notebook) == freeze_amendment(paths, notebook)
    assert frozen["status"] == "post_open_amendment_frozen"
    assert (paths.reports / "post_open_amendment_v1/freeze/complete").is_file()
    assert sha256_file(paths.data / "independent_holdout.npz") == original_dataset
    assert sha256_file(paths.reports / "seed_7026/payload.json") == original_7026
    with notebook.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash|identity|mismatch"):
        verify_amendment_freeze(paths, notebook)


def test_amended_seeds_and_summary_are_append_only(amendment_case, monkeypatch):
    from scripts import known_k_independent_amendment as module

    paths, notebook = amendment_case
    module.freeze_amendment(paths, notebook)

    def frozen_report(given_paths, seed, device):
        assert given_paths == paths and device.type == "cpu"
        report = _report(seed, {7027: 120, 7028: 140}[seed])
        if seed == 7027:
            _snapshot_tie(report["rows"])
        return report

    monkeypatch.setattr(module, "_collect_report", frozen_report)
    for seed in (7027, 7028):
        first = module.run_amended_seed(paths, notebook, seed, torch.device("cpu"))
        assert first == module.run_amended_seed(paths, notebook, seed, torch.device("cpu"))
        assert first["amendment"]["core_pairing_exact"] is True
        assert not (paths.reports / f"seed_{seed}").exists()
    summary = module.save_amended_summary(paths, notebook)
    assert summary == module.save_amended_summary(paths, notebook)
    assert summary["evaluation_role"] == "independent_confirmation_post_open_amendment"
    assert summary["amendment"]["post_open"] is True
    assert (paths.reports / "post_open_amendment_v1/summary/complete").is_file()


def test_amendment_refuses_control_divergence_without_writing_stage(amendment_case, monkeypatch):
    from scripts import known_k_independent_amendment as module

    paths, notebook = amendment_case
    module.freeze_amendment(paths, notebook)
    report = _report(7027, 120)
    report["rows"][1440]["control_sources"] = [0, 2, 9]
    monkeypatch.setattr(module, "_collect_report", lambda *args: report)
    with pytest.raises(ValueError, match="paired core input or control"):
        module.run_amended_seed(paths, notebook, 7027, torch.device("cpu"))
    assert not (paths.reports / "post_open_amendment_v1/seed_7027").exists()


def test_seed_7028_requires_authenticated_7027_before_compute(amendment_case, monkeypatch):
    from scripts import known_k_independent_amendment as module

    paths, notebook = amendment_case
    module.freeze_amendment(paths, notebook)
    monkeypatch.setattr(module, "_collect_report", lambda *args: pytest.fail("must not compute 7028"))
    with pytest.raises(ValueError, match="7027"):
        module.run_amended_seed(paths, notebook, 7028, torch.device("cpu"))
    assert not (paths.reports / "post_open_amendment_v1/seed_7028").exists()
