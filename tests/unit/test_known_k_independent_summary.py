"""Pre-registered known-k independent three-seed decision."""

from functools import lru_cache

import networkx as nx
import numpy as np
import pytest

from diffusion_sources.metrics import set_metrics, source_radius_hits, source_set_distances
from diffusion_sources.temporal_pilot_artifacts import write_stage
from tests.unit.test_known_k_independent_artifacts import case  # noqa: F401
from tests.unit.test_known_k_independent_inference import sealed_case  # noqa: F401


@lru_cache(None)
def _metrics(k, sources, operational=False):
    graph = nx.path_graph(10)
    truth = frozenset(range(k))
    result = {**set_metrics(truth, sources), **source_set_distances(graph, truth, sources),
              **source_radius_hits(graph, truth, sources)}
    if not operational:
        result.pop("count_accuracy")
        result.pop("count_mae")
    return result


def _report(seed, improved):
    rows = []
    for index in range(1998):
        k = index // 666 + 1
        truth = list(range(k))
        control = list(range(k - 1)) + [9]
        candidate = truth if index < improved else control
        rows.append({
            "index": index, "k": k, "candidate_count": 10,
            "candidate_ids": list(range(10)), "true_sources": truth,
            "early_mask_hash": "a" * 64, "candidate_mask_hash": "b" * 64,
            "snapshot_estimated_k": 1, "snapshot_estimated_sources": [0],
            "snapshot_sources": control, "control_sources": control,
            "candidate_sources": candidate,
            "snapshot_estimated": _metrics(k, (0,), True),
            "snapshot": _metrics(k, tuple(control)),
            "control": _metrics(k, tuple(control)),
            "candidate": _metrics(k, tuple(candidate)),
        })
    return {"seed": seed, "beta": .5, "n": 1998, "count_accuracy": None,
            "evaluation_role": "independent_confirmation", "exploratory": False,
            "early_mask_hash": "same", "candidate_mask_hash": "same", "rows": rows,
            "delta_f1": float(np.mean([r["candidate"]["f1"] - r["control"]["f1"] for r in rows]))}


def _three(improvements=(80, 120, 140)):
    return {seed: _report(seed, count) for seed, count in zip((7026, 7027, 7028), improvements)}


def test_bootstrap_uses_1998_cascade_means(monkeypatch):
    from scripts import known_k_independent_summary as module

    seen = []

    def bootstrap(deltas, true_k, *, repetitions=2000, seed=9282026):
        seen.append((np.asarray(deltas).copy(), np.asarray(true_k).copy(), repetitions, seed))
        return (.01, .10)

    monkeypatch.setattr(module, "paired_bootstrap_ci", bootstrap)
    result = module.summarize_reports(_three())
    primary = seen[-1]
    assert primary[0].shape == (1998,)
    assert primary[1].shape == (1998,)
    assert primary[2:] == (2000, 9282026)
    assert primary[0][0] == 1.0
    assert primary[0][100] == pytest.approx(2 / 3)
    assert primary[0][150] == 0.0
    assert result["primary"]["delta_f1"] == pytest.approx((80 + 120 + 140) / (3 * 1998))
    assert result["primary"]["passed"] is True


@pytest.mark.parametrize("improvements,ci,reason", [
    ((80, 120, 0), (.01, .10), "seed"),
    ((3, 3, 3), (.01, .10), "mean"),
    ((80, 120, 140), (0., .10), "CI"),
])
def test_gate_requires_all_three_conditions(monkeypatch, improvements, ci, reason):
    from scripts import known_k_independent_summary as module

    monkeypatch.setattr(module, "paired_bootstrap_ci", lambda *a, **kw: ci)
    result = module.summarize_reports(_three(improvements))
    assert result["primary"]["passed"] is False
    assert any(reason.lower() in item.lower() for item in result["primary"]["reasons"])


def test_snapshot_estimated_k_is_separate_from_known_k_gate(monkeypatch):
    from scripts import known_k_independent_summary as module

    monkeypatch.setattr(module, "paired_bootstrap_ci", lambda *a, **kw: (.01, .10))
    result = module.summarize_reports(_three())
    assert result["count_accuracy"] is None
    assert result["snapshot_estimated"]["mean"]["count_accuracy"] == pytest.approx(1 / 3)
    assert result["snapshot_estimated"]["mean"]["count_mae"] == pytest.approx(1.0)
    assert "count_accuracy" not in result["metrics"]["control"]["mean"]
    assert "count_accuracy" not in result["metrics"]["candidate"]["mean"]


def test_subgroups_publish_secondary_metrics_for_both_known_k_arms(monkeypatch):
    from scripts import known_k_independent_summary as module

    monkeypatch.setattr(module, "paired_bootstrap_ci", lambda *a, **kw: (.01, .10))
    result = module.summarize_reports(_three())
    wanted = {"precision", "recall", "f1", "exact_set_accuracy",
              "symmetric_set_distance", "hit_at_1_hop", "hit_at_2_hop"}
    for family in ("by_k", "by_candidates"):
        for group in result[family].values():
            for arm in ("control", "candidate"):
                assert wanted <= set(group[arm])
                assert "count_accuracy" not in group[arm]
                if not group["n_per_seed"]:
                    assert all(value is None for value in group[arm].values())
            if not group["n_per_seed"]:
                continue
            assert group["control_f1"] == pytest.approx(group["control"]["f1"])
            assert group["candidate_f1"] == pytest.approx(group["candidate"]["f1"])


def test_missing_seed_rejected():
    from scripts.known_k_independent_summary import summarize_reports

    reports = _three()
    reports.pop(7028)
    with pytest.raises(ValueError, match="three|seed"):
        summarize_reports(reports)


@pytest.mark.parametrize("field,replacement", [
    ("true_sources", [9]), ("early_mask_hash", "c" * 64),
    ("control_sources", [8]), ("candidate_ids", list(range(1, 11))),
])
def test_altered_pairing_rejected(field, replacement):
    from scripts.known_k_independent_summary import summarize_reports

    reports = _three()
    reports[7028]["rows"][1000][field] = replacement
    with pytest.raises(ValueError, match="pair|row|candidate"):
        summarize_reports(reports)


def test_summary_authenticates_three_seed_stages(sealed_case):
    from scripts.known_k_independent_inference import _seed_identity, open_evaluation
    from scripts.known_k_independent_summary import save_summary

    open_evaluation(sealed_case, "OPEN_KNOWN_K_INDEPENDENT_HOLDOUT")
    reports = _three()
    for seed, report in reports.items():
        write_stage(sealed_case.reports, f"seed_{seed}",
                    {"identity": _seed_identity(sealed_case, seed)}, report)
    first = save_summary(sealed_case)
    assert first == save_summary(sealed_case)
    assert first["evaluation_role"] == "independent_confirmation"
    with (sealed_case.reports / "seed_7028/payload.json").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash|mismatch"):
        save_summary(sealed_case)
