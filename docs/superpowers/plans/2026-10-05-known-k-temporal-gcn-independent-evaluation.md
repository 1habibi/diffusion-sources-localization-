# Known-k Temporal-GCN Independent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the three frozen known-k Temporal-GCN checkpoints on one new sealed Facebook IC holdout without retraining, retuning, or changing the existing modes.

**Architecture:** New config, scripts, tests and Colab notebook live outside `src/diffusion_sources/*.py`, whose complete hash set is part of the old pilot identity. Reuse existing package replay, feature, model, scoring and stage-authentication functions read-only; authenticate all historical inputs before generating once, seal before explicit open, then store append-only paired seed reports and one summary.

**Tech Stack:** Python, NumPy, PyTorch/PyG, NetworkX, PyYAML, pytest, Colab/Google Drive.

**Spec:** `docs/superpowers/specs/2026-10-05-known-k-temporal-gcn-independent-evaluation-design.md`

## Global Constraints

- Preserve Snapshot S1b estimated-k, Temporal-v3 unknown-k, old known-k checkpoints, Streamlit UI, `master`, and `thesis_report/` unchanged.
- New synthetic IC holdout: same `ego_facebook` mapping/protocol, exactly 1,998 cases (666 per true `k`), dataset seed `5007026`, attempt-seed window `[5007026,5406625]`; no selection on outcomes.
- Frozen primary pair: GCN 7026/7027/7028 versus S1b/7026-based known-k Temporal-v3, early `t=1`, beta `0.5`; snapshot S1b estimated-k is a separate descriptive result.
- No new fit, seed/checkpoint/beta/model choice, old package edits, test or snapshot-holdout label access, or reuse of the already opened independent set.
- Primary rule: mean three-seed paired ΔF1 ≥ `0.02`, 95% stratified paired CI lower > `0`, and each seed ΔF1 > `0`; bootstrap 2,000 replicates with RNG seed `9282026`, over 1,998 cascade-index means.
- New Drive roots: `data/generated/facebook_known_k_temporal_gcn_independent_v1` and `reports/runs/known_k_temporal_gcn_independent_evaluation/v1`; append-only stages and exact typed confirmation `OPEN_KNOWN_K_INDEPENDENT_HOLDOUT`.

## Review Focus

1. Missing historical snapshot-holdout metadata: preflight stops with the missing path; it never creates a substitute (Task 1 test).
2. A protected output root nested inside an input or vice versa: reject before any write (Task 1 test).
3. Stale or partially written generation/open stage: fail closed; neither overwrite nor quietly regenerate (Tasks 2–3 tests).
4. Same index but changed early mask or control predictions across seed runs: reject the report/summary (Tasks 4–5 tests).
5. One negative GCN seed hidden by a positive mean, or 5,994 predictions pooled as independent: gate must fail or bootstrap must use 1,998 averaged cascade deltas (Task 5 tests).

## File map

- Create `configs/facebook_known_k_temporal_gcn_independent_holdout.yaml`: exact new dataset generation config.
- Create `scripts/known_k_independent_artifacts.py`: paths, historical preflight, freeze/identity and verification for append-only stages.
- Create `scripts/known_k_independent_generation.py`: fixed generation, seed isolation and target-blind seal.
- Create `scripts/known_k_independent_inference.py`: typed open, frozen replay/inference and per-seed authenticated reports.
- Create `scripts/known_k_independent_summary.py`: alignment checks, cascade-paired statistics, gate and separate snapshot table.
- Create `notebooks/colab_known_k_temporal_gcn_independent.ipynb`: visible stage-by-stage Colab entry point, never `Run all`.
- Create `docs/colab_known_k_temporal_gcn_independent.md`: exact cells, paths, expected stops and no-retuning interpretation.
- Create corresponding `tests/unit/test_known_k_independent_{artifacts,generation,inference,summary}.py` and `tests/smoke/test_known_k_independent_notebook.py`; modify `development_log.md` only to record completed, verified milestones.

---

### Task 1: Authenticate frozen inputs and freeze identity

**Files:** Create config, `scripts/known_k_independent_artifacts.py`, `tests/unit/test_known_k_independent_artifacts.py`.

**Interfaces:** `IndependentKnownKPaths` is a frozen dataclass with `Path` fields `repo, reference, snapshot_holdout, prior_independent, pilot, repeats, s1b, data, reports, generation_config, notebook`; `preflight(paths: IndependentKnownKPaths) -> dict`; `freeze_inputs(paths: IndependentKnownKPaths) -> dict`; `stage_identity(paths: IndependentKnownKPaths, stage: str) -> dict`; `verify_freeze(paths: IndependentKnownKPaths) -> dict`. Later tasks import these exact names.

- [ ] **Step 1: Write failing tests.** `test_missing_snapshot_metadata_stops`: `with pytest.raises(FileNotFoundError, match="final_holdout.npz"): preflight(paths)`. `test_output_overlap_has_no_writes`: assert `ValueError` and `not paths.reports.exists()`. `test_tampered_frozen_input_rejected`: change each named historical file/status/hash, assert `preflight` fails. `test_freeze_is_idempotent`: assert `freeze_inputs(paths) == freeze_inputs(paths)` and SHA identity covers all named inputs.
- [ ] **Step 2: Run `pytest tests/unit/test_known_k_independent_artifacts.py -q`; expect FAIL for missing module/functions.**
- [ ] **Step 3: Implement interfaces.** Reuse `read_stage`, `write_stage`, `sha256_file`, `scripts.known_k_temporal_repeats.preflight`, and `metadata_seed_union` without editing package sources. Require pilot `quality_gate.passed`, repeat `status=completed`, seeds 7027/7028 and `positive_delta_each_repeat=true`; compare current package hashes to pilot identity. Resolve paths and reject overlapping outputs before creation. Freeze includes the complete script/notebook/config hashes, seed window and evaluation decision rule.
- [ ] **Step 4: Run `pytest tests/unit/test_known_k_independent_artifacts.py -q`; expect PASS.**
- [ ] **Step 5: Commit `feat: freeze known-k independent inputs`.**

### Task 2: Generate once and seal without target inspection

**Files:** Create `scripts/known_k_independent_generation.py`, `tests/unit/test_known_k_independent_generation.py`.

**Interfaces:** `validate_generation_config(config: dict, repo: Path) -> dict`; `generate_and_seal(paths: IndependentKnownKPaths) -> dict`; `verify_seal(paths) -> dict`. Consumes Task 1 freeze/identity.

- [ ] **Step 1: Write failing tests.** `test_only_seed_changes`: assert old/new configs differ only at `dataset.seed == 5007026`. `test_seed_window_collision`: inject `5007026` and `5406625` into reference metadata; assert `ValueError`. `test_seal_is_target_blind_and_idempotent`: forbidden target-array accessor raises if called, then assert first/second seal payload equal and generator called once. Separate parametrized tests reject duplicate seeds, topology/config drift, changed freeze and existing partial/output directories.
- [ ] **Step 2: Run `pytest tests/unit/test_known_k_independent_generation.py -q`; expect FAIL.**
- [ ] **Step 3: Implement interfaces.** Follow `temporal_independent_generation.py` mechanics but use the new fixed config and old train/validation/test, prior independent and snapshot seed metadata. Use temporary sibling directory, atomically publish only complete generation, schema-check 1,998 rows, hash files and write `sealed_unopened`. Never choose a replacement seed or overwrite a partial artifact.
- [ ] **Step 4: Run `pytest tests/unit/test_known_k_independent_generation.py -q`; expect PASS.**
- [ ] **Step 5: Commit `feat: seal known-k independent dataset`.**

### Task 3: Explicit open and frozen single-seed inference

**Files:** Create `scripts/known_k_independent_inference.py`, `tests/unit/test_known_k_independent_inference.py`.

**Interfaces:** `open_evaluation(paths: IndependentKnownKPaths, confirmation: str) -> dict`; `verify_opened(paths) -> dict`; `evaluate_seed(paths: IndependentKnownKPaths, seed: int, device: torch.device) -> dict`. Consumes Tasks 1–2 identities; writes `opened`, `seed_7026`, `seed_7027`, `seed_7028` stages.

- [ ] **Step 1: Write failing tests.** `test_open_requires_exact_token`: assert wrong string raises and no `opened` directory, exact token writes one authenticated marker. `test_target_read_requires_open`: target-array accessor spies on `verify_opened` and rejects reads first. `test_frozen_pairing`: mocked two-case replay asserts `make_observation` has no label argument, checkpoint paths equal frozen paths, control beta is `0.5`, and S1b operational `k` comes from its count head. Parametrize invalid seed, bad seal, changed early/control hash, partial stage and completed-stage repeat.
- [ ] **Step 2: Run `pytest tests/unit/test_known_k_independent_inference.py -q`; expect FAIL.**
- [ ] **Step 3: Implement interfaces.** Reuse `replay_early_mask`, `make_observation`, `NodeOnlyGCN`, `_load_frozen_s1b`, `predict_joint`, `predict_oracle_k`, `correct_sources`, and existing metric functions read-only. Build separate inference-only input before attaching truth for scoring. Save all 1,998 ordered rows (`index`, true `k`/sources, candidate count, early/candidate fingerprints, fixed control and candidate source sets/metrics, S1b estimated-k sets/metrics), per-seed aggregates and SHA-checked stage. Mark `exploratory=false`, `evaluation_role=independent_confirmation`, known-k count accuracy `null`.
- [ ] **Step 4: Run `pytest tests/unit/test_known_k_independent_inference.py -q`; expect PASS.**
- [ ] **Step 5: Commit `feat: evaluate frozen known-k seeds on opened holdout`.**

### Task 4: Parity and metrics at the seed boundary

**Files:** Modify `scripts/known_k_independent_inference.py`; extend `tests/unit/test_known_k_independent_inference.py`.

**Interfaces:** `validate_seed_rows(rows: list[dict], reference_rows: list[dict] | None = None, *, expected_n: int = 1998) -> None`; called before publishing a seed stage. Produces complete, finite, exactly aligned rows for Task 5.

- [ ] **Step 1: Write failing tests.** `test_missing_or_reordered_row`: assert `ValueError` for 1,997 rows and shuffled/duplicate indices. `test_changed_pairing_rejected`: mutate one truth/candidate/early/control field against reference rows and assert `ValueError` each time. `test_known_k_metrics_do_not_fake_count`: assert report count accuracy is `None` and hand-calculated exact/distance/Hit@1/2 values match. Include nonfinite scores and source-outside-candidates cases.
- [ ] **Step 2: Run `pytest tests/unit/test_known_k_independent_inference.py -q`; expect new tests FAIL.**
- [ ] **Step 3: Implement `validate_seed_rows` and metric parity before `write_stage`;** enforce full index order `0..1997` by default, while tiny synthetic tests pass `expected_n=2`. Compare saved reference rows as well as hashes, not only summary F1.
- [ ] **Step 4: Run `pytest tests/unit/test_known_k_independent_inference.py -q`; expect PASS.**
- [ ] **Step 5: Commit `test: enforce paired known-k inference parity`.**

### Task 5: Pre-registered three-seed summary

**Files:** Create `scripts/known_k_independent_summary.py`, `tests/unit/test_known_k_independent_summary.py`.

**Interfaces:** `summarize_reports(reports: dict[int, dict]) -> dict`; `save_summary(paths: IndependentKnownKPaths) -> dict`. Requires exact keys `{7026,7027,7028}` and authenticated Task 3 reports.

- [ ] **Step 1: Write failing tests.** `test_bootstrap_uses_cascade_means`: spy on `paired_bootstrap_ci` and assert input shape `(1998,)`, repetitions `2000`, seed `9282026`; each index value equals the mean of three seed deltas. `test_gate_requires_all_three`: inject one nonpositive seed, low mean or CI lower≤0 and assert `passed is False`. `test_snapshot_separate`: assert estimated-k count accuracy/MAE appear outside the primary pair and known-k count accuracy is `None`. Parametrize missing seed and altered pairing fields.
- [ ] **Step 2: Run `pytest tests/unit/test_known_k_independent_summary.py -q`; expect FAIL.**
- [ ] **Step 3: Implement interfaces.** Authenticate each seed stage, verify complete alignment and identical S1b/control rows, aggregate precision/recall/F1/exact/distance/Hit@1/2 and descriptive k/candidate slices. Store explicit pass/fail reasons, negative results, per-seed values, mean/sample SD, 95% CI, provenance and limitation to synthetic cascades on the same graph. Repeated summary reads the authenticated saved stage.
- [ ] **Step 4: Run `pytest tests/unit/test_known_k_independent_summary.py -q`; expect PASS.**
- [ ] **Step 5: Commit `feat: summarize paired known-k independent results`.**

### Task 6: Guarded Colab workflow and operator guide

**Files:** Create notebook, guide and `tests/smoke/test_known_k_independent_notebook.py`.

**Interfaces:** Notebook imports Tasks 1–5 scripts from the pinned checkout and exposes exactly `PATHS`, `FROZEN`, `SEALED`, `REPORT_7026`, `REPORT_7027`, `REPORT_7028`, `SUMMARY`; no fit or automatic open.

- [ ] **Step 1: Write failing structural/smoke tests.** `test_notebook_stage_order`: inspect JSON cells and assert setup → paths → preflight → freeze → seal/STOP → typed open → three seed cells → summary. `test_no_training_or_automatic_open`: reject any fit call, old-holdout evaluation or cell joining seal to open. `test_missing_drive_path_stops`: mock missing path and assert generator not called; each mock stage emits a compact visible result.
- [ ] **Step 2: Run `pytest tests/smoke/test_known_k_independent_notebook.py -q`; expect FAIL.**
- [ ] **Step 3: Implement notebook/guide.** Resolve all Drive paths explicitly, pin the new experiment branch/revision after publication, show device/runtime and stage identity, place separate STOP text after seal and each GPU inference cell. Guide states what output the user should send before the next cell, especially before typed open; setup must not generate/open/evaluate. Do not include real result numbers before running.
- [ ] **Step 4: Run `pytest tests/smoke/test_known_k_independent_notebook.py -q`; expect PASS.**
- [ ] **Step 5: Commit `docs: add guarded known-k independent Colab workflow`.**

### Task 7: End-to-end safety verification and handoff

**Files:** Extend the Task 1–6 tests as needed; modify `development_log.md` and experiment status documentation only after evidence.

**Interfaces:** No new public API; produces a reviewed, pinned branch and precise Colab instructions, not a generated real holdout.

- [ ] **Step 1: Write a failing `test_synthetic_guarded_lifecycle`** using a small fixture graph and stage roots: assert clean freeze→seal→open→three reports→summary works; changing any frozen input or stage payload raises; no real Drive paths or holdout labels are touched.
- [ ] **Step 2: Run `pytest tests/smoke/test_known_k_independent_notebook.py::test_synthetic_guarded_lifecycle -q`; expect FAIL.**
- [ ] **Step 3: Implement only the necessary boundary fix and rerun that test; expect PASS.**
- [ ] **Step 4: Run `pytest -q`, `python -m compileall scripts tests`, `git diff --check`, notebook JSON/structural tests and a read-only search proving no edits in `src/diffusion_sources/*.py`, UI, `thesis_report/` or `master`; record exact results.**
- [ ] **Step 5: Request whole-branch review; fix findings with targeted RED→GREEN tests, rerun full verification, and update `development_log.md` with the verified facts and remaining Colab-only work.**
- [ ] **Step 6: Commit/push the separate experiment branch without PR; give the user the notebook link and exact first cells. Stop after preflight/freeze or seal as specified; real independent opening remains a separate later approval.**

## Execution handoff

The design is approved, but this plan requires user review and choice of execution method before implementation. Recommend **Native** because the stage interfaces are tightly coupled and the expensive/irreversible Colab steps remain outside local execution; request one independent whole-branch review before publication.
