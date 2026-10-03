# Known-k Temporal-GCN Fixed Repeats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely run fixed training seeds 7027 and 7028 after the passing 7026 pilot, and report stability without changing the original model code or opening independent data.

**Architecture:** A new runner under `scripts/` reuses the frozen `diffusion_sources` package but is not part of the package source hashes sealed by pilot 7026. It authenticates the old pilot and all inputs, creates separate append-only repeat stages, trains each seed under the same fixed config, then summarizes paired validation deltas. A new Colab notebook exposes one explicit GPU approval per seed.

**Tech Stack:** Python 3.10+, PyTorch/PyG, NumPy, PyYAML, pytest, Colab notebooks; existing `temporal_pilot_artifacts` SHA-256 stage helpers.

**Spec:** `docs/superpowers/specs/2026-10-03-known-k-temporal-gcn-repeats-design.md`.

## Global Constraints

- Do not modify **any** `src/diffusion_sources/*.py`; the completed 7026 pilot hashes them all. Work from a new branch/worktree based on `experiment/known-k-temporal-gcn` and prove package-source diff is empty before publishing.
- Seeds are exactly `[7027, 7028]`; seed 7026 remains the unmodified reference. Training examples = 9,990, validation examples = 1,998, same archived cascades and same fixed `configs/known_k_temporal_gcn_pilot.yaml` values: 9 input channels, hidden 64, dropout 0.2, LR 0.001, batch 3, maximum 100 epochs, patience 10, Temporal-v3 beta 0.5.
- Existing pilot root and frozen S1b are read-only. New Drive output root is `reports/runs/known_k_temporal_gcn_repeats/v1`. Never read `test.npz` or independent holdout and never write `thesis_report/`.
- Refuse seed 7028 if seed 7027 `delta_f1 <= 0`; if seed 7027 is negative, `summary` records the stopped protocol. Positive `delta_f1` for **both** repeats is required before proposing a new independent evaluation.
- Do not print all per-example rows in Colab. Each long seed run needs separate typed approval, is not triggered by setup/freeze/summary, and must stop for user review.

## Review Focus

- A `complete` marker with a modified pilot checkpoint or payload must fail pilot preflight before any repeat output is written (Task 1 test).
- An output root equal to, inside, or enclosing the pilot/data/S1b roots must be rejected before stage creation (Task 1 test).
- A partial repeat checkpoint from another seed or identity, a stale SHA sidecar, or an interrupted finalization must not silently resume or overwrite (Task 2 test).
- Reordered validation indices or changed Temporal-v3 control predictions must fail paired comparison, not yield a plausible aggregate (Task 2 test).
- Seed 7027 with non-positive delta must block seed 7028, yet still permit a truthful one-repeat failure summary; one-repeat sample SD is N/A (Task 3 test).

---

## File map

- Create `scripts/__init__.py` and `scripts/known_k_temporal_repeats.py`: importable `RepeatPaths`, read-only preflight, frozen identity, guarded seed stages, and summary API. Reuse package APIs without changing their implementation.
- Create `configs/known_k_temporal_gcn_repeats.yaml`: exact seed order and fixed stop rule, no tunable model fields.
- Create `tests/unit/test_known_k_temporal_repeats.py`: identity, training, resume, same-control and summary regressions on synthetic fixtures.
- Create `notebooks/colab_known_k_temporal_gcn_repeats.ipynb` and `tests/smoke/test_known_k_temporal_repeats_notebook.py`: staged Colab flow and approval checks.
- Create `docs/colab_known_k_temporal_gcn_repeats.md`; modify `development_log.md` only for factual local implementation status. Do not touch `thesis_report/`.

### Task 1: Authenticate the existing pilot and freeze repeat inputs

**Files:** Create runner/config and `tests/unit/test_known_k_temporal_repeats.py`.

**Interfaces:**
- `RepeatPaths(pilot: KnownKPaths, output_dir: Path, protocol_path: Path, notebook_path: Path)`; `preflight(paths: RepeatPaths) -> dict` is read-only, returning the verified 7026 payload and repeat identity.
- `run_repeat_stage(stage: Literal['freeze','seed_7027','seed_7028','summary'], paths: RepeatPaths, device: torch.device, *, resume: bool = False) -> dict` is the notebook/CLI entrypoint. Task 1 implements `freeze`; later tasks fill other stages.
- Identity includes old pilot identity and manifest/checkpoint/payload hashes; current package source hashes; runner/protocol/notebook hashes; new absolute output root. `read_stage` must verify old freeze/smoke/pilot before checking `quality_gate.passed is True`.

- [ ] **Step 1: Write failing tests.** `test_preflight_authenticates_old_pilot_without_writing`: verified fixture pilot returns 7026 gate; changed pilot checkpoint/payload, missing complete, failed gate, changed package-hash digest, or output overlap all raise before output exists. `test_freeze_refuses_changed_script_or_protocol`: repeat freeze is idempotent only for exactly matching SHA identity.
- [ ] **Step 2: Verify RED.** `python -m pytest -q tests/unit/test_known_k_temporal_repeats.py` fails for absent runner/API.
- [ ] **Step 3: Implement minimal preflight/freeze.** Use pilot `_config`/`_input_identity`, `read_stage`, `sha256_file`, and `write_stage`; validate YAML is exactly `seeds: [7027, 7028]` and `stop_on_nonpositive_delta: true`. Reject all protected-root overlaps before `write_stage`. Add `main(argv)` with stage/paths/device/resume arguments but do not train yet.
- [ ] **Step 4: Verify GREEN.** Focused tests pass; `git diff <pilot-base> -- src/diffusion_sources` is empty.
- [ ] **Step 5: Commit.** Commit only Task 1 runner/config/tests, including `scripts/__init__.py`.

### Task 2: One fixed seed with safe checkpoint and paired report

**Files:** Extend runner and unit tests.

**Interfaces:** `run_repeat_stage('seed_7027' | 'seed_7028', ...) -> dict` returns one persisted seed payload, including `seed`, `paired_report`, `delta_f1`, `best_epoch`, `fit_elapsed_seconds`, and output path. `_run_seed(paths, seed: int, device: torch.device, identity: dict, *, resume: bool) -> dict` is independently CPU-testable on fixture archives; public long-run path requires CUDA.

- [ ] **Step 1: Write failing tests.** `test_seed_7027_uses_same_train_validation_and_hyperparameters`: synthetic archive, real tiny fit, seed set to 7027, checkpoint loadable, paired rows on same indices, count accuracy N/A, beta 0.5. `test_resume_rejects_foreign_seed_and_corrupt_sha`: valid 7027 checkpoint accepted only with matching identity/seed; incomplete digest or 7028 checkpoint refused. `test_control_matches_pilot_rows`: altered control source set, candidate count, missing/duplicate/reordered index rejected before completed stage; archive hash protects candidate masks. `test_finalization_retry` with stale manifest/complete succeeds without self-hash.
- [ ] **Step 2: Verify RED.** Focused tests fail for the missing seed stage/guards.
- [ ] **Step 3: Implement seed training/evaluation.** Reuse `load_known_k_split`, `set_seed`, `NodeOnlyGCN`, `calculate_pos_weight`, `fit_node_model`, pilot `_evaluate_pilot`, `save_training_result`, and existing binary-stage/artifact helpers. Pass checkpoint metadata `{repeat_identity, seed}` and verify SHA + required resume state before unpickling. Recompute exact paired validation, require unchanged per-index truth/candidate counts/Temporal-v3 control compared with pilot 7026; persist and return the full report, but print only concise fields in notebook. Never choose between seed checkpoints using a new score.
- [ ] **Step 4: Verify GREEN.** Focused tests pass; actual local backup (if present) permits a small CPU mechanics smoke without interpreting F1. Re-run Task 1 tests.
- [ ] **Step 5: Commit.** Commit Task 2 runner/tests.

### Task 3: Precommitted stop rule and honest summary

**Files:** Extend runner and unit tests.

**Interfaces:** `summarize_repeats(pilot: dict, repeats: Mapping[int, dict]) -> dict` displays 7026 as reference and returns per-repeat ΔF1/CI/exact/group metrics, mean over completed **repeat** seeds, sample SD for two repeats (otherwise `None`), and `positive_delta_each_repeat`. `run_repeat_stage('summary', ...)` persists a SHA-checked stage after reading only verified completed seed stages.

- [ ] **Step 1: Write failing tests.** `test_negative_7027_blocks_7028_but_allows_failure_summary`, `test_positive_two_seeds_summary_has_mean_and_sample_sd`, `test_missing_7028_after_positive_7027_refuses_summary`, and `test_summary_rejects_tampered_seed_stage`; require no pooled pseudo-independent CI and `sample_sd=None` when only one repeat exists.
- [ ] **Step 2: Verify RED.** Focused summary tests fail.
- [ ] **Step 3: Implement guards and summary.** Reject seed 7028 before training when verified 7027 delta is `<=0`; summary may end after negative 7027 or after both verified repeats, never after only a positive 7027. Do not select a winning seed or open holdout. Record no `count_accuracy` achievement.
- [ ] **Step 4: Verify GREEN.** Unit tests pass, including all Task 1/2 tests.
- [ ] **Step 5: Commit.** Commit Task 3 runner/tests.

### Task 4: Colab handoff, documentation and full verification

**Files:** Create repeat notebook, notebook smoke tests, runbook; modify development log.

**Interfaces:** Notebook uses `RepeatPaths`/`run_repeat_stage`, a new clone directory `/content/diffusion-sources-known-k-repeats`, and separate Drive output root; prints only concise seed/summary fields. No existing pilot notebook edits.

- [ ] **Step 1: Write failing notebook tests.** Compile cells and assert order `setup,paths,preflight,freeze,approve_7027,seed_7027,gate_7027,approve_7028,seed_7028,summary`; without exact typed approval neither seed stage runs; 7028 is refused after non-positive 7027; setup/paths/freeze do not train or open test/holdout; output has visible STOP.
- [ ] **Step 2: Verify RED.** Notebook test fails before notebook exists.
- [ ] **Step 3: Implement notebook/runbook.** Keep setup import/version/GPU checks and Drive paths explicit. Approval phrases are exactly `RUN_KNOWN_K_REPEAT_7027` and `RUN_KNOWN_K_REPEAT_7028`, consumed before launching their seed. Read-only preflight precedes a separate freeze cell. `gate_7027` prevents 7028 if non-positive and directs to summary. Explain metrics and exploratory status in `docs/`; log implementation and local test evidence, not fictional GPU results.
- [ ] **Step 4: Verify GREEN.** Notebook tests pass; full `python -m pytest -q -p no:cacheprovider` passes; JSON/compileall and `git diff --check` pass; package source diff and `thesis_report/` diff remain empty.
- [ ] **Step 5: Commit.** Commit notebook, docs, tests and development log. Request independent whole-branch review; fix Critical/Important findings with tests. Publish **only** the repeat branch after verified tests and the user's existing no-PR preference; do not merge to `master`.

## Handoff

After publication, tell the user to open the **new** repeat notebook and run only its `setup` cell first. Do not tell them to re-run the old 7026 pilot or to launch 7027/7028 automatically. Any positive repeat summary remains exploratory until a separately planned, sealed independent evaluation.
