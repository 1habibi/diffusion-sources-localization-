# Temporal-v3 Streamlit Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pilot-only Streamlit screen with a reproducible, local Temporal-v3 demonstration on simulated Facebook cascades.

**Architecture:** A read-only artifact loader restores frozen S1b; a scenario service generates one accepted IC cascade; an inference service returns paired Snapshot/Temporal predictions without labels in model input. A separate view builder caps only the displayed graph; `app.py` owns Streamlit controls and session state.

**Tech Stack:** Python 3.10+, Streamlit >=1.49, PyTorch/PyG, NetworkX, NumPy, Plotly >=6.0 for node selection, pytest/AppTest.

**Spec:** `docs/superpowers/specs/2026-09-30-temporal-v3-streamlit-demo-design.md`

## Global Constraints

- Use exactly one fixed Facebook topology, the read-only local backup `reports/backups/temporal_v3_20260929`, frozen S1b seed 7026, `beta=0.5`, `t1=1`, `t3=3`; no training, tuning, artifact overwrite, or holdout/test reading.
- Controls: true simulation `k` in 1..3, `p` in {0.01, 0.02, 0.03}, observation fraction in {0.50, 0.75, 1.00}, nonnegative integer seed. Reference acceptance: `min_candidates=5`, `max_infected_fraction=0.5`, at most 100 deterministic attempts; false positives 0.
- True source labels and source count must be removed from PyG input before inference. S1b predicts count once; correction preserves count and candidate set. Raw GCN score is not a calibrated probability.
- Run inference on all 4,039 nodes; visual capping never changes model input. Snapshot/Temporal, t1/t3/score, source-visibility and selected-node controls reuse the current successful scenario.
- Existing modified `development_log.md` and `docs/temporal_learned_reranker_status.md` are user work; preserve them. Keep development notes out of `thesis_report/`.

## Review Focus

- Missing/corrupted backup or mismatched SHA256: show a specific path and fail closed; test in Task 1.
- Small cascades, including `k=1,p=0.01`: retry deterministically and report exhaustion without treating a rejected sample as success; test in Task 2.
- Empty early observation: temporal correction remains valid and Snapshot prediction/count is unchanged; test in Task 3.
- More important nodes than the normal visual cap: never silently omit truth, prediction or selected ID; test in Task 4.
- A failed rerun after a successful example: keep the previous result and show an error; test in Task 5.

## File Map

- Create `src/diffusion_sources/temporal_demo_assets.py`: backup path resolution, integrity checks, frozen config/model/graph/feature builder loading.
- Create `src/diffusion_sources/temporal_demo_scenario.py`: validated settings and deterministic source sampling, IC cascade, accepted final/early observations.
- Create `src/diffusion_sources/temporal_demo_inference.py`: target-free S1b inference, paired prediction and per-case metrics.
- Create `src/diffusion_sources/temporal_demo_view.py`: bounded, stable-position graph view, Plotly marks and node IDs for selection.
- Modify `app.py`: Russian Streamlit workbench, cached resources, session result, Plotly selection and tables. Modify `pyproject.toml` for Plotly and `.streamlit/config.toml` for restrained theme if useful; keep CSS minimal.
- Create corresponding unit tests under `tests/unit/`; replace the obsolete pilot-specific expectation in `tests/smoke/test_streamlit_app.py` without changing legacy `demo.py` behavior.

### Task 1: Frozen local resources

**Files:** Create `src/diffusion_sources/temporal_demo_assets.py`; test `tests/unit/test_temporal_demo_assets.py`.

**Interfaces:** Produce `DemoResources(graph: nx.Graph, model: JointSourceCountGCN, builder: SnapshotFeatureBuilder, generation_config: dict, model_config: dict)` and `load_temporal_resources(backup_root: Path) -> DemoResources`. The root defaults in `app.py` to the repository backup, with an optional environment override to an equivalent backup directory.

- [ ] **Step 1: Write failing tests** `test_load_frozen_s1b_cpu` (`assert graph.number_of_nodes() == 4039`, `assert not model.training`, `assert parameter_count == 21252`), `test_missing_checkpoint_names_path` (`pytest.raises(FileNotFoundError, match="best_model.pt")`), `test_manifest_hash_mismatch_rejected` (`pytest.raises(ValueError, match="SHA256")`), `test_wrong_graph_or_model_config_rejected` (`pytest.raises(ValueError)`). Use a minimal manifest fixture/monkeypatch for failures and the real local backup for the smoke assertion.
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/unit/test_temporal_demo_assets.py -q`; expect failing imports/assertions.
- [ ] **Step 3: Implement** `load_temporal_resources`; resolve `local_paths.json`, check required entries against `backup_manifest.json`, require 4,039-node/88,234-edge graph, reconstruct all frozen model options, use `torch.load(..., map_location="cpu", weights_only=True)`, `eval`, and in-memory distance cache fallback. Never load a backup split archive.
- [ ] **Step 4: Run** the same test file; expect PASS.
- [ ] **Step 5: Commit** only Task 1 code/tests: `feat: load frozen Temporal-v3 demo resources`.

### Task 2: Deterministic accepted scenario

**Files:** Create `src/diffusion_sources/temporal_demo_scenario.py`; test `tests/unit/test_temporal_demo_scenario.py`.

**Interfaces:** Consume `DemoResources`. Produce `DemoSettings(true_k: int, probability: float, observation_fraction: float, seed: int)`, `DemoScenario(cascade: Cascade, final: Observation, early_nodes: frozenset[int], simulation_seed: int, observation_seed: int, attempt: int)` and `generate_demo_scenario(resources: DemoResources, settings: DemoSettings) -> DemoScenario`.

- [ ] **Step 1: Write failing tests** `test_same_settings_replay_same_sources_masks` (equal sources, final observed, early nodes and attempt twice), `test_reject_invalid_controls` (`pytest.raises(ValueError)` for each out-of-range value), `test_reject_then_accept_is_deterministic` (`attempt == 2` with a stub rejecting attempt 1), `test_exhaustion_is_explicit` (100 rejected attempts raise a named error), `test_early_sampling_does_not_force_sources` (stub/sample with empty early nodes stays empty). Include `k=1,p=.01` and exact control grids.
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/unit/test_temporal_demo_scenario.py -q`; expect RED.
- [ ] **Step 3: Implement** `generate_demo_scenario` using `SourceSampler`, `simulate_ic`, `observe_cascade`, and `sample_early_nodes`. Derive integer seeds with `np.random.SeedSequence([0xD3A0, user_seed, attempt, role]).generate_state(1, dtype=np.uint32)[0]`, with role 0 for simulation and 1 for observation; choose distance range by `(user_seed + attempt) % 2`. Accept only candidate count >=5 and infected fraction <=.5; never read a frozen split.
- [ ] **Step 4: Run** the same test file; expect PASS.
- [ ] **Step 5: Commit** only Task 2 code/tests: `feat: generate reproducible demo cascades`.

### Task 3: Paired frozen predictions

**Files:** Create `src/diffusion_sources/temporal_demo_inference.py`; test `tests/unit/test_temporal_demo_inference.py`.

**Interfaces:** Consume `DemoResources` and `DemoScenario`. Produce `TemporalDemoResult(scenario: DemoScenario, snapshot: SourcePrediction, temporal_sources: frozenset[int], candidate_scores: CandidateScores, snapshot_metrics: dict[str,float], temporal_metrics: dict[str,float])` and `infer_demo(resources: DemoResources, scenario: DemoScenario) -> TemporalDemoResult`.

- [ ] **Step 1: Write failing tests** `test_targets_removed_before_model_call` (fake model asserts no `source_labels`/`source_count` attributes), `test_beta_zero_equals_snapshot` (`correct_sources(candidate_scores, 0) == snapshot.sources`), `test_temporal_keeps_estimated_count_and_candidates` (`len(temporal_sources) == snapshot.source_count` and subset of candidate IDs), `test_empty_early_equals_snapshot` (equal source sets), `test_metrics_match_source_sets` (exact F1/exact/distance on a tiny graph). Fake model returns controlled logits/counts.
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/unit/test_temporal_demo_inference.py -q`; expect RED.
- [ ] **Step 3: Implement** `infer_demo`: build the five configured features with `resources.builder.build(observed_mask, feature_names, candidate_mask=candidate_mask)`, construct full-graph PyG `Data(x, edge_index, candidate_mask, observed_mask)` directly **without target attributes**, call frozen model once under `torch.inference_mode`, run `predict_joint`, create `CandidateScores` with early mask, run `correct_sources(...,.5)`, and compute `set_metrics` plus symmetric set distance independently for each answer. Do not mutate the frozen resource or candidate mask.
- [ ] **Step 4: Run** the same test file and one real-backup CPU example; expect PASS and no artifact hash changes.
- [ ] **Step 5: Commit** only Task 3 code/tests: `feat: pair snapshot and Temporal-v3 demo inference`.

### Task 4: Bounded interactive graph view

**Files:** Create `src/diffusion_sources/temporal_demo_view.py`; test `tests/unit/test_temporal_demo_view.py`; modify `pyproject.toml` for `plotly>=6.0`.

**Interfaces:** Consume `TemporalDemoResult`. Produce `build_demo_graph_view(result: TemporalDemoResult, selected_node: int | None, max_nodes: int = 300) -> DemoGraphView` with visible node IDs, induced edges, stable positions and displayed/total counts; `plot_demo_graph(view: DemoGraphView, result: TemporalDemoResult, method: str, frame: str, show_truth: bool) -> plotly.graph_objects.Figure`. The node trace's `customdata` carries original graph IDs for `st.plotly_chart(..., on_select="rerun")`.

- [ ] **Step 1: Write failing tests** `test_full_graph_not_modified_by_view` (`len(resources.graph) == before`), `test_truth_predictions_selection_survive_cap` (`mandatory_ids <= set(view.node_ids)` even when cap is smaller), `test_positions_identical_across_methods_and_frames` (`view.positions` equal), `test_plot_marks_match_t1_t3_truth_and_prediction` (different observed marks; same node coordinates), `test_plot_selected_point_maps_to_original_id` (`customdata[point_index] == original_id`).
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/unit/test_temporal_demo_view.py -q`; expect RED.
- [ ] **Step 3: Implement** deterministic graph selection: mandatory `truth ∪ snapshot ∪ temporal ∪ selected`; fill remaining budget with observed nodes and then nearest neighbors in sorted order, increasing the effective cap if mandatory count exceeds it. Compute layout once per result; frames change only marks. Use separate shape/outline and color for truth/prediction, accessible hover ID/status, no invented confidence. Keep figure legible on narrow widths.
- [ ] **Step 4: Install project dependencies in the isolated execution environment**, then rerun this test file; expect PASS. Inspect one rendered wide and narrow figure.
- [ ] **Step 5: Commit** Task 4 code/tests/dependency: `feat: display selectable Temporal-v3 cascade graph`.

### Task 5: Streamlit screen and integration

**Files:** Modify `app.py`, `tests/smoke/test_streamlit_app.py`; create `tests/unit/test_temporal_demo_app.py`; optionally create `.streamlit/config.toml` for theme; update `development_log.md` **without overwriting existing edits**.

**Interfaces:** Consume Tasks 1–4. Keep `main() -> None`. `st.session_state` holds the last successful `TemporalDemoResult` and corresponding `DemoGraphView`; controls may change draft settings without replacing them. Render selected-node detail from original graph ID, not Plotly trace index.

- [ ] **Step 1: Write failing AppTest/integration tests** `test_initial_screen_has_no_fake_metrics` (zero metrics before Run), `test_success_persists_on_switch` (same scenario identity after method/frame toggle), `test_failed_new_run_keeps_previous_result` (old F1 remains and error appears), `test_missing_backup_fails_closed` (no pilot inference call), `test_predicted_k_is_not_control_value` (stub model's `k=1` displayed when simulation control is `k=2`). Isolate Streamlit cache and inject fake services.
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/unit/test_temporal_demo_app.py tests/smoke/test_streamlit_app.py -q`; expect RED.
- [ ] **Step 3: Implement** Russian Streamlit controls, one Run action, persisted result, compact metrics/comparison/candidate table, frame/method/truth controls, Plotly graph selection and source-ID detail. Expose path-specific errors without stack traces in UI; apply native Streamlit theme and restrained chrome matching the approved mockup. No real inference before the Run action.
- [ ] **Step 4: Run** the Task 5 tests and full `.venv/Scripts/python.exe -m pytest -q`; expect PASS. Launch `streamlit run app.py` with the local backup and manually verify wide/narrow layouts, different/same seeds, toggles, selected nodes, and error recovery. Verify backup hashes and `git diff --check`.
- [ ] **Step 5: Commit** only Task 5-owned code/tests/theme with `feat: ship Temporal-v3 Streamlit demo`; then append an accurate implementation/verification entry to `development_log.md` without staging or rewriting its pre-existing edits. Report the still-uncommitted log explicitly at handoff; do not push without a separate user request.

## Completion Gate

Fresh local launch produces a real cascade and paired predictions on the full frozen Facebook graph; controls and graph are usable at desktop and narrow widths; all relevant tests pass; checkpoint/data/report hashes remain unchanged. A failed smoke, UI regression, or failed hash check stops completion claims and gets diagnosed before handoff.
