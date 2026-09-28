# Temporal-v3 Independent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently evaluate the already frozen temporal correction against snapshot and early-only controls on 1998 new cascades without training or consuming the snapshot holdout.

**Architecture:** New modules add an authenticated freeze → protected generation/seal → explicit open → three frozen inference stages → paired summary workflow. Existing historical generation/replay/scoring modules remain unchanged so previous source hashes still authenticate. All artifacts are append-only, and notebook calls run directly in the kernel.

**Tech Stack:** Existing Python, NumPy, NetworkX, PyTorch/PyG, PyYAML, pytest and notebook JSON; no new product dependencies. Windows local tests, Colab CUDA inference.

**Spec:** `docs/superpowers/specs/2026-09-28-temporal-v3-independent-evaluation-design.md` (approved 2026-09-28).

## Global Constraints

- Dataset seed=4007026, единственный split independent_holdout=1998.
- S1b checkpoints seeds 7026/7027/7028, beta=0.5, t1=1, max_steps=3.
- Simulation: source_counts=[1,2,3], probabilities=[0.01,0.02,0.03], distance_ranges=[{min:1,max:2},{min:3,max:5}].
- Observation: fractions=[1.0,0.75,0.5], false_positive_count=0, hide_source_count=0.
- min_candidates=5, max_infected_fraction=0.5, max_attempt_factor=100, distance_cache_size=512, show_progress=true.
- All possible attempt seeds: [4007026,4406625]. Check both seed fields together against train/validation/test and sealed snapshot holdout metadata, before and after generation.
- Early RNG: SeedSequence([5000000,observation_seed,1]), sorted nodes; independent-bernoulli-t1-seedsequence-v1.
- No training, retuning, checkpoint selection, old test/holdout target reads, filtering by early coverage, or silent input substitution.
- Source scores/count remain frozen; all candidate nodes remain available; early-only uses the same GCN count with analytic uniform ties.
- Bootstrap: 2000 repetitions, RNG seed=9282026, percentile 95% CI [.025,.975], true-k strata; average per-cascade deltas across seeds before aggregate bootstrap.
- Primary: aggregate temporal-minus-snapshot >=0.02, aggregate CI lower >0, positive delta each seed, mean k=2/3 delta >=-0.02, count unchanged per example.
- Secondary: only after primary pass; aggregate temporal-minus-early CI lower >0 and positive delta each seed.
- Old modules/train pipeline and pilot notebook unchanged. New output suffix never permits retuning after opening.
- Drive root `/content/drive/MyDrive/diffusion-sources`; new data `data/generated/facebook_temporal_v3_independent_holdout`; reports `reports/runs/temporal_v3_independent_evaluation/v1`.
- Reference data paths and frozen run paths are exactly those in the spec; missing paths stop, no guessed fallback.
- Existing Colab reference is `data/facebook_main`, pilot `reports/runs/temporal_v3_paired_pilot/v1`, baseline `reports/runs/temporal_v3_early_baseline/v1`. Expected snapshot metadata path `data/facebook_snapshot_final_holdout` must be checked; ask user for actual location if absent.
- New data generation and evaluation are not run by the coding agent; the user separately launches numbered Colab cells. Stop before open until the user approves evaluation.

## Review Focus

1. Existing output or partial generation after disconnect: refuse overwrite and retain evidence, never generate another seed (Task 2).
2. Graphs with matching sizes but different edges or node mapping: reject before inference, compare fingerprints not NPZ timestamp bytes (Tasks 1–2).
3. Missing cache versus cache appearing after freeze: absence is identity, read-only in-memory mode must not silently switch (Tasks 1 and 3).
4. Restarted notebook or git pull with cached module functions: refresh dependency order or require fresh runtime; opening is never implied by setup (Task 4).
5. Same mean F1 with reordered rows/changed early masks across seed reports: reject instead of pairing unrelated cascades (Tasks 3–4).

## File map and shared interfaces

Create these product files only during implementation:

- `temporal_independent_artifacts.py`: paths, frozen identities, metadata reads, authorization and artifact verification.
- `temporal_independent_generation.py`: exact config, safe generation and dataset seal.
- `temporal_independent_inference.py`: explicit open, guarded independent loader, per-seed inference/metrics.
- `temporal_independent_summary.py`: row pairing, aggregate CIs and fixed gates.
- `configs/facebook_temporal_v3_independent_holdout.yaml` and `notebooks/colab_temporal_v3_independent_evaluation.ipynb`.
- All Python product modules above live in `src/diffusion_sources/`.
- Tests: matching `tests/unit/test_temporal_independent_*.py`, shared fixture file `tests/fixtures/temporal_independent.py`, notebook test `tests/smoke/test_temporal_independent_notebook.py`, pipeline test `tests/smoke/test_temporal_independent_flow.py`.
- Create empty `tests/__init__.py` and `tests/fixtures/__init__.py` so shared test-helper imports resolve to this repository rather than another installed tests package; verify collection of all existing tests remains unchanged.
- Status: `docs/temporal_v3_independent_evaluation_status.md`; append journal in original checkout without overwriting existing changes.

Paths value object, owned by Task 1:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class IndependentPaths:
    repo: Path
    reference: Path
    snapshot_holdout: Path
    pilot: Path
    baseline: Path
    runs: dict[int, Path]
    data: Path
    reports: Path
    generation_config: Path
```

Public APIs and ownership:

```python
# temporal_independent_artifacts (Task 1)
# metadata_seed_union(paths: list[Path]) -> set[int]
# assert_disjoint_paths(outputs: list[Path], protected: list[Path]) -> None
# freeze_inputs(paths: IndependentPaths) -> dict
# verify_freeze(paths: IndependentPaths) -> dict
# verify_seal(paths: IndependentPaths) -> dict
# verify_opened(paths: IndependentPaths) -> dict

# temporal_independent_generation (Task 2)
# validate_generation_config(config: dict, repo: Path) -> dict
# generate_and_seal(paths: IndependentPaths) -> dict

# temporal_independent_inference (Task 3)
# open_evaluation(paths: IndependentPaths, confirmation: str) -> dict
# collect_independent_records(paths: IndependentPaths, seed: int,
#                             device: torch.device) -> list[PilotRecord]
# evaluate_seed(paths: IndependentPaths, seed: int,
#               device: torch.device) -> dict

# temporal_independent_summary (Task 4)
# summarize_reports(reports: list[dict]) -> dict
# save_summary(paths: IndependentPaths) -> dict
```

Artifacts use existing `write_stage(root,stage,manifest,payload)` and
`read_stage(root,stage,expected_identity)`. Stages: `freeze`, `seal`, `opened`,
`seed_7026`, `seed_7027`, `seed_7028`, `summary`. All identity builders reconstruct
current hashes, never simply trust identity read from the same new manifest.
Historical stages may use authenticated historical identity via `_old_stage`;
compare that identity against current real inputs before freezing.

### Task 1: Authenticate frozen inputs and metadata-only state

**Files:** Create artifacts module, its unit tests, and shared test fixture file.
**Consumes:** existing `_old_stage`, `_identity`, `read_stage`, `write_stage`, `sha256_file`, `SnapshotFeatureBuilder.graph_fingerprint`.
**Produces:** paths object and six APIs listed above. `verify_seal`/`verify_opened`
verify identities and file hashes only; writing those stages belongs to Tasks 2/3.

- [ ] Step 1: Add failing metadata/path tests with actual temporary NPZs.

```python
import numpy as np
import pytest

def test_metadata_only(tmp_path, monkeypatch):
    from diffusion_sources.temporal_independent_artifacts import metadata_seed_union
    p = tmp_path / 'reference.npz'
    np.savez(p, simulation_seeds=[10, 12], observation_seeds=[11, 13],
             source_labels=np.ones((2, 4)), source_counts=[1, 1])
    real_get = np.lib.npyio.NpzFile.__getitem__
    reads = []
    def checked_get(archive, name):
        assert name in ('simulation_seeds', 'observation_seeds')
        reads.append(name)
        return real_get(archive, name)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', checked_get)
    assert metadata_seed_union([p]) == {10, 11, 12, 13}
    assert reads == ['simulation_seeds', 'observation_seeds']

def test_parent_and_child_outputs_rejected(tmp_path):
    from diffusion_sources.temporal_independent_artifacts import assert_disjoint_paths
    protected = tmp_path / 'frozen'
    for out in (protected, protected / 'nested', tmp_path):
        with pytest.raises(ValueError, match='overlap'):
            assert_disjoint_paths([out], [protected])
```

- [ ] Step 2: Run `pytest tests/unit/test_temporal_independent_artifacts.py -q`;
  confirm RED on missing new APIs, not a dependency/encoding failure.
- [ ] Step 3: Implement metadata reader/path guard. Seed arrays must be 1D,
  equal length, nonempty, nonnegative finite integers, unique in each field;
  verify arrays before integer casts, reject malformed/duplicate inputs.

```python
def assert_disjoint_paths(outputs, protected):
    resolved_outputs = [Path(p).resolve() for p in outputs]
    resolved_protected = [Path(p).resolve() for p in protected]
    for i, out in enumerate(resolved_outputs):
        for other in resolved_protected + resolved_outputs[:i]:
            if out == other or out in other.parents or other in out.parents:
                raise ValueError('output/input overlap')
```

- [ ] Step 4: Implement `freeze_inputs`/verification with the exact beta,
  seeds and protocol constants. Authenticate old select, validation, seed_7027,
  seed_7028 and all baseline stages/summary. Compare original graph/config,
  each run checkpoint/config/cache hash, saved metrics, locked selection hashes,
  n=1998 and old early-mask identities. Require each run config seed to match key.
  Freeze raw graph hash, generator config hash, reference NPZ hashes, all source
  module hashes, git SHA and versions. Never load checkpoint weights/new targets
  in this task. Reconstruct expected identity on resume; hash JSON with sorted
  keys and `allow_nan=False`. Include cache absence explicitly.
- [ ] Step 5: In `tests/fixtures/temporal_independent.py` define
  `make_case(tmp_path, monkeypatch) -> IndependentPaths` using real temporary
  files and real append-only stages. Three runs have configs with exact seed
  keys, checkpoint bytes, metrics files; old report identities use current
  historical source hashes and matching actual file hashes, with historical SHA.
  Reference archives contain only seed arrays. Raw graph is a ten-node path
  edge list; actual graph archives use the repository topology schema. Config
  has the exact independent generation parameters. Old report and baseline
  summaries may have controlled numeric values, but their saved hashes must
  match; do not mock new verification or persistence. Tests may stub generation
  and inference boundaries later, not identity checks.
- [ ] Step 6: Add parameterized mutation tests for run seed/checkpoint/config,
  locked beta, historical code digest, raw graph fingerprint, missing files,
  reference metadata corruption, cache contents/absence, input/output alias
  (including relative path), changed source code and partial freeze artifact.
  Each test asserts no new stage and no target-array access. Preserve original
  files' hashes. Freeze twice must read identical stage without overwrite.
- [ ] Step 7: Run artifacts tests GREEN and original pilot/baseline tests.
  Commit only Task 1 module/tests/fixture: `feat: authenticate independent evaluation inputs`.

### Task 2: Exact new generation and authenticated seal

**Files:** Create generation module/config/tests; expand shared fixture.
**Consumes:** Task 1 paths, `verify_freeze`, metadata seeds and path guard;
existing `generate_dataset`, topology reader and stage primitives.
**Produces:** `validate_generation_config` and `generate_and_seal` APIs;
immutable data directory and authenticated `seal` stage.

- [ ] Step 1: Add RED tests for exact config and forbidden regeneration:

```python
import copy
import pytest
import yaml

def test_exact_seed_and_split(tmp_path):
    from pathlib import Path
    from diffusion_sources.temporal_independent_generation import validate_generation_config
    root = Path(__file__).parents[2]
    cfg = yaml.safe_load((root / 'configs/facebook_temporal_v3_independent_holdout.yaml').read_text())
    assert validate_generation_config(cfg, root)['dataset']['seed'] == 4007026
    for bad in (4007027, 7026):
        altered = copy.deepcopy(cfg)
        altered['dataset']['seed'] = bad
        with pytest.raises(ValueError, match='protocol'):
            validate_generation_config(altered, root)
    altered = copy.deepcopy(cfg)
    altered['dataset']['splits'] = {'validation': 1998}
    with pytest.raises(ValueError, match='protocol'):
        validate_generation_config(altered, root)
```

- [ ] Step 2: Run generation tests RED. Create YAML with all exact values from
  Global Constraints, graph id/kind/path from spec, no alternate splits or
  hidden sources. Validate protocol semantically, resolving only raw graph path
  against repo root; reject unknown behavior-changing options rather than
  ignoring them. Read actual old generation config and confirm simulation,
  observation and acceptance conditions match.
- [ ] Step 3: Before calling generator, verify freeze, check full attempt-window
  overlap, output disjointness and absence of final/partial directories. Create
  temp generation directory under new data parent (not protected old data).
  Call original generator exactly once. Detect partial directory on retry and
  stop; do not overwrite, remove or silently regenerate it. On success compare
  graph fingerprint/node mapping and read only new seed metadata plus features
  schema header/shape for count=1998. Validate combined seed uniqueness/isolation.
- [ ] Step 4: Publish complete generated directory by rename; write `seal`
  stage referencing SHA256 of graph.npz, independent_holdout.npz, config.yaml,
  generation_summary.json and freeze payload. Store metadata seed-array digests,
  example count and role `sealed_unopened`. Do not call old seal function, which
  reads target counts/labels. Crash after publish but before seal must stop on
  next invocation, not treat the unsealed directory as reusable.

```python
def possible_seed_window(config):
    base = config['dataset']['seed']
    attempts = config['dataset']['splits']['independent_holdout'] * config['dataset']['max_attempt_factor']
    return base, base + 2 * attempts - 1

# Before invoking generate_dataset:
lo, hi = possible_seed_window(config)
if any(lo <= seed <= hi for seed in reference_seeds):
    raise ValueError('reference seed overlap with generation window')
```

- [ ] Step 5: With Task 1 fixture and a boundary replacement for original
  generator, write a synthetic 1998-row archive and real graph/config/summary
  files using matching schemas, generated seeds base+2*i/base+2*i+1.
  Instrument all NPZ gets: seal/freeze must never access source_counts/labels,
  infected masks or early masks. Generator boundary may write targets; the
  access prohibition applies to consumers after generation, not generation.
  Assert one generator call, successful seal and hash-checked resume without
  calling generator again. Parameterize size, graph, seeds, config tampering,
  reference interval collision, partial directories, existing unsealed data,
  rename/persist interruptions and nested output path. Failed seal publishes
  no usable stage and never computes F1.
- [ ] Step 6: Run Task 1+2 tests GREEN and old generation/holdout tests.
  Commit Task 2 files: `feat: generate and seal fixed independent cascades`.

### Task 3: Explicit opening and guarded frozen inference

**Files:** Create inference module/tests and synthetic flow test.
**Consumes:** Task 1 verifiers; Task 2 sealed data; existing `PilotRecord`,
`CandidateScores`, `replay_early_mask`, `correct_sources`, `_readonly_feature_builder`,
`predict_joint`, `JointSourceCountGCN`, `load_pyg_split`, `evaluate_pairs`,
`evaluate_early_baseline`, `set_metrics`, stage primitives.
**Produces:** open, collector and per-seed evaluator APIs; full per-cascade rows.

- [ ] Step 1: Add RED no-open test that spies on first target access:

```python
def test_no_open_no_inference(tmp_path, monkeypatch):
    import pytest
    import torch
    from tests.fixtures.temporal_independent import make_case
    from diffusion_sources.temporal_independent_artifacts import freeze_inputs
    from diffusion_sources.temporal_independent_inference import evaluate_seed
    paths = make_case(tmp_path, monkeypatch)
    freeze_inputs(paths)
    with pytest.raises(ValueError, match='sealed|opened'):
        evaluate_seed(paths, 7026, torch.device('cpu'))
    assert not (paths.reports / 'opened').exists()
    assert not (paths.reports / 'seed_7026').exists()
```

- [ ] Step 2: Run inference test RED. Implement `open_evaluation`: require
  exact confirmation string `OPEN_INDEPENDENT_HOLDOUT`; verify freeze and seal;
  write/read `opened` stage BEFORE any target-array get. Marker contains both
  authenticated payload hashes and current identities. Wrong string/mismatch
  never opens. Resume opened marker with same identity returns saved marker;
  changed input, code, runtime or incomplete marker stops.
- [ ] Step 3: Implement collector using current historical collect_records
  loading/inference steps, but a NEW authorized loader. It accepts no arbitrary
  split/indices/limit; loads only independent_holdout after `verify_opened` and
  rechecks full size/index order. Use original generation config for replay,
  frozen run config/features, read-only feature builder, `weights_only=True`,
  model.eval(), torch.inference_mode(), clone().to(device), real checkpoint.
  Reuse replay and source-blind sampling unchanged. Stop on mismatch, do not
  skip. History module source bytes stay unchanged. Before model loading validate
  seed, shared source head, input dimension, CUDA availability and model config.
- [ ] Step 4: Implement evaluator: verify open/current hashes even on resume;
  collect all records, assert beta=0 control identity and exact count/set-size
  invariance. Calculate existing paired geometry report and expected early F1
  report at beta=.5; merge rows by matching index, true-k, candidate count and
  per-cascade F1. Save full report with identities, early-mask hash, protocol,
  versions/device and evaluation_role='independent_confirmation'. Old helper
  reports' exploratory flags describe their origin, not final evaluation role;
  replace envelope interpretation explicitly without changing helper code.
  Print JSON without rows and saved paths directly to kernel.

```python
mask_rows = [(r.index, r.candidates.candidate_ids,
              r.candidates.early_observed, r.early_empty) for r in records]
early_mask_hash = hashlib.sha256(json.dumps(mask_rows).encode()).hexdigest()
assert [r.index for r in records] == list(range(1998))
# Each persisted row carries fields needed for correct cross-seed pairing:
# index, k, candidate_count, true_sources, early_count, early_expected_f1,
# snapshot_f1, temporal_f1, temporal_minus_snapshot, temporal_minus_early,
# snapshot_sources, temporal_sources, snapshot/temporal metric dictionaries.
```

- [ ] Step 5: In synthetic flow test define controlled inference records:

```python
from diffusion_sources.temporal_scoring import CandidateScores, PilotRecord

def controlled_records():
    records = []
    for i in range(1998):
        k = i % 3 + 1
        truth = frozenset(range(k))
        baseline = frozenset(range(10 - k, 10))
        scores = tuple(.6 if node in truth else .8 if node in baseline else .1 for node in range(10))
        early = tuple(node <= k for node in range(10))
        c = CandidateScores(tuple(range(10)), scores, early, baseline, k)
        records.append(PilotRecord(i, truth, c, False))
    return records
```

  Stub only collector inference boundary for full freeze→seal→open→three seed
  evaluation flow; use real scoring/metrics/persistence. Assert perfect temporal
  F1, nonperfect early baseline, zero snapshot F1, count invariant, marker before
  first target read, old file hashes unchanged, and no optimizer/train call.
  Separately exercise the real new collector with actual PyG/model weights and
  replay boundary on synthetic archives (no Drive/real independent targets).
  Confirm model outputs use sources-free scoring inputs and empty-early cases.
- [ ] Step 6: Parameterize interrupted collection, unchanged resume without
  inference, tampered checkpoint/cache/data/config after opening, wrong seed,
  record duplicates/reordering, mismatched masks, malformed features and missing
  CUDA. Require no completed seed artifact on failure; opened marker remains
  evidence of the attempt. Old collect_records still rejects test/holdout/new split.
- [ ] Step 7: Run Tasks 1–3 tests GREEN plus old temporal scoring/replay/stage
  tests. Commit Task 3 files: `feat: guard independent frozen inference`.

### Task 4: Paired summary, fixed gates and user notebook

**Files:** Create summary module/tests, new notebook/notebook tests, status doc.
**Consumes:** Task 3 reports; stage verifiers; `paired_bootstrap_ci`.
**Produces:** `summarize_reports`, `save_summary` and numbered Colab workflow.

- [ ] Step 1: Write pure summary RED tests with no file fixture:

```python
def synthetic_reports():
    reports = []
    for seed in (7026, 7027, 7028):
        rows = []
        for i in range(1998):
            rows.append({'index': i, 'k': i % 3 + 1, 'candidate_count': 10,
                         'true_sources': list(range(i % 3 + 1)),
                         'snapshot_f1': .3, 'early_expected_f1': .4,
                         'temporal_f1': .5, 'temporal_minus_snapshot': .2,
                         'temporal_minus_early': .1,
                         'snapshot': {'precision': .3, 'recall': .3, 'f1': .3,
                                      'exact_set_accuracy': 0., 'count_accuracy': 1.,
                                      'count_mae': 0., 'symmetric_set_distance': 1.,
                                      'hit_at_1_hop': .6, 'hit_at_2_hop': .9},
                         'temporal': {'precision': .5, 'recall': .5, 'f1': .5,
                                      'exact_set_accuracy': .2, 'count_accuracy': 1.,
                                      'count_mae': 0., 'symmetric_set_distance': .5,
                                      'hit_at_1_hop': .8, 'hit_at_2_hop': .95}})
        reports.append({'seed': seed, 'beta': .5, 'n': 1998,
                        'early_mask_hash': 'same', 'dataset_hash': 'same',
                        'freeze_hash': 'same', 'count_unchanged': True,
                        'rows': rows})
    return reports

def test_aggregate_cascade_ci():
    import pytest
    from diffusion_sources.temporal_independent_summary import summarize_reports
    result = summarize_reports(synthetic_reports())
    assert result['bootstrap_n'] == 1998
    assert result['primary']['delta_f1'] == pytest.approx(.2)
    assert result['primary']['ci'] == pytest.approx([.2, .2])
    assert result['primary']['passed'] is True
    assert result['secondary']['passed'] is True
```

- [ ] Step 2: Run summary tests RED. Validate exactly three distinct seeds,
  matching beta/dataset/freeze/masks, aligned index/k/true_sources/candidate_count,
  finite metrics in [0,1], count invariance and complete rows. Reject summaries
  whose reported deltas disagree with row metrics. Preserve seed reports,
  absolute means/sample SD and geometry aggregates without rounding before CI.

```python
seeds = (7026, 7027, 7028)
by_seed = {r['seed']: r for r in reports}
matrix = np.array([[row['temporal_minus_snapshot'] for row in by_seed[s]['rows']]
                   for s in seeds], dtype=float)
mean_deltas = matrix.mean(axis=0)  # shape (1998,), NOT (5994,)
true_k = np.array([row['k'] for row in by_seed[7026]['rows']])
ci = paired_bootstrap_ci(mean_deltas, true_k, repetitions=2000, seed=9282026)
```

- [ ] Step 3: Implement reasons-list gates exactly from Global Constraints;
  secondary must not pass when primary fails. Build both aggregate comparisons
  and per-seed CIs, absolute mean/sample SD metrics, k/candidate-size tables,
  coverage and conditional-CI interpretation. `save_summary` reauthenticates
  opened/seal/freeze plus seed stages and reconstructs source payload identities;
  compute from saved reports, not mutable notebook memory. Persist/read summary
  append-only. Test negative mean/CI, nonpositive seed, k loss, count mismatch,
  shuffled rows with same means, changed masks, duplicate seeds, nonfinite F1,
  missing seed stage, corrupted CSV/payload and resumed summary identity.
- [ ] Step 4: Add asymmetric seed-by-cascade synthetic differences to test that
  aggregate CI equals bootstrap of row-wise averaged deltas rather than a
  flattened 5994 array or average per-seed CI. Use actual `paired_bootstrap_ci`
  and assert exact deterministic values for repeated calls; do not mock CI.
- [ ] Step 5: Create notebook cells with these IDs and operations:

  1. `setup`: mount Drive, git pull/install into kernel Python, restart or reload
     dependencies before the new modules, print import path/device/version.
  2. `freeze`: instantiate paths, invoke freeze_inputs and print identities.
  3. `seal`: invoke generate_and_seal, print sealed manifest; no inference.
  4. `open_7026`: interactive confirmation, open_evaluation then evaluate_seed.
  5. `seed_7027`: only evaluate_seed; requires opened marker, no retuning.
  6. `seed_7028`: only evaluate_seed.
  7. `summary`: save_summary and print compact tables/gates/paths.

```python
# Cell open_7026: deliberately requires a user action after the seal output.
answer = input('Введи OPEN_INDEPENDENT_HOLDOUT для независимой оценки без подбора: ').strip()
open_evaluation(PATHS, answer)
REPORT_7026 = evaluate_seed(PATHS, 7026, torch.device(DEVICE))
```

  State explicitly: do not Run all; stop and send freeze/seal outputs before
  opening; old pilot notebook stages are not rerun. Paths/seed/config are printed
  before generation; no nested subprocess for inference. No guarantee of runtime.
  If new files updated in an already imported runtime, invalidate/reload in
  artifacts→generation→inference→summary dependency order (historical modules
  must remain byte-identical); remove stale imported function bindings.
- [ ] Step 6: Execute notebook cells under tests with Drive/subprocess/input
  boundaries replaced. Assert setup never freezes/generates/opens/infers;
  freeze never opens; seal never infers; wrong confirmation never opens; each
  seed cell routes exactly one seed; summary reads disk stages. Verify visible
  kernel output, cached-module reload and restart/resume independent of memory.
- [ ] Step 7: Add integration test from Task 3 continuing through actual summary
  persistence, with successful and failed gates. Verify no old file changed,
  input identities include every consumed artifact and new data are not loaded
  by pre-open test code except producer writing/generation. Failure stops before
  incomplete subsequent stages and output is never overwritten.
- [ ] Step 8: Run full `pytest -q`, compileall and diff check on feature branch;
  inspect actual output before claims. Add status doc and journal entry with
  measured local checks and explicit absence of real Colab results. Request one
  independent whole-branch review if Native execution is chosen; fix findings
  with regression tests, rerun full suite. Commit Task 4 files:
  `feat: report independent paired evaluation in Colab`.

## Execution environment and completion

Implementation starts only after plan approval and execution-method choice.
Use using-git-worktrees at execution time; current docs are in the main checkout.
Branch proposal: `codex/temporal-independent-evaluation`, isolated worktree.
Keep existing user changes; stage only task files. Historical source module hash
diffs must be empty. Local Python runtime is the existing main repo `.venv`;
in a worktree set PYTHONPATH to that worktree's `src`:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
& 'D:/projects/diplom/diffusion-sources-localization/.venv/Scripts/python.exe' -m pytest -q
& 'D:/projects/diplom/diffusion-sources-localization/.venv/Scripts/python.exe' -m compileall -q src scripts
git diff --check
```

Windows temp/cache tests may require sandbox escalation; use the tool approval
mechanism, not arbitrary dependency installs. No real independent data generation
or target metrics during implementation. Only after reviewed green branch and
user-authorized merge/push can user run the new notebook. Preserve old manifests
and record publication commit. Provide one cell at a time, first setup, then
freeze, then seal; wait for outputs and separate explicit permission before open.

## Plan self-review and user handoff

Coverage: fixed methods/protocol → Tasks 1/3; seed isolation/seal → Task 2;
authorization/replay/identity → Tasks 1/3; statistics/gates/interpretation → Task 4;
notebook/visibility/reload → Task 4; preservation/resume → all tasks. Public API
names above match every task. Review Focus checks assigned to owning tasks.
No product code or independent results are created by writing this plan.

Recommended execution: Native, because all four stages depend on the previous
stage identities and one implementer can preserve those interfaces; one fresh
whole-branch review follows. Subagent-driven is also available with separate
implementer/reviewer contexts per task, at higher coordination cost. User reviews
this plan and chooses before implementation. Merge/push remains a separate choice.
