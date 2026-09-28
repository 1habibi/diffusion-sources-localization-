# Early-only ranking baseline: no-training analysis

## Purpose and approved protocol

2026-09-28: compare the completed temporal-v3 pilot against an early-only
ranking baseline before investing in another training run. Use the same 1998
validation cascades, frozen checkpoints 7026/7027/7028, and fixed beta=0.5.
No training, hyperparameter selection, test or final holdout access.

Compare snapshot S1b, early-first ranking without learned source scores, and
S1b plus the fixed early correction. All three retain the same frozen predicted
count. This is a ranking ablation, not a completely neural-free estimator.
Early-first ties are uniform within early and late candidate groups; compute
their analytic expected F1 rather than resolving by node IDs or learned scores.

Reports include overall F1, true-k and candidate-size strata, and a paired
true-k-stratified cascade bootstrap CI for temporal F1 minus expected early F1
(2000 repetitions, seed 9282026). This CI integrates baseline tie randomness;
it describes cascade sampling uncertainty, not variability of a randomly drawn
baseline prediction. Geometry metrics are not claimed for expected predictions.
Results remain exploratory on the existing validation split.

## Artifact protection

Old pilot stages are read-only. Before new inference, authenticate historical
stage payloads, historical source modules, current data/checkpoint/config/cache
hashes, selected beta and saved metrics. Replayed early masks, example ordering,
snapshot F1 and temporal F1 must match the completed pilot or execution stops.
New artifacts are append-only in `reports/runs/temporal_v3_early_baseline/v1`.
Changed identities require a new output suffix; no old files are overwritten.

## Colab sequence

Use the updated `notebooks/colab_temporal_v3_pilot.ipynb` after publication.
Do not repeat old stages 4–7: their strict code identity changes after adding
analysis code. If the runtime reset, first run setup cells 1–3.

8. Baseline setup: git pull and reload scoring before runner, no inference.
9. Frozen seed 7026: replay/inference only; print and save the comparison.
10. Frozen seed 7027: same protocol, no retuning.
11. Frozen seed 7028: same protocol, no retuning.
12. Print/save the three-seed mean and sample SD, individual paired CIs.

Each new analysis stage prints directly from the notebook kernel. Original
subprocess output behavior is outside this change. Runtime duration and actual
early-only F1 have not yet been measured on Colab.

## Implementation checks

Unit tests cover analytic ties (including exhaustive tiny cases), score and
node-ID invariance, count clamping, strata and bootstrap. Runner tests exercise
real persistence and scoring with controlled inference records, old artifact
immutability, resume and hash/mask/F1 mismatches. Notebook cells are executed in
tests, including cached module reload after an update.

Independent review found one important cached-import provenance issue; scoring
and runner reload in dependency order closes it, with a RED→GREEN regression.
The final review reports no remaining issues. Local checks do not substitute
for the forthcoming real frozen-checkpoint Colab run.

Final local verification: **202 passed**, two existing PyTorch deprecation
warnings, 43.84 seconds. `compileall -q src scripts` and `git diff --check`
passed. No real Colab result is claimed.

## Integration

2026-09-28: user authorized merge and push. Feature commit `54d9331`
fast-forward merged into `master` without conflicts. Publication includes the
development log and existing local experiment-document updates. Next external
step remains Colab baseline setup (cell 8), then frozen seed 7026 (cell 9),
without rerunning old stages 4–7.

Merged-master verification: 202 passed in 44.60 seconds, two existing warnings;
compileall and diff check passed.

## Reported Colab result (2026-09-28)

User-provided summary reports 1998 validation examples per frozen seed.
Drive manifests and individual payloads were not independently fetched here.
F1 mean ± sample SD: snapshot 0.358542 ± 0.004843; early-first expected
baseline 0.448700 ± 0.004556; temporal 0.538589 ± 0.001046.
Mean temporal-minus-early = +0.089888 (sample SD 0.004528).
7026/7027/7028 paired deltas = +0.084985/+0.090766/+0.093914;
bootstrap 95% CIs = [0.073397,0.097583]/[0.078836,0.102776]/
[0.081285,0.106080]. Both all-seed positivity flags are true.

This supports learned ranking relative to the specified early-first control,
not every possible heuristic. The control shares learned count. All seeds
use the same exploratory validation split; CIs are conditional on frozen
models and beta, with ties integrated. No new training or final test/holdout
evaluation is reported. Next inexpensive step: read saved true-k and
candidate-size strata before considering additional training.

### Reported marginal strata

User supplied the read-only table output, averaged over the same three seeds:

| Group | n per seed | Snapshot | Early expected | Temporal | Temporal minus early |
|---|---:|---:|---:|---:|---:|
| k=1 | 666 | 0.132549 | 0.377602 | 0.411662 | +0.034060 |
| k=2 | 666 | 0.421705 | 0.471204 | 0.577461 | +0.106257 |
| k=3 | 666 | 0.521371 | 0.497295 | 0.626643 | +0.129348 |
| 1–10 candidates | 845 | 0.442209 | 0.512820 | 0.601275 | +0.088456 |
| 11–20 candidates | 516 | 0.346770 | 0.453110 | 0.544208 | +0.091098 |
| 21–50 candidates | 396 | 0.283305 | 0.397685 | 0.484848 | +0.087164 |
| 51+ candidates | 241 | 0.214016 | 0.298268 | 0.395067 | +0.096799 |

Each marginal group has positive temporal-minus-early on every seed. This
shows descriptive consistency, not subgroup statistical significance: no
subgroup CIs were computed. Multi-source groups show larger ranking gains;
the large-candidate group still benefits but retains lower absolute F1.
These tables do not identify the intersection k=1 with 51+ candidates.
Recommended next decision: freeze this exploratory temporal protocol before
an independently authorized evaluation; no automatic training/test access.
