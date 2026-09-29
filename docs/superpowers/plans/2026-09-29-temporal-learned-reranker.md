# Temporal Learned Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Проверить маленький обучаемый корректор поверх frozen S1b против текущего Temporal-v3, с дешёвым train-only gate и без изменения принятой модели.

**Architecture:** Source-blind features и JSON linear head отделены от supervision. Collector один раз сохраняет frozen scores и observable features в hash-checked cache. Staged runner обучает три линейных кандидата, выбирает один на dev и разрешает validation/repeats лишь после соответствующих gates; notebook запускает те же функции в kernel с видимым выводом.

**Tech Stack:** Python, NumPy, NetworkX, PyTorch/PyG, scikit-learn, PyYAML, tqdm, pytest; существующие зависимости, без новых библиотек.

**Spec:** [Утверждённый дизайн](../specs/2026-09-29-temporal-learned-reranker-design.md). Пользователь одобрил письменный дизайн сообщением «го»; этот implementation plan ещё ожидает просмотра и выбора способа выполнения.

## Global Constraints

- Frozen GCN/count-head/финальный candidate set не меняются; контроль — correct_sources(beta=0.5), не snapshot.
- Только reference train/validation; test, final_holdout и independent_holdout не читаются.
- Fit indices 0..1619 (1620), dev 1620..2159 (540), всего 2160 из train=9990; validation=1998.
- Split по целым каскадам; 54-condition balance и непересечение union simulation+observation seeds обязательны.
- Six features и порядок строго из spec §4; source-blind early policy t1=1/max_steps=3 неизменна.
- C=[0.1,1.0,10.0], L2/lbfgs, max_iter=1000, tol=1e-8; scaler fit-only, cascade/class sample weights из spec §5.
- Выбор по dev mean per-cascade F1, ties 1e-12 → меньшее C; no refit; один head применяется к 7026/7027/7028.
- Gates относительно v3: delta F1>=0.02, exact-set delta>=0, k/subgroup decline не больше 0.02; validation CI lower>0; count/cardinality unchanged.
- Bootstrap: paired, true-k stratified, 2000 repeats, seed 9282026; summary по 1998 усреднённым per-cascade deltas, не 5994 независимым строкам.
- Output reports/runs/temporal_v3_learned_reranker/v1, append-only и вне protected inputs; backup только читается.
- Budget=1800s на вычислительную стадию, проверки на границах операций; partial/timeout не считаются успешным результатом.
- Все результаты exploratory=true; accepted v3, UI и thesis_report не изменяются этим пилотом.
- Реализация не означает разрешение реального обучения: после tests/smoke показать результат готовности и получить разрешение запускать fit/select.

## Review Focus

1. CPU/CUDA или runtime поменялся после cache: явная identity mismatch, не смешение scores (Task 2).
2. Пустой early, изолированный узел, перестановка candidate ID: корректные source-blind features и выравнивание (Task 1).
3. NaN/дублированные ragged offsets/labels/seed: остановка до fit и до записи результатов (Tasks 2, 3).
4. Нулевой exact delta, threshold .02, пустая subgroup: точный gate без округления, n=0 явно (Task 4).
5. Restart после failed/partial стадии: нет hidden refit или bypass gates, notebook печатает причину (Tasks 6, 7).

---

## Структура и окружение исполнения

Новые modules в `src/diffusion_sources/`:

- `temporal_learned_features.py`: FEATURE_NAMES, pure features, LinearHead и ranking.
- `temporal_learned_data.py`: CandidateTable, validation/subsets, cache identity и NPZ artifact IO.
- `temporal_learned_collect.py`: frozen replay/inference и balanced split checks.
- `temporal_learned_evaluation.py`: per-cascade metrics, baseline controls, gates/summary.
- `temporal_learned_training.py`: weights, three fits, selection, frozen head export.
- `temporal_learned_cli.py`: paths, stages/guards, CLI main.

Также создать `configs/temporal_learned_reranker_v1.json`,
`notebooks/colab_temporal_learned_reranker.ipynb`, `docs/temporal_learned_reranker_status.md`,
`tests/fixtures/temporal_learned.py`, по одному unit-test module на шесть modules,
и smoke tests `tests/smoke/test_temporal_learned_flow.py`, `test_temporal_learned_notebook.py`.
В конце изменить только ссылки README, registry и dev-log, не чистовик.
Не добавлять новый pyproject entrypoint: CLI через `python -m diffusion_sources.temporal_learned_cli`.

Перед исполнением прочитать spec, using-git-worktrees и создать изолированный checkout
от текущего HEAD, включающего этот план. Предпочтительное имя ветки
`codex/temporal-learned-reranker`. Существующие dirty report/docs в основной копии
не переносить, не stash/reset и не включать в новые commits.
Backup read-only: `D:/projects/diplom/diffusion-sources-localization/reports/backups/temporal_v3_20260929`.
Python для разработки: `D:/projects/diplom/diffusion-sources-localization/.venv/Scripts/python.exe`.
В worktree установить `$env:PYTHONPATH = Join-Path $PWD 'src'`; проверить импорт
из worktree, не старого editable install. Далее `python` означает этот executable;
в notebook — sys.executable. В PowerShell команды выполнить отдельно, без shell chains.

### Task 1: Observable features и переносимый linear scorer

**Files:** Create `src/diffusion_sources/temporal_learned_features.py`; Test `tests/unit/test_temporal_learned_features.py`.

**Interfaces:**
- `build_candidate_features(graph: nx.Graph, candidate_ids: tuple[int,...], probabilities: np.ndarray, early_mask: np.ndarray, final_mask: np.ndarray) -> np.ndarray`: float64 [candidates,6], probabilities выровнены с IDs, masks длины graph.number_of_nodes; contiguous graph IDs.
- `LinearHead` frozen dataclass: `feature_names: tuple[str,...]`, `mean/scale/coefficients: tuple[float,...]`, `intercept: float`, `C: float`, `schema_version: int=1`; `to_dict() -> dict`, `from_dict(payload: dict) -> LinearHead`, `decision_function(features: np.ndarray) -> np.ndarray`.
- `rank_candidates(candidate_ids: tuple[int,...], probabilities: np.ndarray, features: np.ndarray, predicted_count: int, head: LinearHead) -> frozenset[int]`: tie-break (-learned_score,-original_probability,node_id), min(k,n) выдача. Нет labels/true k в signatures.

- [ ] **1. RED tests:** `test_features_manual_graph`, `test_empty_early_isolate_and_external_neighbor`, `test_permutation_alignment`, `test_invalid_masks_ids_probabilities`, `test_linear_head_schema_nonfinite_and_zero_scale`, `test_rank_ties_and_cardinality`, `test_target_free_signatures`. Ручной граф 0–1–2 плюс isolate3: кандидат1 с early={0} имеет early fraction .5; кандидату3 обе fractions=0. При перестановке IDs строки переставляются, значения не меняются. Некорректные ID/finite/ranges/shape/schema и k вне 1..3 → ValueError; положительные scale обязательны. JSON round-trip scores atol=1e-12, одинаковые selected sets.
- [ ] **2. Run RED:** `python -m pytest tests/unit/test_temporal_learned_features.py -q`; ожидается missing module/API, не ошибка fixture.
- [ ] **3. Implement:** шесть признаков из spec, vectorized neighbor counts через shared graph edges, finite/range validation; score ((X-mean)/scale)@coefficients+intercept. Не добавлять extra features или преобразование GCN scores.
- [ ] **4. Run GREEN:** та же команда; все новые tests pass.
- [ ] **5. Commit:** только module/test, message `feat: add source-blind learned reranker features and scorer`.

### Task 2: Ragged candidate table и безопасный cache

**Files:** Create `src/diffusion_sources/temporal_learned_data.py`, `tests/fixtures/temporal_learned.py`; Test `tests/unit/test_temporal_learned_data.py`.

**Interfaces:** consumes FEATURE_NAMES/LinearHead Task1; produces:
- `CandidateTable` dataclass с arrays `features` [R,6], `candidate_ids/probabilities/early_flags/labels` [R], `offsets` [N+1], `indices/predicted_counts/true_counts/early_empty` [N], `snapshot_sources: tuple[frozenset[int],...]`, `early_mask_hash: str`.
- `validate_table(table: CandidateTable) -> None`; `subset_table(table: CandidateTable, positions: Sequence[int]) -> CandidateTable` сохраняет весь каскад и recalculates offsets.
- `cache_identity(data_dir: Path, run_dir: Path, split: str, indices: Sequence[int], device: torch.device) -> dict` принимает только train/validation; читает только явные допустимые files, runtime/source hashes и replay policy.
- `save_cache(root: Path, stage: str, identity: dict, table: CandidateTable) -> None`, `load_cache(root: Path, stage: str, identity: dict) -> CandidateTable`: atomic directory, schema1, cache.npz+manifest.json+complete, no pickle/object arrays.
- Fixture `tiny_table() -> CandidateTable`: шесть каскадов с k=1/2/3 дважды, по шесть candidates, finite features и корректными offsets; `tiny_graph() -> nx.Graph` содержит ID0..5. Fixtures без обращения к реальным reports.

- [ ] **1. RED tests:** `test_table_roundtrip_and_whole_cascade_subset`, `test_offsets_labels_and_duplicate_indices_rejected`, `test_identity_runtime_device_and_input_change`, `test_partial_or_tampered_cache_rejected`, `test_output_overlap_and_windows_alias`, `test_forbidden_split_never_opens_archive`. Offsets start0/endR, строго растут, n>=1; labels binary и sum=true_count 1..3; per-cascade IDs уникальны, sources subset IDs/cardinality=min(count,n); probabilities [0,1]; indices integer/unique. Пустые subsets запрещены. Protected output equality/ancestor/descendant после resolve → ValueError. Пересохранение готового stage → FileExistsError, partial stage → error.
- [ ] **2. Run RED:** `python -m pytest tests/unit/test_temporal_learned_data.py -q`.
- [ ] **3. Implement:** explicit SHA256 identity и versions Python/NumPy/Torch/PyG/sklearn/NetworkX/SciPy/PyYAML; CPU/CUDA device и dtype в identity, paths вне content identity для переносимости. Перед/после generation identity перепроверяется. Numeric NPZ включает ragged snapshot sources с собственными offsets, а supervision labels хранит отдельным именованным массивом. Reuse sha256_file и защиту resolved input/output paths; не менять старые IO modules.
- [ ] **4. Run GREEN:** та же команда; fixture отдельно импортируется из tests.fixtures.temporal_learned.
- [ ] **5. Commit:** только files Task2, message `feat: add authenticated learned reranker candidate cache`.

### Task 3: Read-only frozen collector и контроль train split

**Files:** Create `src/diffusion_sources/temporal_learned_collect.py`, `configs/temporal_learned_reranker_v1.json`; Test `tests/unit/test_temporal_learned_collect.py`.

**Interfaces:** consumes features/table Task1–2 and existing replay/predict_joint; produces:
- `check_train_partition(config: dict, archive: Mapping[str,np.ndarray]) -> tuple[list[int],list[int]]`: exact fit/dev ranges, balance 54 conditions, unique seeds/union isolation.
- `collect_candidates(data_dir: Path, run_dir: Path, split: str, indices: Sequence[int], device: torch.device, *, budget_seconds: float=1800) -> CandidateTable`.
- Config JSON: protocol/features/C/splits/budget из spec и `expected_inputs` (reference graph/generation/train/validation + best_model/config каждого seed), только hashes без machine-specific paths. Извлечь точные locked hashes из локального backup_manifest.json; это metadata, не independent target metrics. CLI позже сверяет их до открытия target arrays.

- [ ] **1. RED tests:** `test_partition_balanced_and_seed_union_disjoint`, `test_partition_tampering_stops`, `test_real_tiny_replay_and_frozen_model`, `test_labels_removed_before_gcn`, `test_candidate_coverage_no_filtering`, `test_control_beta_zero`, `test_deadline_and_changed_inputs_stop`. Генерационный fixture строить с existing simulate_ic/observe/replay policy и fixed tiny model; SourceSampler RNG не обходить. Spy model assert source_labels/source_count отсутствуют; все требуемые indices сохранены в порядке, corrupted replay не пропускается. Deadline monkeypatch monotonic, не ждать реально.
- [ ] **2. Run RED:** `python -m pytest tests/unit/test_temporal_learned_collect.py -q`.
- [ ] **3. Implement:** source-head shared, frozen config dimension checks, strict weights_only checkpoint load, eval()+torch.inference_mode. Existing feature builder читает только совместимый cache либо пересчитывает in-memory. Shared graph/edge_index, один CPU→device example за раз, не держать все GPU graph clones. Replay label access только для validation восстановления и training supervision. Сохранять native snapshot sources (не пересортировать torch ties), predicted count и fingerprint полных early masks всех graph nodes с indices, не только candidate flags. Лимит проверять до/после каждой операции загрузки/replay/inference; ошибки не создают complete.
- [ ] **4. Run GREEN:** та же команда; проверить locked config keys и числа против spec без чтения test/holdout arrays.
- [ ] **5. Commit:** только files Task3, message `feat: collect frozen development scores for temporal reranking`.

### Task 4: Парные метрики и точные gates

**Files:** Create `src/diffusion_sources/temporal_learned_evaluation.py`; Test `tests/unit/test_temporal_learned_evaluation.py`.

**Interfaces:** consumes CandidateTable/LinearHead; produces:
- `evaluate_reranker(table: CandidateTable, head: LinearHead, graph: nx.Graph, *, include_ci: bool) -> dict` с rows и агрегатами `snapshot/v3/learned`, `delta_f1`, `delta_exact`, `delta_by_k`, `delta_by_candidates`, `count_unchanged`, `cardinality_unchanged`, optional f1_ci, exploratory=true.
- `dev_gate(report: dict) -> dict`, `validation_gate(report: dict) -> dict`: `{passed:bool,reasons:list[str]}`.
- `summarize_repeats(reports: Sequence[dict]) -> dict`: exact seeds7026/27/28, aligned indices/true sets/k/candidate IDs/early mask hash, per-seed metrics/CI/sample SD, mean deltas and summary gate.
- `check_saved_snapshot_f1(report: dict, run_metrics: dict, *, tolerance: float=1e-6) -> None`: use existing saved metrics estimated-k/all schema; не hardcode overall F1 для подвыборки.

- [ ] **1. RED tests:** `test_pair_metrics_against_manual_sets`, `test_gate_exact_zero_and_f1_boundary`, `test_missing_k_and_empty_candidate_bins`, `test_ci_and_count_failure`, `test_summary_averages_cases_before_bootstrap`, `test_repeated_seed_or_mask_mismatch`, `test_snapshot_control_mismatch_stops`. delta=.02 passes с 1e-12 tolerance; .019 fails; exact delta< -1e-12 fails; k/size=-.02 passes, -.021 fails. CI lower<=0 строго fail. Empty size bin n=0/None/not_applicable, не NaN; отсутствующая k группа fail. Summary bootstrap получает N aligned averaged deltas, не 3N.
- [ ] **2. Run RED:** `python -m pytest tests/unit/test_temporal_learned_evaluation.py -q`.
- [ ] **3. Implement:** existing set_metrics/source_set_distances/source_radius_hits, correct_sources(.5), CandidateScores, paired_bootstrap_ci(2000,9282026). True sets из labels только evaluator. Snapshot/temporal native count и cardinality проверять на каждой строке; группы до округления. Вывод learned-v3, snapshot только secondary diagnostic. Dev include_ci=false, validation=true; summary gate строго spec §6, один head неизменен.
- [ ] **4. Run GREEN:** та же команда; существующий `tests/unit/test_temporal_statistics.py` также pass.
- [ ] **5. Commit:** только files Task4, message `feat: evaluate learned temporal reranker against frozen v3`.

### Task 5: Три CPU fit и immutable selection

**Files:** Create `src/diffusion_sources/temporal_learned_training.py`; Test `tests/unit/test_temporal_learned_training.py`.

**Interfaces:** consumes table/head/evaluator; produces:
- `cascade_class_weights(table: CandidateTable) -> np.ndarray`: веса spec §5, mean1.
- `fit_head(table: CandidateTable, C: float) -> LinearHead`: fit-only StandardScaler и LogisticRegression, no extra class_weight.
- `select_head(fit: CandidateTable, dev: CandidateTable, graph: nx.Graph, *, budget_seconds:float=1800) -> tuple[LinearHead,dict]`: ровно C .1/1/10, таблица dev metrics, выбранный C, dev report/gate; без IO, validation или refit.

- [ ] **1. RED tests:** `test_weights_equal_cascade_and_class_mass`, `test_scaler_only_fit_rows`, `test_sklearn_json_score_and_selection_parity`, `test_three_C_and_smaller_C_tie`, `test_selected_head_not_refitted`, `test_nonconvergence_is_error`, `test_fit_dev_overlap_and_timeout`. На каскадах n=6,k=1/2/3 sum weights per cascade равны; class sums внутри равны; mean1. Shift dev features не меняет fit scaler. JSON decision_function allclose atol1e-12/rtol1e-10, selected sets идентичны. Spy fit count=3, четвёртого fit нет; ConvergenceWarning raise; selection не принимает forbidden paths.
- [ ] **2. Run RED:** `python -m pytest tests/unit/test_temporal_learned_training.py -q`.
- [ ] **3. Implement:** проверенная установленная версия sklearn API L2/lbfgs/max_iter1000/tol1e-8, warnings-as-errors. Scaler из fit, coefficients/intercept экспортируются в LinearHead; C positive и входит в fixed grid. До/после каждого fit проверить elapsed бюджет, print start/end и rows. No probability calibration, no target-derived input column. Selection возвращает все результаты даже при failed dev gate.
- [ ] **4. Run GREEN:** та же команда; synthetic end-to-end fit не использует реальные данные.
- [ ] **5. Commit:** только files Task5, message `feat: train and select lightweight temporal reranker`.

### Task 6: Stage runner без обхода gates

**Files:** Create `src/diffusion_sources/temporal_learned_cli.py`; Test `tests/unit/test_temporal_learned_cli.py`, `tests/smoke/test_temporal_learned_flow.py`.

**Interfaces:** consumes Tasks1–5; produces:
- `LearnedPaths` dataclass: `data_dir:Path`, `runs:dict[int,Path]`, `output_dir:Path`, `protocol_config:Path`; никаких test/holdout paths.
- `run_stage(stage: str, paths: LearnedPaths, device: torch.device) -> dict`: smoke/cache/select/validate/confirm/summary; `main(argv: Sequence[str]|None=None) -> None` с соответствующим argparse.
- Stage artifacts: smoke/selection/validation/confirm/summary через existing write_stage/read_stage; caches `cache_train_7026`, `cache_validation_7026/7027/7028` через Task2 IO. head embedded в selection/payload.json и защищён file hashes; самостоятельного неподтверждённого head.json нет.

- [ ] **1. RED tests:** `test_smoke_has_no_fit_and_prints`, `test_train_cache_only_requested_once`, `test_failed_dev_never_reads_validation`, `test_failed_primary_never_collects_repeats`, `test_direct_confirm_cannot_bypass_gate`, `test_restart_does_not_refit`, `test_tampered_selection_and_partial_stop`, `test_frozen_input_hashes_and_bad_paths`, `test_complete_stage_flow_synthetic`. Spy collector/fit/open отслеживает boundaries; даже direct CLI validate/confirm проверяет saved prerequisites и immutable selection identity. Save failed gates, print reasons, return blocked status без успешного следующего stage. Cached resume разрешён только с exact identity, не runtime/device mixing.
- [ ] **2. Run RED:** `python -m pytest tests/unit/test_temporal_learned_cli.py tests/smoke/test_temporal_learned_flow.py -q`.
- [ ] **3. Implement:** smoke6 train indices, без fit; cache2160 только7026 после authenticated smoke pass; selection требует готовый train cache и splits ровно Task3 ranges; validation onlydevpass+full1998+saved snapshot control; confirm onlyprimarypass+7027/28+same head and observations; summary onlycomplete3. Locked graph/generation/train/7026 проверяются до smoke, validation archive hash читается только после dev pass, 7027/28 runtime inputs — после primary pass. CLI arguments `--stage`, `--data-dir`, `--run-7026/7027/7028`, `--output-dir`, `--protocol-config`, `--device cpu|cuda`. No recursive data/archive discovery. Повторный confirm уже сохранённого результата ничего не обучает. Все exit/status сообщения видимы; исключения не проглатывать.
- [ ] **4. Run GREEN:** та же команда; synthetic flow с actual scoring/fitting/evaluation/artifact IO, mock только внешней frozen collection границы. Полный suite в Task8.
- [ ] **5. Commit:** только files Task6, message `feat: add guarded learned reranker experiment stages`.

### Task 7: Notebook и понятные инструкции

**Files:** Create `notebooks/colab_temporal_learned_reranker.ipynb`, `docs/temporal_learned_reranker_status.md`; Test `tests/smoke/test_temporal_learned_notebook.py`.

**Interfaces:** notebook импортирует LearnedPaths/run_stage из Task6, вызывает stages прямо в kernel. Локально default paths разрешаются от project root + backup/local_paths.json; Colab — исходные Drive folders facebook_main_v2 и data/facebook_main, output new learned root. Protocol config идёт с кодом, не требует загрузки backup в Drive.

- [ ] **1. RED tests:** `test_notebook_valid_python_and_stage_order`, `test_setup_kernel_import_and_no_model_run`, `test_false_gate_prints_and_does_not_advance`, `test_selection_requires_explicit_run_confirmation`, `test_output_is_not_captured_silently`. Parse/compile все cells; выполнить notebook logic с stage spy. Setup не вызывает smoke/fit/inference. Fit cell требует exact input `RUN_LEARNED_RERANKER_PILOT`, неверный текст stops; failed gate запрещает downstream даже при ручном запуске нижней ячейки. Нет subprocess capture_output для реальных stages.
- [ ] **2. Run RED:** `python -m pytest tests/smoke/test_temporal_learned_notebook.py -q`.
- [ ] **3. Implement:** markdown и Python ячейки: 1 setup/import/device (опционально Drive mount только в Colab); 2 paths+locked train/7026 input check (validation и остальные seed пока не открывать); 3 smoke6+timing+STOP перед дорогой стадией; 4 train cache2160; 5 explicit confirmation+fit/select+dev gate; 6 conditional validation; 7 conditional confirms; 8 summary. Setup использует sys.executable, не заменяет CUDA torch wheel без нужды; не git pull поверх dirty checkout. Print полные stage results и location, никакого тихого run_stage. В status описать что значит F1/exact/count/delta/CI и что результаты пока отсутствуют.
- [ ] **4. Run GREEN:** та же команда; local setup cell проверить с текущим kernel Python без запуска cache/fit.
- [ ] **5. Commit:** только files Task7, message `docs: add stepwise learned reranker experiment notebook`.

### Task 8: Полная проверка и handoff перед реальным fit

**Files:** Modify `README.md` (ссылка на notebook/status), `docs/experiment_registry.md` (новая запись без invented metrics), `development_log.md` (новые факты), spec/status (фактический этап). Не изменять thesis_report/ или app.py.

- [ ] **1. Verify import:** `python -c "import diffusion_sources.temporal_learned_cli as m; print(m.__file__)"`; ожидается worktree src path. Если original editable path — исправить PYTHONPATH, не тестировать старый код.
- [ ] **2. Run full suite:** `python -m pytest -q -p no:cacheprovider --basetemp=reports/learned_reranker_verification`; выбрать новый basetemp при существующем каталоге, не удалять старые test artifacts. Ожидается все tests pass; прежние warnings отделить от новых.
- [ ] **3. Run syntax/diff checks:** `python -m compileall -q src scripts`; `git diff --check`. Ожидается exit0, без syntax/whitespace ошибок.
- [ ] **4. Real read-only smoke:** run_stage('smoke', local paths, CPU) из worktree с output `reports/runs/temporal_v3_learned_reranker/local_smoke_v1`, не рабочий v1. Ровно 6 train примеров; assert replay/beta0/count/schema, показать elapsed и projected2160 cost. Fit/cache2160/validation не запускать. До/после сравнить backup_manifest SHA и protected graph/config/checkpoint SHA; никаких old targets.
- [ ] **5. Review:** самостоятельная spec/plan coverage проверка и, для Native execution, один независимый whole-branch review по requesting-code-review skill. Исправления через RED→GREEN и повторную проверку затронутых tests/full suite перед завершением. Не создавать новые задачи в sidebar; reviewer — только рабочий subagent при разрешённом выбранном способе.
- [ ] **6. Update docs/log:** record фактические test counts/times, smoke duration и limitations; ни один synthetic gate не выдавать за результат реальной модели. Links должны вести к существующим files. Commit только implementation docs этой ветки, не чужие dirty изменения основной копии.
- [ ] **7. Handoff:** показать пользователю notebook, точные номера ячеек, время smoke и что fit ещё не выполнен. Получить разрешение реального эксперимента; при необходимости Git merge/push согласовать отдельно. Не открывать независимый набор и не менять принятую модель даже при synthetic pass.

## Self-review и передача

Coverage: spec §1–2 → Tasks3/6/8; §3 → Tasks2/3/5; §4 → Task1;
§5 → Tasks1/5; §6 → Tasks4/6; §7 → Tasks2/6; §8 → Tasks3/5/7;
§9 → tests всех Tasks + Task8; §10 → этот план и отдельное согласование исполнения.
Сигнатуры CandidateTable/LinearHead/run_stage одинаковы во всех Tasks.
Ни один Task не требует нового dataset, изменения UI или ретюнинга independent evaluation.

Рекомендованный способ — Native: основной агент выполняет этот связанный pipeline
в текущей задаче по executing-plans; один независимый review в конце, не отдельный
исполнитель на каждый module. Альтернатива — subagent-driven с отдельными
исполнителями/reviewers по Tasks. Выбор и просмотр плана остаются за пользователем;
до ответа продуктовый код, worktree исполнения и обучение не создаются.
