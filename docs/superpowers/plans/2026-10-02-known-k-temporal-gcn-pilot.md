# Known-k Temporal GCN Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Получить один воспроизводимый Colab pilot новой GCN, которая обучается на раннем и конечном снимках с заданным `k`, и решить по заранее заданному gate, нужны ли повторы.

**Architecture:** Существующий `NodeOnlyGCN` получает 9 каналов: 5 конечных признаков фактического frozen S1b, раннюю маску и one-hot `k`. Отдельный сборщик воспроизводит раннее наблюдение из train/validation архивов; существующий `fit_node_model` обучает с нуля и выбирает checkpoint по validation oracle-k F1. Парная оценка сравнивает его с замороженным S1b + Temporal-v3, которому дано то же `k` и те же наблюдения.

**Tech Stack:** Python ≥3.10, PyTorch/PyG, NumPy, NetworkX, YAML, pytest, Colab notebook.

**Spec:** `docs/superpowers/specs/2026-10-02-known-k-temporal-gcn-design.md`.

## Global Constraints

- Работа только в `experiment/known-k-temporal-gcn`; текущие checkpoints, backup, отчёты, `master` и неизвестный-k UI не перезаписывать.
- Только существующие `facebook_main` train (`9990`) и validation (`1998`); `test`, старый independent holdout и новый holdout в pilot не читать.
- Вход: 5 конечных признаков реального frozen S1b + бинарное раннее наблюдение `t=1` + one-hot `k ∈ {1,2,3}`. Не подмешивать истинные source labels или infection times в `x`.
- Модель: существующий двухслойный `NodeOnlyGCN(input_dim=9, hidden_dim=64, dropout=0.2)`, начальная инициализация с нуля; без count-head и ranking-loss.
- Pilot: seed `7026`; Adam `0.001`, batch `3`, максимум `100` эпох, patience `10`, candidate-only weighted BCE, validation oracle-k F1 для ранней остановки.
- Основной контроль: замороженный S1b + Temporal-v3 `beta=0.5` с **тем же известным k**. Count accuracy в отчёте известного-k режима — `N/A`.
- Gate: `ΔF1 ≥ 0.02`, нижняя граница paired stratified bootstrap CI `>0`, exact-set accuracy не ниже контроля, падение F1 в каждой группе `k` и candidate count не более `0.02`.
- Все новые результаты — append-only с SHA-256 входов/выходов; при изменении входов или неполной стадии отказ, не тихий перезапуск.
- После pilot остановиться. Повторы `7027/7028`, новый sealed holdout, перенос checkpoint и подключение UI — **отдельный следующий план только при успешном gate**. Если gate не пройден, сохранить отрицательный результат.

## Review Focus

- Пустая ранняя маска допустима: Task 1 тестирует 9 каналов без выдуманного раннего заражения.
- `k` вне 1–3 либо кандидатов меньше `k`: Task 1 отвергает пример до обучения/инференса.
- Испорченный replay, seed или labels вне candidate mask: Task 1 останавливается с индексом примера, не пропускает его молча.
- Смена frozen checkpoint, архива или конфигурации между стадиями: Task 4 отвергает hash/identity mismatch.
- Повторный запуск и частичная запись: Task 4 возвращает только проверенный готовый результат. Незавершённый pilot разрешает лишь явный resume с тем же frozen identity и проверенным last-checkpoint; иначе требует нового output root. Старый результат не затирается.

---

## File map

- `src/diffusion_sources/known_k_temporal_data.py`: чистая сборка входа без targets и чтение train/validation с replay; общий `edge_index` между примерами.
- `src/diffusion_sources/known_k_temporal_eval.py`: парные прогнозы, агрегаты, CI и фиксированный gate.
- `src/diffusion_sources/known_k_temporal_pilot.py`: защищённые стадии `freeze/smoke/pilot`, обучение существующим trainer и неизменяемые артефакты.
- `configs/known_k_temporal_gcn_pilot.yaml`: ровно один зафиксированный pilot-конфиг без сетки параметров.
- `notebooks/colab_known_k_temporal_gcn_pilot.ipynb` и `docs/colab_known_k_temporal_gcn_pilot.md`: Colab setup, явные стадии и STOP перед долгим pilot.
- `tests/unit/test_known_k_temporal_data.py`, `tests/unit/test_known_k_temporal_eval.py`, `tests/unit/test_known_k_temporal_pilot.py`, `tests/smoke/test_known_k_temporal_notebook.py`: независимые RED→GREEN проверки.
- `development_log.md`: ход, ограничения и фактический результат; `thesis_report/` в pilot не менять.

### Task 1: Два снимка и заданное k без утечки меток

**Files:** Create `src/diffusion_sources/known_k_temporal_data.py`; Test `tests/unit/test_known_k_temporal_data.py`.

**Interfaces:**
- `make_observation(final_features: np.ndarray, early_mask: np.ndarray, candidate_mask: np.ndarray, k: int, edge_index: torch.Tensor) -> Data`: только inference-вход (`x`, `edge_index`, `candidate_mask`, `observed_mask`, `early_observed_mask`), без `source_labels`/`infection_times`.
- `load_known_k_split(data_dir: Path, split: Literal['train','validation'], graph: nx.Graph, builder: SnapshotFeatureBuilder, feature_names: Sequence[str], indices: Sequence[int]) -> list[Data]`: вызывает `replay_early_mask`, пересчитывает пять конечных признаков фактического S1b, добавляет `source_labels`, `source_count` и `example_index` только к обучающей/оценочной копии Data. Предоставляет Task 2–4 упорядоченные примеры; индекс нужен для парной проверки.

- [ ] **Step 1: Write failing tests.** `test_make_observation_is_source_blind`: `assert observation.x.shape == (n, 9)`, `assert not hasattr(observation, 'source_labels')`, `assert observation.x[:, 5].tolist() == early_mask.astype(float).tolist()` и one-hot `k`. `test_empty_early_and_bad_k`: пустая маска даёт нулевой канал; `k=0/4` или `candidate_count<k` дают `ValueError`. `test_archive_replay_rejects_corruption`: повреждённые seed/labels дают ошибку с индексом.
- [ ] **Step 2: Verify RED.** `python -m pytest -q tests/unit/test_known_k_temporal_data.py` → FAIL по отсутствующему API/ожидаемым контрактам.
- [ ] **Step 3: Implement minimal data module.** Использовать `SnapshotFeatureBuilder.build(..., base_features=..., candidate_mask=...)` и существующий `replay_early_mask`; не пересимулировать вручную. Один `edge_index` разделяется примерами, NPZ-поля распаковываются один раз на split.
- [ ] **Step 4: Verify GREEN.** Та же команда → PASS; отдельно показать, что изменение только `source_labels` не меняет `make_observation(...).x`.
- [ ] **Step 5: Commit.** `git add src/diffusion_sources/known_k_temporal_data.py tests/unit/test_known_k_temporal_data.py && git commit -m "feat: build source-blind known-k temporal inputs"`.

### Task 2: Парная оценка и заранее фиксированный gate

**Files:** Create `src/diffusion_sources/known_k_temporal_eval.py`; Test `tests/unit/test_known_k_temporal_eval.py`.

**Interfaces:**
- `evaluate_known_k_pairs(cases: Sequence[Data], graph: nx.Graph, candidate_logits_by_index: Mapping[int, torch.Tensor], s1b_logits_by_index: Mapping[int, torch.Tensor], *, beta: float = 0.5) -> dict`: требует одинаковые ключи `example_index`, уникальные индексы, truth, masks и `k`; берёт `early_observed_mask` из Data, строит snapshot-known-k, Temporal-v3-known-k и новую модель; для `CandidateScores` переводит S1b logits в вероятности через sigmoid; возвращает rows, `all/by_k/by_candidates`, `delta_f1`, `delta_exact`, `f1_ci`.
- `known_k_pilot_gate(report: dict) -> dict`: `passed` и исчерпывающие `reasons` из Global Constraints; CI — существующий `paired_bootstrap_ci(repetitions=2000, seed=9282026)`.

- [ ] **Step 1: Write failing tests.** `test_paired_rows_require_matching_indices`: разные keys у двух mapping дают `ValueError`; при совпадении `assert len(row['candidate_sources']) == row['k']` и `assert row['control_sources'] == expected_beta_half`. `test_gate_checks_every_group`: параметризация отчёта с `delta_f1=.02`, `f1_ci=[.001,.04]`, `delta_exact=0`, каждой `delta_by_k`/`delta_by_candidates=0`; по одному нарушенному полю `assert not gate['passed']`. Агрегаты не содержат count accuracy как достижения.
- [ ] **Step 2: Verify RED.** `python -m pytest -q tests/unit/test_known_k_temporal_eval.py` → FAIL по отсутствующему API.
- [ ] **Step 3: Implement paired evaluator.** Использовать `predict_oracle_k`, `CandidateScores`/`correct_sources`, `set_metrics`, `source_set_distances`, `source_radius_hits`, `_candidate_bin`; не читать файлы и не подбирать beta.
- [ ] **Step 4: Verify GREEN.** Та же команда → PASS, включая тест перестановки/mismatch кейсов, который обязан падать до агрегирования.
- [ ] **Step 5: Commit.** `git add src/diffusion_sources/known_k_temporal_eval.py tests/unit/test_known_k_temporal_eval.py && git commit -m "feat: evaluate known-k temporal pilot against paired control"`.

### Task 3: Обучение NodeOnlyGCN без новой архитектурной ветки

**Files:** Create `configs/known_k_temporal_gcn_pilot.yaml`; Create `src/diffusion_sources/known_k_temporal_pilot.py`; Test `tests/unit/test_known_k_temporal_pilot.py`.

**Interfaces:**
- `KnownKPaths(data_dir: Path, frozen_s1b_dir: Path, output_dir: Path, config_path: Path)` — неизменяемый набор путей.
- `run_stage(stage: Literal['freeze','smoke','pilot'], paths: KnownKPaths, device: torch.device, *, resume: bool = False) -> dict` — главный API для CLI/notebook. `smoke` использует 6 train и 6 validation примеров, одну CPU-эпоху и `training_performed=True`, но не делает вывод о F1; `pilot` использует весь train/validation. Resume допустим только для прерванного `pilot` с проверенной identity, никогда для завершённой стадии.
- `main(argv: list[str] | None = None) -> int` — CLI с теми же стадиями.

- [ ] **Step 1: Write failing tests.** Конфиг строго задаёт 9/64/0.2, seed 7026, 0.001/3/100/10, `evaluate_test: false`; локальный smoke на fixture действительно обновляет вес, сохраняет лучший checkpoint и не создаёт test predictions. `pilot` при неверном device/конфиге отказывается до fit.
- [ ] **Step 2: Verify RED.** `python -m pytest -q tests/unit/test_known_k_temporal_pilot.py` → FAIL по отсутствующему API.
- [ ] **Step 3: Implement training path.** Переиспользовать `NodeOnlyGCN`, `calculate_pos_weight`, `fit_node_model`, `save_training_result`, `set_seed`, `predict_oracle_k`; адаптер Data из Task 1. Историю `count_accuracy=1.0` у legacy trainer явно маркировать технической oracle-k величиной, в pilot-отчёте `N/A`.
- [ ] **Step 4: Verify GREEN.** Та же команда → PASS; CPU smoke на настоящих первых шести train/validation примерах, если локальный архив доступен, иначе fixture-only и явная пометка в журнале. Проверить `torch.load(..., weights_only=True)` выбранного checkpoint.
- [ ] **Step 5: Commit.** `git add configs/known_k_temporal_gcn_pilot.yaml src/diffusion_sources/known_k_temporal_pilot.py tests/unit/test_known_k_temporal_pilot.py && git commit -m "feat: train guarded known-k temporal pilot"`.

### Task 4: Freeze, защищённые артефакты и запрет лишних данных

**Files:** Modify `src/diffusion_sources/known_k_temporal_pilot.py`; Test `tests/unit/test_known_k_temporal_pilot.py`.

**Interfaces:** `run_stage` из Task 3; существующие `sha256_file`, `write_stage`, `read_stage` из `temporal_pilot_artifacts.py`.

- [ ] **Step 1: Write failing tests.** `freeze` фиксирует SHA-256 `graph.npz`, `train.npz`, `validation.npz`, generation/config YAML, S1b config/checkpoint; `smoke` требует freeze, `pilot` требует smoke и GPU, кроме явно тестового fixture; изменение защищённого входа, частичная стадия и чужой output root отвергаются. Только явный `resume=True` принимает незавершённый pilot с тем же frozen identity и неповреждённым last-checkpoint. Spy/fixture доказывает отсутствие открытия `test.npz` и любого independent holdout.
- [ ] **Step 2: Verify RED.** `python -m pytest -q tests/unit/test_known_k_temporal_pilot.py` → FAIL на новых защитных тестах.
- [ ] **Step 3: Implement guards and persistence.** Полный pilot сохраняет `last_checkpoint.pt` для явного resume, `best_model.pt`, history, per-example report и stage manifest. Frozen input identity не зависит от выходных файлов; SHA-256 checkpoint/history/predictions хранить в `file_hashes` готовой стадии и проверять при чтении. Для незавершённого обучения отдельно фиксировать frozen identity рядом с last-checkpoint и проверять его перед `fit_node_model(resume_from=...)`; не перезаписывать завершённую стадию. Парные logits и metrics формируются через Task 2; нельзя сравнивать разные индексы или маски.
- [ ] **Step 4: Verify GREEN.** Та же команда → PASS; повторный `smoke` возвращает проверенную стадию, а повреждённый checkpoint/manifest вызывает отказ.
- [ ] **Step 5: Commit.** `git add src/diffusion_sources/known_k_temporal_pilot.py tests/unit/test_known_k_temporal_pilot.py && git commit -m "feat: freeze and authenticate known-k pilot stages"`.

### Task 5: Colab handoff и локальная проверка

**Files:** Create `notebooks/colab_known_k_temporal_gcn_pilot.ipynb`; Create `docs/colab_known_k_temporal_gcn_pilot.md`; Test `tests/smoke/test_known_k_temporal_notebook.py`; Modify `development_log.md`.

**Interfaces:** `KnownKPaths` и `run_stage` из Task 3; Colab вызывает `freeze`, затем `smoke`, останавливается, затем только после явного согласия `pilot`.

- [ ] **Step 1: Write failing notebook tests.** Проверить порядок ячеек, GPU/setup проверку, отсутствие `Run all`-пути к pilot без явного подтверждения, отсутствие `test`/holdout reads, видимые progress/checkpoint/output paths и STOP после smoke и pilot.
- [ ] **Step 2: Verify RED.** `python -m pytest -q tests/smoke/test_known_k_temporal_notebook.py` → FAIL, notebook отсутствует.
- [ ] **Step 3: Implement notebook and runbook.** Ячейки: setup/code revision; Drive paths/config; `freeze`; `smoke`; явный ввод `RUN_KNOWN_K_TEMPORAL_PILOT`; `pilot`; печать gate и путей, затем STOP. Документ поясняет F1/precision/recall/exact/distance/Hit/CI и статус exploratory. В `development_log.md` записать только реализованное и локально проверенное, без вымышленных GPU-метрик.
- [ ] **Step 4: Verify GREEN.** Notebook smoke test → PASS; `python -m pytest -q -p no:cacheprovider` → весь набор PASS (в worktree предварительно доступен игнорируемый backup). `git diff --check` → 0; проверить, что `thesis_report/`, existing checkpoints и unknown-k UI не изменены.
- [ ] **Step 5: Commit.** `git add notebooks/colab_known_k_temporal_gcn_pilot.ipynb docs/colab_known_k_temporal_gcn_pilot.md tests/smoke/test_known_k_temporal_notebook.py development_log.md && git commit -m "docs: hand off guarded known-k Colab pilot"`.

## Handoff after implementation

Предоставить пользователю notebook и указать **ровно следующую ячейку**: сначала setup/freeze/smoke. До получения smoke-вывода не просить запускать pilot; после pilot gate решает, создавать ли отдельный план повторов/нового независимого набора. Никаких утверждений об улучшении до реального Colab-результата. UI известного `k` проектируется отдельным планом после pilot и не выдаёт неудачную модель за улучшение.
