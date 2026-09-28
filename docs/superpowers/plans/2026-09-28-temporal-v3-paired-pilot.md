# Temporal-v3 Paired Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать воспроизводимую парную проверку ранней временной подсказки поверх frozen S1b без обучения и без чтения test/holdout.

**Architecture:** Отдельный replay-компонент восстанавливает ранний source-blind снимок. Scoring-компонент корректирует готовые candidate scores, evaluator считает парные метрики/CI, CLI разделяет train-selection и validation. Новый notebook запускает каждый этап отдельно; snapshot-training не меняется.

**Tech Stack:** Python, NumPy, NetworkX, PyTorch/PyG, PyYAML, tqdm, pytest; уже установленные зависимости проекта, без новых библиотек.

**Spec:** `docs/superpowers/specs/2026-09-28-temporal-v3-paired-pilot-design.md` (одобрена пользователем 2026-09-28).

## Global Constraints

- Этап называется `temporal_v3_paired_pilot`; обучения и создания финального temporal-датасета нет.
- Только существующие train/validation; исходные test и final holdout не читаются.
- Финальный снимок, граф, признаки S1b и candidate mask неизменны.
- Ранний снимок: `t1=1`, исходный `max_steps=3`; source-blind Bernoulli sampling с observation_fraction, пустое наблюдение допустимо.
- RNG: `SeedSequence([5000000, observation_seed, 1])`; узлы отсортированы.
- Beta grid `[0, 0.1, 0.25, 0.5, 1.0]`; выбор только по первым 540 train-примерам (10 циклов 54 условий), ties — меньшее beta.
- Score `sigmoid(source_logit) + beta * early_observed`; k берётся из неизменной count-head; beta=0 в точности сохраняет исходный predict_joint.
- Validation: все 1998 примеров; control F1 для seed 7026 `0.362996329663`, tolerance `1e-6`.
- Bootstrap: paired, стратифицированный по k, 2000 повторов, seed 9282026, percentile CI 95%.
- Первичный gate: delta F1 >= 0.02, CI lower > 0, снижение k=2 и k=3 не больше 0.02; count accuracy неизменна.
- После primary pass: inference seeds 7027/7028 на тех же масках и beta; переход к следующему дизайну при delta > 0 на каждом seed и mean delta >= 0.02.
- Новые артефакты только в `reports/runs/temporal_v3_paired_pilot/`; старые runs, архивы и manifests не перезаписываются.
- Validation уже участвовала в разведке: отчёт обозначается exploratory, а не окончательным доказательством generalization.

## Review Focus

1. Повреждённый архив, несовместимый config или checkpoint: явная остановка с именем поля/примера, без пропуска каскадов (Tasks 1, 4).
2. Пустой ранний снимок или каскад без роста после шага 1: нет фильтрации и исчезновения примеров (Tasks 1, 3).
3. Одинаковые scores, beta=0, candidate_count < predicted_count: детерминированный выбор и неизменный контроль (Task 2).
4. Возобновление с изменёнными входами, beta или code revision: нет молчаливого использования старых результатов (Task 4).
5. Colab после сброса runtime или с неудачной установкой: kernel import проверяется явно, ошибка setup не маскируется дальнейшим запуском (Task 5).

---

## Структура файлов и запуск разработки

Создать:

- `src/diffusion_sources/temporal_replay.py`: source-blind sampler и проверенный replay.
- `src/diffusion_sources/temporal_scoring.py`: scoring-only типы, inference records, correction и train-only выбор beta.
- `src/diffusion_sources/temporal_statistics.py`: парные метрики, bootstrap и gates.
- `src/diffusion_sources/temporal_pilot_artifacts.py`: hashes, manifests и безопасное сохранение новых результатов.
- `src/diffusion_sources/temporal_pilot_cli.py`: этапы smoke/select/validate/confirm.
- `scripts/evaluate_temporal_pilot.py`: тонкий wrapper main.
- `notebooks/colab_temporal_v3_pilot.ipynb`: новый пошаговый notebook без обучения.
- `tests/unit/test_temporal_replay.py`, `test_temporal_scoring.py`, `test_temporal_statistics.py`, `test_temporal_pilot_artifacts.py`, `test_temporal_pilot_cli.py`.
- `tests/smoke/test_temporal_pilot.py`, `test_temporal_notebook.py`.

Изменить только соответствующие разделы `docs/experiment_registry.md` и `development_log.md`. Не менять поведение существующих CLI и не добавлять зависимости в pyproject.toml. CLI вызывается через `python -m diffusion_sources.temporal_pilot_cli`, поэтому entry point в pyproject не нужен.

До исполнения прочитать spec и using-git-worktrees skill и создать изолированную рабочую копию. Существующие незакоммиченные документы в исходной копии сохранить; не переносить их автоматически. Данные использовать read-only из `D:/projects/diplom/diffusion-sources-localization/data/generated/facebook_main`. Локальный Python: `D:/projects/diplom/diffusion-sources-localization/.venv/Scripts/python.exe`. В рабочей копии выставить `$env:PYTHONPATH = Join-Path $PWD 'src'`, чтобы editable install не загрузил старую исходную копию.

Команда `python` ниже означает этот проверенный Python executable; в Colab — `sys.executable`. Коммиты включают только файлы завершённой задачи.

### Task 1: Source-blind sampling и точный replay

**Files:** Create `temporal_replay.py`; Test `tests/unit/test_temporal_replay.py`.

**Interfaces:**

```python
def sample_early_nodes(infected_by_t1: Iterable[int], fraction: float,
                       observation_seed: int, *, t1: int = 1) -> frozenset[int]: ...
def replay_early_mask(graph: nx.Graph, config: dict,
                      archive: Mapping[str, np.ndarray], index: int) -> np.ndarray: ...
```

Sampler не принимает источники/labels; replay возвращает только bool mask длины graph.number_of_nodes(). Входные archive-поля: features, candidate_masks, source_labels, infected_masks, source_counts, simulation_seeds, observation_seeds, probabilities, observation_fractions. Replay использует `SourceSampler`, `simulate_ic`, `observe_cascade` из существующего проекта; config читается из data-dir/config.yaml, не training config.

- [ ] **1. Написать failing tests sampler.**

```python
def test_source_blind_sampling_and_empty_snapshot():
    assert sample_early_nodes([], .5, 13) == frozenset()
    assert sample_early_nodes([4, 1, 3], 1., 13) == frozenset({1, 3, 4})
    assert sample_early_nodes([4, 1, 3], .5, 13) == sample_early_nodes([3, 4, 1], .5, 13)
    assert 'sources' not in inspect.signature(sample_early_nodes).parameters
    assert 'labels' not in inspect.signature(sample_early_nodes).parameters

@pytest.mark.parametrize('fraction,t1', [(0., 1), (1.1, 1), (.5, 0)])
def test_invalid_sampling_policy(fraction, t1):
    with pytest.raises(ValueError):
        sample_early_nodes([1], fraction, 13, t1=t1)
```

- [ ] **2. Написать fixture/replay tests и увидеть RED.** Создать local fixture конфигом ниже через generate_dataset; loading только graph/train, не существующие project test archives.

```python
config = {
    'graph': {'id': 'karate', 'kind': 'karate'},
    'simulation': {'source_counts': [1, 2, 3], 'probabilities': [.4],
                   'max_steps': 3, 'distance_ranges': [{'min': 1, 'max': 5}]},
    'observation': {'fractions': [1.], 'false_positive_count': 0},
    'dataset': {'seed': 13, 'splits': {'train': 6}, 'min_candidates': 3,
                'max_infected_fraction': .99, 'max_attempt_factor': 100},
}
```

Для всех 6 примеров assert mask shape/dtype и повторяемость. Parametrize повреждение одного элемента source_labels, infected_masks, candidate_masks, features[:, :, 0], source_counts, probabilities, observation_fractions: каждый replay обязан raise ValueError. Для каскада probability=0 создать archive вручную из SourceSampler + simulate_ic + observe_cascade (без фильтра min_candidates): assert ранняя маска существует, источники не маркируются отдельным полем, пример не отбрасывается. Проверить удалённое обязательное поле, неконтигуозные graph node IDs и unsupported max_steps/hidden-source protocol. Run `python -m pytest tests/unit/test_temporal_replay.py -q`; ожидается RED до реализации.

- [ ] **3. Реализовать sampler/replay.** Основная sampling-логика:

```python
nodes = np.asarray(sorted(set(infected_by_t1)), dtype=np.int64)
rng = np.random.default_rng(np.random.SeedSequence([5000000, observation_seed, t1]))
return frozenset(nodes[rng.random(len(nodes)) < fraction].tolist())
```

В replay построить `list(itertools.product(distance_ranges, probabilities, fractions, source_counts))`, выбрать по `index % len(conditions)`, проверить metadata, восстановить источники с RNG simulation_seed и вызвать simulate_ic с тем же RNG. Сверить источники/итоговое заражение и observe_cascade с observation_seed, включая features observed-mask. Любая разница -> ValueError с index и полем. Ранний infected set получить из infection_times <=1 внутри генератора; передать sampler только set/fraction/seed. Протокол оригинального Facebook с false_positive_count=0, hide_source_count=0, max_steps=3; иной протокол отклонять явно.

- [ ] **4. Увидеть GREEN и commit.** Run указанного test file; затем `git add` только module/test и `git commit -m "feat: add checked source-blind temporal replay"`.

### Task 2: Frozen inference и temporal score correction

**Files:** Create `temporal_scoring.py`; Test `tests/unit/test_temporal_scoring.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class CandidateScores:
    candidate_ids: tuple[int, ...]
    scores: tuple[float, ...]  # sigmoid scores, aligned with IDs
    early_observed: tuple[bool, ...]
    baseline_sources: frozenset[int]
    predicted_count: int  # original count-head argmax + 1

@dataclass(frozen=True)
class PilotRecord:
    index: int
    true_sources: frozenset[int]
    candidates: CandidateScores
    early_empty: bool  # whole early mask, not just its candidate intersection

def correct_sources(candidates: CandidateScores, beta: float) -> frozenset[int]: ...
def select_beta(records: Sequence[PilotRecord], grid: Sequence[float]) -> tuple[float, dict[float, float]]: ...
def collect_records(data_dir: Path, run_dir: Path, split: str,
                    indices: Sequence[int], device: torch.device) -> list[PilotRecord]: ...
```

correct_sources никогда не получает true_sources. collect_records разрешает только split train/validation; явные paths, без glob всех npz. FeatureBuilder и constructor model соответствуют `train_cli.py`; actual feature_names читаются из frozen config, не repo S1b-template. Shared source-head обязательна. Использовать saved `best_model.pt`, model.eval(), torch.inference_mode(); возвращать данные на CPU без сохранения GPU tensors между примерами.

- [ ] **1. Написать RED тесты correction и selection.**

```python
def test_correction_preserves_candidate_pool_and_count():
    c = CandidateScores((0, 1, 2), (.9, .8, .1), (False, False, True), frozenset({0}), 1)
    assert correct_sources(c, 0.) == c.baseline_sources
    assert correct_sources(c, 1.) == frozenset({2})
    assert len(correct_sources(c, 1.)) == 1

def test_train_beta_tie_uses_smaller_value():
    c = CandidateScores((0, 1), (.9, .1), (False, False), frozenset({0}), 1)
    beta, table = select_beta([PilotRecord(0, frozenset({0}), c, True)], [1., 0.])
    assert beta == 0.
    assert table == {1.: 1., 0.: 1.}
```

Добавить тест ties: adjusted score -> исходный score -> меньший ID; beta0 сохраняет baseline даже при ties; count3/candidates1 возвращает1; empty candidate pool/duplicate IDs/unequal lengths/NaN score/negative beta вызывают ValueError. Сделать spy для `np.load`/Path.open: collect_records(split='test')/('holdout') сразу падает до чтения файлов. Run `python -m pytest tests/unit/test_temporal_scoring.py -q`; RED.

- [ ] **2. Реализовать коррекцию и inference.**

```python
if beta == 0:
    return candidates.baseline_sources
ranked = sorted(zip(candidates.candidate_ids, candidates.scores, candidates.early_observed),
                key=lambda row: (-(row[1] + beta * int(row[2])), -row[1], row[0]))
return frozenset(row[0] for row in ranked[:min(candidates.predicted_count, len(ranked))])
```

select_beta считает mean(set_metrics(true_sources, correct_sources(...))['f1']) для каждой beta и выбирает максимум с tie на меньшую beta. collect_records вызывает replay_early_mask для выбранных индексов, load_pyg_split с limit=max(indices)+1, model + predict_joint; хранит sigmoid scores только для candidate_ids и original baseline_sources/source_count. Проверяет len(labels)>0, labels внутри candidate mask, selected indices unique/nonnegative/in-range. Distance cache путь берётся из frozen config; отсутствие cache не меняет признаки, но логируется.

- [ ] **3. GREEN и commit.** Run tests Task1+Task2. Commit только module/test: `feat: add frozen temporal score correction`.

### Task 3: Парные метрики, bootstrap и gates

**Files:** Create `temporal_statistics.py`; Test `tests/unit/test_temporal_statistics.py`.

**Interfaces:**

```python
def paired_bootstrap_ci(deltas: np.ndarray, true_k: np.ndarray,
                         *, repetitions: int = 2000, seed: int = 9282026) -> tuple[float, float]: ...
def evaluate_pairs(records: Sequence[PilotRecord], beta: float, graph: nx.Graph) -> dict: ...
def primary_gate(report: dict) -> dict: ...
def confirmation_gate(primary: dict, repeats: Sequence[dict]) -> dict: ...
```

Report schema: `baseline`, `temporal` содержат `all`, `by_k` (JSON string keys '1'/'2'/'3'), `by_candidates` ('1-10'/'11-20'/'21-50'/'51+'); `delta_f1`, `delta_by_k`, `f1_ci`, `early_coverage`, `rows`. Строка содержит index/k/candidate_count, true sources, обе predictions, обе set/distance/hit metrics. ранний source recall — pooled по истинным источникам; all-source coverage — mean по каскадам. Empty group -> n=0 и null метрики, не деление на ноль.

- [ ] **1. RED тесты статистики.**

```python
def test_constant_paired_delta_has_exact_interval():
    delta = np.full(6, .125)
    assert paired_bootstrap_ci(delta, np.array([1, 1, 2, 2, 3, 3])) == (.125, .125)

def test_zero_delta_is_not_primary_pass():
    report = {'delta_f1': 0., 'f1_ci': [0., 0.],
              'delta_by_k': {'1': 0., '2': 0., '3': 0.}, 'count_unchanged': True}
    assert not primary_gate(report)['passed']
```

Добавить воспроизводимость bootstrap, shape/NaN/empty input errors, equal records baseline=temporal, low CI lower rejection, k2 или k3=-.021 rejection, count changed rejection; отсутствие k2/k3 в primary gate -> explicit insufficient-data fail. Test confirmation: primary pass + deltas .03/.03 -> pass; delta0 или missing repeat -> fail. Geometry tests на path_graph используют существующие set_metrics/source_set_distances/source_radius_hits и сравнивают конкретные ожидаемые значения. Early empty не удаляет record. Run test file; RED.

- [ ] **2. Реализовать pair aggregation/bootstrap.**

```python
rng = np.random.default_rng(seed)
strata = [np.flatnonzero(true_k == k) for k in sorted(set(true_k.tolist()))]
means = [deltas[np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])].mean()
         for _ in range(repetitions)]
lower, upper = np.quantile(means, [.025, .975])
```

Для predictions использовать Task2, для метрик existing `metrics.py`. Assert count accuracy unchanged; нарушение — stop, не только warning. primary_gate возвращает passed и список reasons по точным порогам spec. confirmation_gate принимает primary pass и ровно2 reports, требует delta>0 для всех3 и mean delta>=.02; beta/mask identity checks принадлежат runner Task4.

- [ ] **3. GREEN и commit.** Run Task1–3 tests. Commit `feat: add paired temporal metrics and decision gates`.

### Task 4: CLI stages и безопасные артефакты

**Files:** Create `temporal_pilot_artifacts.py`, `temporal_pilot_cli.py`, `scripts/evaluate_temporal_pilot.py`; Test `tests/unit/test_temporal_pilot_artifacts.py`, `test_temporal_pilot_cli.py`, `tests/smoke/test_temporal_pilot.py`.

**Interfaces:**

```python
def sha256_file(path: Path) -> str: ...
def write_stage(output_dir: Path, stage: str, manifest: dict, payload: dict) -> None: ...
def read_stage(output_dir: Path, stage: str, expected_identity: dict) -> tuple[dict, dict]: ...
def smoke(data_dir: Path, run_dir: Path, output_dir: Path, device: torch.device) -> dict: ...
def select(data_dir: Path, run_dir: Path, output_dir: Path, device: torch.device) -> dict: ...
def validate(data_dir: Path, run_dir: Path, output_dir: Path, device: torch.device) -> dict: ...
def confirm(data_dir: Path, run_dir: Path, output_dir: Path,
             repeat_runs: Sequence[Path], device: torch.device) -> dict: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

CLI: positional stage smoke/select/validate/confirm, required --data-dir/--run-dir/--output-dir, --device (cpu/cuda); confirm adds repeatable --repeat-run exactly2. No arbitrary beta flag, no arbitrary split flag, no training command. Output directories stages smoke/select/validation/seed_7027/seed_7028/confirmation under new pilot root. Every stage payload JSON + manifest JSON; pair reports additionally predictions.csv (sorted source IDs in JSON strings) and metrics.json. Intermediate masks need not be cached: reproducible RNG suffices; no mandatory cache feature in this first implementation.

- [ ] **1. RED tests artifact safeguards.**

```python
def test_changed_identity_is_not_silently_resumed(tmp_path):
    write_stage(tmp_path, 'select', {'identity': {'config': 'a'}}, {'beta': .25})
    with pytest.raises(ValueError, match='identity'):
        read_stage(tmp_path, 'select', {'config': 'b'})

def test_existing_results_are_not_overwritten(tmp_path):
    write_stage(tmp_path, 'select', {'identity': {}}, {'beta': .25})
    with pytest.raises(FileExistsError):
        write_stage(tmp_path, 'select', {'identity': {}}, {'beta': .5})
```

Также malformed/missing manifest, modified payload hash, partial interrupted stage, output_dir resolving inside input data/run, symlink resolving into input root -> error. Stage writes через tempfile в output parent, manifest/payload hashes, completion marker last; на partial stage — явная ошибка с просьбой выбрать новый output path, без recursive deletion. Re-run complete stage разрешён только read_stage по совпадающей identity. Identity includes graph/config/split/checkpoint/training-config SHA256, code revision + hash новых module files (dirty revision недостаточно), policy version/RNG/beta/train indices. В select identity нет validation hash: selection вообще не открывает validation.

- [ ] **2. RED tests stage isolation и gates.** Monkeypatch collect_records/select_beta/evaluate_pairs чистыми fixtures из Task2/3, файл-reader записывает opened paths. assert select читает graph/config/train/checkpoint, но не validation/test/holdout; validate не вызывает select_beta и использует сохранённый beta; confirm не делает train tuning, отказывает без primary pass/двух runs/изменения masks policy. На изменённый baseline F1 более1e-6 validate обязан остановиться без completed report. Реальный fixture smoke generate_dataset only train6, создаёт random initialized JointSourceCountGCN checkpoint/config; torch.save state_dict, никаких optimizer/training loops. assert smoke exits0, пишет manifest, не создаёт test.npz. Run все Task4 test files; RED.

- [ ] **3. Реализовать coordinator и CLI.**

```python
# smoke: six train examples, no validation; replay correctness + inference + beta0 equality + timing
records = collect_records(data_dir, run_dir, 'train', list(range(6)), device)
# select: verify first540 metadata gives ten of each54 original conditions, then tune only here
records = collect_records(data_dir, run_dir, 'train', list(range(540)), device)
beta, table = select_beta(records, [0., .1, .25, .5, 1.])
# validate: read locked selection manifest; read only validation, full count1998
report = evaluate_pairs(records, beta, graph)
expected = saved_metrics['validation_prediction_metrics']['joint_estimated_k']['all']['f1']
if abs(report['baseline']['all']['f1'] - expected) > 1e-6:
    raise ValueError('baseline F1 does not match saved checkpoint metrics')
```

Использовать saved metrics контроль для всех3 seeds, для primary7026 дополнительно требовать expected близко .362996329663. Confirm run configs одинаковы по model/data/loss (кроме runtime paths и training seed) и все frozen checkpoint fingerprints записаны. Train per-example predictions если существуют читать по индексам для контрольной проверки, иначе явно note unavailable, не сравнивать со full train metrics. Stage APIs возвращают summary dict; CLI ловит ошибки, печатает concise error, возвращает nonzero. Wrapper:

```python
from diffusion_sources.temporal_pilot_cli import main
if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **4. GREEN и commit.** Run all temporal unit+smoke tests. `python -m diffusion_sources.temporal_pilot_cli --help` показывает четыре stages. Commit `feat: add staged temporal pilot runner and manifests`.

### Task 5: Colab notebook, документация и итоговая проверка

**Files:** Create `notebooks/colab_temporal_v3_pilot.ipynb`, `tests/smoke/test_temporal_notebook.py`; Modify status sections `docs/experiment_registry.md`, `development_log.md`.

**Interfaces:** Notebook вызывает CLI Task4 через subprocess с `check=True`; model/checkpoint пути соответствуют Drive `reports/runs/facebook_main_v2/s1b/seed_7026` и `frozen_candidate/seed_7027/7028`. Код проекта остаётся `/content/diffusion-sources`; новый pilot output — `/content/drive/MyDrive/diffusion-sources/reports/runs/temporal_v3_paired_pilot`.

- [ ] **1. RED notebook contract test.** Load JSON, assert nbformat4, outputs empty. Python cells без магических команд compile(); tests assert наличие отдельных named cells setup/smoke/select/validate/confirm, subprocess check=True, отсутствие run_training_job/run_frozen_seed/optimizer.step/test.npz, установка через текущий sys.executable и явный src в sys.path. Проверить что confirm cell читает gate и пропускается при fail; assert setup импортирует diffusion_sources в kernel после install. Run `python -m pytest tests/smoke/test_temporal_notebook.py -q`; RED.

- [ ] **2. Создать notebook с семью последовательно пронумерованными кодовыми ячейками.**

1. Drive mount + Path constants DATA_DIR, RUN_DIR, PILOT_DIR, REPO; не создавать/изменять старые runs.
2. GPU/runtime диагностика, clone/pull --ff-only через subprocess(check=True); pip --no-deps editable через sys.executable; сначала необходимые existing project dependencies. print stages, error abort. `sys.path.insert(0, str(REPO/'src'))`, import diffusion_sources и print __file__. Не принуждать restart если импорт уже работает.
3. `--help` + paths + primary saved metrics/config hashes; проверка доступности graph/config/train/checkpoint без test/holdout. Validation count проверяется лишь validation-stage, не во время train-selection.
4. CLI smoke на6 train, print timing; просьба прислать вывод при failure, не запускать следующую ячейку.
5. CLI select540 train; print выбранный beta и manifest path.
6. CLI validate full1998; print baseline/temporal F1, delta, CI, k strata, gate. При fail остановка до confirm.
7. CLI confirm7027/7028 только после primary pass, без обучения/нового tuning. Print final decision.

Общий вызов с unbuffered stdout:

```python
subprocess.run([sys.executable, '-u', '-m', 'diffusion_sources.temporal_pilot_cli',
                'smoke', '--data-dir', str(DATA_DIR), '--run-dir', str(RUN_DIR),
                '--output-dir', str(PILOT_DIR), '--device', 'cuda'],
               cwd=REPO, check=True)
```

Markdown предупреждает: НЕ Run all, ни одна ячейка не обучает; независимые выборки не являются накопительным наблюдением; весь пилот exploratory; нет прямого сравнения с исходным overall test. Не менять существующие notebooks.

- [ ] **3. Обновить журнал/registry статусом implemented-not-run.** Записать реальные локальные проверки, не выдумывать GPU/validation metrics. Только после полученного от пользователя отчёта фиксировать F1/gate. Перед интеграцией перенести изменения журналов с учётом текущих пользовательских edits, не перезаписывать файлы целиком.

- [ ] **4. Финальная локальная проверка.** Run `python -m pytest -q`, `python -m compileall -q src scripts`, `git diff --check`. Выполнить реальный CLI smoke на первых6 train Facebook с доступным локальным shared checkpoint/config (S1b если доступен; иначе synthetic integration и отдельно replay six Facebook, явно сообщить отсутствие frozen checkpoint). Не загружать project test.npz. Проверить by-hash неизменность входных архивов/checkpoint и отсутствие новых test predictions. Сохранить raw output тестов/timing в handoff.

- [ ] **5. Commit и review.** Commit только notebook/test/documentation edits задачи. Проверить diff всей ветки против spec, отдельно source-blind sampling, beta0 identity и stage read guards. Следовать выбранному пользователем способу исполнения/review и verification-before-completion. Не запускать push автоматически: после локальной проверки согласовать публикацию изменений, необходимую для Colab git pull.

- [ ] **6. Передать одну первую инструкцию пользователю.** Указать новый notebook и точную setup-ячейку. Не отправлять сразу все команды, не просить повторно обучать frozen seeds. Следующая инструкция выдаётся после подтверждения успешного setup/smoke.

## Самопроверка покрытия плана

Replay/source-blind observation covered Task1; immutable candidate/count/beta selection Task2; metrics/strata/CI/gates Task3; manifests/hashes/checkpoint validation/read isolation/resume Task4; kernel import/one-cell instructions/documentation/no training Task5. Все пять Review Focus имеют конкретные тесты. Типы CandidateScores/PilotRecord и dict report-schema едины для consumers. Необязательный cache исключён из MVP; независимый новый temporal-датасет и модель явно вне текущего scope.

## Перед исполнением

Этот план ожидает просмотра пользователя и выбора способа выполнения. Рекомендуется native: основной агент реализует в этой задаче по шагам, поскольку все пять частей зависят от одних и тех же интерфейсов и цель — экономный пилот. Независимая финальная проверка выполняется по выбранному workflow; subagent-driven альтернатива требует отдельного согласования. До согласования плана реализацию не начинать.
