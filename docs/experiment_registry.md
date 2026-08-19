# Реестр версий и результатов экспериментов

Этот документ задает правила работы с текущей моделью и ее улучшенными версиями.

## Главное правило

Артефакты `v1` не изменяются и не перезаписываются. Любая новая архитектура, конфигурация, диагностика или серия seed получает отдельный каталог и отдельное имя версии.

## Версия v1: текущая baseline-модель

`v1` - фактически завершенная версия проекта:

- двухслойный GCN;
- признаки `[observed_infected, log_degree_normalized]`;
- Joint без consistency-loss как основная рабочая конфигурация;
- Node-only как нейросетевой baseline;
- seed `7026`, `7027`, `7028`;
- Facebook train/validation/test без изменения протокола.

Импортированные из Colab результаты находятся в:

```text
diffusion-sources/reports/
├── joint_full/seed_7026/
├── tuning/joint_consistency_001/seed_7026/
├── no_consistency/seed_7026/
├── no_consistency/seed_7027/
├── no_consistency/seed_7028/
├── node_only/seed_7026/
├── node_only/seed_7027/
└── node_only/seed_7028/
```

Агрегаты и графики `v1` находятся в:

```text
reports/series/facebook_main/
reports/figures/facebook_main/
```

Диагностика `v1` находится в:

```text
reports/diagnostics/facebook_main_v1/
```

Нельзя направлять новые обучения или диагностику в каталоги `diffusion-sources/reports/no_consistency`, `diffusion-sources/reports/node_only`, `reports/series/facebook_main` и `reports/figures/facebook_main`.

## Версия snapshot-v2: улучшенная модель одного снимка

`snapshot-v2` сохраняет исходную постановку и тот же Facebook train/validation/test. Изменения проверяются последовательно и по одному:

1. группы структурных признаков;
2. global-context голова;
3. третий residual GCN-слой;
4. специализированные source-head для `k=1/2/3`;
5. ranking-loss с hard negatives;
6. нормированный Jordan-признак;
7. safe shortlist с контролем candidate recall.

После последовательного отбора для финального кандидата обязательны leave-one-component-out абляции. Полный протокол, критерии выбора и критерии успеха зафиксированы в `docs/model_improvement_plan.md`.

Планируемые каталоги:

```text
reports/diagnostics/facebook_main_v2/
reports/series/facebook_main_v2/
reports/runs/facebook_main_v2/
data/generated/facebook_snapshot_final_holdout/
```

Отслеживаемый manifest закрытого набора: `configs/holdouts/facebook_snapshot_final_holdout_manifest.json`.

Конфигурации v2 должны иметь отдельные имена:

```text
configs/snapshot_v2/ablations/*.yaml
configs/snapshot_v2/final.yaml
configs/snapshot_v2/final_holdout.yaml
```

Нельзя заменять существующие `configs/train_facebook.yaml`, `configs/train_node_facebook.yaml` или `configs/train_facebook_no_consistency.yaml`: они описывают v1 и нужны для воспроизводимого сравнения.

## Правило именования запуска

Каталог запуска должен явно содержать версию, вариант и seed:

```text
reports/runs/facebook_main_v2/joint_context/seed_7026/
reports/runs/facebook_main_v2/joint_context/seed_7027/
reports/runs/facebook_main_v2/joint_context/seed_7028/
```

Внутри каждого запуска сохраняются:

- `config.yaml`;
- `best_model.pt`;
- `last_checkpoint.pt`;
- `history.csv`;
- `history.json`;
- `metrics.json`;
- `test_predictions.csv` только после freeze кандидата и разрешенной test-оценки.

У exploratory-запусков `S0-S7` с `evaluation.evaluate_test: false` файл `test_predictions.csv` должен отсутствовать; это является частью проверки test-lock, а не неполным артефактом запуска.

Для каждого абляционного запуска дополнительно фиксируются parent-вариант, единственное измененное условие, Git revision, seed, config checksum и решение `accepted/rejected` с validation-метриками.

## Версия temporal-v2: отдельный временной режим

`temporal-v2` не является прямой заменой `snapshot-v2`. Он использует 2-3 неполных снимка после начала распространения и получает отдельные datasets, splits, holdout, конфигурации и результаты:

```text
data/generated/facebook_temporal_v2/
configs/temporal_v2/
reports/runs/facebook_temporal_v2/
reports/series/facebook_temporal_v2/
reports/holdout/facebook_temporal_v2/
```

Temporal-результаты сравниваются с лучшей snapshot-моделью на последнем снимке того же temporal-протокола. Их нельзя подставлять в таблицу исходного snapshot test как улучшение архитектуры при одинаковом входе.

## Закрытый final holdout

До реализации `snapshot-v2` создается `snapshot_final_holdout` с непересекающимися seeds. До freeze финальной конфигурации запрещены просмотр labels/агрегатов, tuning и выбор checkpoint по holdout. Перед единственным открытием фиксируются:

- manifest и checksum holdout;
- Git revision;
- config и checkpoint checksums;
- точная команда/скрипт оценки.

Существующий test остается сравнительным набором, поскольку его диагностика уже повлияла на постановку гипотез `v2`. Final holdout является подтверждающим набором. После его открытия дальнейшая настройка этой версии запрещена.

## Сравнение v1 и v2

Test не используется для выбора v2. Порядок:

1. обучить v2 на train;
2. выбрать архитектуру и параметры по validation;
3. зафиксировать победившую конфигурацию;
4. повторить ее для трех seed;
5. один раз сравнить v1 и snapshot-v2 на существующем test;
6. после фиксации checksums один раз оценить обе версии на final holdout;
7. сохранить обе версии и paired bootstrap CI разности в итоговом отчете.

Основная таблица должна содержать минимум:

```text
v1 Joint
v1 Node-only
snapshot-v2
Uniform
Degree
Multi-Jordan
```

Обязательные метрики включают F1, exact set accuracy, count accuracy/MAE, symmetric graph distance, `Hit@1-hop`, `Hit@2-hop`, разрезы по `k` и размеру candidate set. Для shortlist дополнительно сохраняются candidate recall и latency.

## Текущая точка продолжения

Диагностика `v1` уже сформировала confusion matrix, bootstrap CI и анализ размера candidate set; Facebook checkpoint проверен в Streamlit. Первый инфраструктурный этап `snapshot-v2` начат:

1. `Hit@1-hop` и `Hit@2-hop` добавлены в общий evaluation pipeline; переоценка checkpoints `v1` завершена без обучения и без изменения старых CSV, результаты находятся в `reports/diagnostics/facebook_main_v1/hop_metrics`;
2. `snapshot_final_holdout` из 1 998 примеров создан и запечатан, seeds не пересекаются с `v1`, manifest/checksums зафиксированы;
3. добавлены test-lock для абляций, контрольная конфигурация `S0` и первая конфигурация `S1a`;
4. реализовано именованное вычисление локальных structural features и выполнен локальный smoke-run без доступа к test;
5. для `S1b` реализованы mean/max distance и induced eccentricity, shortest-path cache защищен fingerprint топологии; два вырожденных boundary-признака отклонены по feature-only диагностике train/validation;
6. для `S1c` проверены шесть global scalars: candidate count дублировал observed count, observed/candidate fraction был константой; итоговая конфигурация оставляет размер, плотность, число компонент и долю крупнейшей компоненты;
7. для `S2` реализована отдельная global-context source-head: candidate embedding + observed mean/max + candidate mean/max + четыре global scalars; backbone и count-head сохранены как в родительском варианте;
8. для `S3` реализован трехслойный residual GCN: первый слой формирует hidden representation, второй и третий используют additive skip-связи; остальные компоненты `S2` сохранены;
9. для `S4` реализованы три source-head для `k=1/2/3`; train-loss выбирает голову по истинному `k`, estimated inference — по count-head, oracle inference — по заданному `k`;
10. для `S5` реализован pairwise ranking-loss только внутри графа с bounded negative sampling и hard negatives; `lambda_rank=0` полностью пропускает ranking-ветку и является обязательным контролем;
11. для `S6` реализован полный нормированный greedy Multi-Jordan rank кандидатов; top-`k` совпадает с baseline без использования target `k`, а избыточный center-score отклонён до GPU;
12. для `S7` реализована detached preliminary-head, top-`M` ranking с полным fallback и validation-only eligibility по micro/per-k recall, bootstrap CI, F1 и latency;
13. feature-only диагностика Jordan rank и локальные smoke-run `S2-S7` прошли с закрытым test; все компоненты готовы локально;
14. создан `notebooks/colab_training_snapshot_v2.ipynb` с persisted state, последовательными GPU-абляциями, S5-grid, S7 gates, LOO и freeze; полноценные GPU-запуски и визуальные примеры ошибок еще не выполнены.

Полное GPU-обучение `snapshot-v2` еще не начиналось. Для него подготовлен отдельный `notebooks/colab_training_snapshot_v2.ipynb`. Старые каталоги и notebook `v1` не изменяют назначение и не перезаписываются.
