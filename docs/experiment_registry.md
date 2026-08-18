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

Диагностика `v1` должна находиться в:

```text
reports/diagnostics/facebook_main_v1/
```

Нельзя направлять новые обучения или диагностику в каталоги `diffusion-sources/reports/no_consistency`, `diffusion-sources/reports/node_only`, `reports/series/facebook_main` и `reports/figures/facebook_main`.

## Версия v2: улучшенная модель

`v2` создается только после завершения диагностики `v1`. Планируемые изменения:

1. структурные признаки наблюдаемого каскада;
2. третий residual GCN-слой;
3. global-context узловая голова;
4. затем отдельная проверка ranking-loss.

Планируемые каталоги:

```text
reports/diagnostics/facebook_main_v2/
reports/series/facebook_main_v2/
reports/runs/facebook_main_v2/
```

Конфигурации v2 должны иметь отдельные имена:

```text
configs/train_facebook_v2_node.yaml
configs/train_facebook_v2_joint.yaml
configs/train_facebook_v2_no_ranking.yaml
configs/train_facebook_v2_ranking.yaml
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
- `test_predictions.csv`.

## Сравнение v1 и v2

Test не используется для выбора v2. Порядок:

1. обучить v2 на train;
2. выбрать архитектуру и параметры по validation;
3. зафиксировать победившую конфигурацию;
4. повторить ее для трех seed;
5. один раз сравнить v1 и v2 на test;
6. сохранить обе версии в итоговом отчете.

Основная таблица должна содержать минимум:

```text
v1 Joint
v1 Node-only
v2 Node-only
v2 Joint
Uniform
Degree
Multi-Jordan
```

## Текущая точка продолжения

Перед созданием v2 нужно завершить диагностику v1:

- confusion matrix для `k`;
- bootstrap CI по тестовым каскадам;
- качество по размеру candidate set;
- визуальные примеры удачных и ошибочных каскадов;
- проверка текущего checkpoint в Streamlit.

Только после этой диагностики разрешается создавать `configs/*_v2.yaml` и запускать новые обучения.
