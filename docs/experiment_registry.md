# Реестр версий и результатов экспериментов

Этот документ задает правила работы с текущей моделью и ее улучшенными версиями.

## Новый exploratory пилот: temporal_v3_learned_reranker/v1

Три лёгких LogisticRegression C=0.1/1/10 поверх frozen S1b, шесть наблюдаемых
признаков; count-head и candidate set неизменны. Контроль — текущая Temporal-v3
beta=0.5. Fit 1620 + selection-dev 540 целых train-каскадов; validation 1998 и
frozen repeats разрешаются только gates. Scaler fit-only; выбранный head без refit.
Независимая оценка и интеграция в UI не разрешаются автоматически.

[Notebook](../notebooks/colab_temporal_learned_reranker.ipynb),
[технический статус](temporal_learned_reranker_status.md).
Новых реальных метрик качества нет: выполнен только CPU smoke шести train-примеров
без обучения (9.797 s), защищённые входы и manifest сохранили SHA256.
Артефакты smoke: `reports/runs/temporal_v3_learned_reranker/local_smoke_v2/`.
Рабочий `v1` пока не запускался. Это черновая экспериментальная ветка, не чистовик.

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

### Фактические GPU-абляции snapshot-v2

Все значения ниже получены только на validation в режиме `estimated-k`; test и final holdout не открывались.

| ID | Parent | Seed | Macro F1 | F1, k=1 | Count accuracy | Exact | Distance | Hit@1-hop | Params | Решение |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `S0` | `v1 Joint` | 7026 | 0.352903 | 0.130881 | 0.629630 | 0.068068 | 1.130422 | 0.533951 | 21 060 | `accepted` как контроль |
| `S1a` | `S0` | 7026 | 0.353837 | 0.119620 | 0.703203 | 0.072072 | 1.135552 | 0.510427 | 21 380 | `rejected`: ΔF1=+0.000934 < 0.005 |
| `S1b` | `S0` | 7026 | 0.362996 | 0.145646 | 0.741241 | 0.076577 | 1.100225 | 0.556056 | 21 252 | `accepted`: ΔF1=+0.010093 |
| `S1c` | `S1b` | 7026 | 0.364114 | 0.120120 | 0.668669 | 0.071071 | 1.153445 | 0.480230 | 21 508 | `rejected`: ΔF1=+0.001118 < 0.005 |
| `S2` | `S1b` | 7026 | 0.357090 | 0.117868 | 0.631632 | 0.066567 | 1.151777 | 0.510344 | 41 796 | `rejected`: ΔF1=-0.005906 |
| `S3` | `S1b` | 7026 | 0.352886 | 0.104855 | 0.648649 | 0.064064 | 1.167751 | 0.480147 | 25 412 | `rejected`: ΔF1=-0.010110 |
| `S4` | `S1b` | 7026 | 0.354121 | 0.135135 | 0.746246 | 0.082082 | 1.134885 | 0.524858 | 21 382 | `rejected`: ΔF1=-0.008876 |
| `S5_L01_N8_H05` | `S1b`, `lambda_rank=0` | 7026 | 0.362262 | 0.135135 | 0.740240 | 0.074575 | 1.092634 | 0.566483 | 21 252 | не выбран: ΔF1=-0.000734 |
| `S5_L02_N8_H05` | `S1b`, `lambda_rank=0` | 7026 | 0.362246 | 0.139890 | 0.767768 | 0.081582 | 1.094553 | 0.553136 | 21 252 | не выбран: ΔF1=-0.000751 |
| `S5_L02_N4_H05` | `S1b`, `lambda_rank=0` | 7026 | 0.359426 | 0.137638 | 0.732232 | 0.071572 | 1.096013 | 0.568151 | 21 252 | не выбран: ΔF1=-0.003570 |
| `S5_L02_N8_H075` | `S1b`, `lambda_rank=0` | 7026 | 0.362729 | 0.139890 | 0.749249 | 0.074575 | 1.106607 | 0.549466 | 21 252 | не выбран: ΔF1=-0.000267 |
| `S6` | `S1b` | 7026 | 0.361628 | 0.141391 | 0.729730 | 0.077077 | 1.090507 | 0.572322 | 21 316 | `rejected`: ΔF1=-0.001368 |

`S1a` повысил count accuracy на `0.073574` и exact accuracy на `0.004004`, но основной F1 практически не изменился, F1 для `k=1` снизился на `0.011261`, distance ухудшился на `0.005130`, Hit@1-hop — на `0.023524`, а число параметров выросло на 320. По заранее зафиксированному правилу при `|ΔF1| < 0.005` оставлен более простой `S0`; локальные structural features не переносятся в `S1b`.

`S1b`, построенный непосредственно от `S0`, повысил macro F1 на `0.010093`, F1 для `k=1` на `0.014765`, count accuracy на `0.111612`, exact accuracy на `0.008509` и Hit@1-hop на `0.022105`; symmetric distance уменьшился на `0.030197`. Прирост основного score превышает порог `0.005`, остальные метрики согласованно улучшились, цена составляет 192 дополнительных параметра. Distance/position features приняты, текущий победитель перед `S1c` — `S1b`.

`S1c` дал лишь `ΔF1=+0.001118` относительно `S1b`, при этом F1 для `k=1` снизился на `0.025526`, count accuracy на `0.072573`, exact accuracy на `0.005506`, Hit@1-hop на `0.075826`; symmetric distance вырос на `0.053220`, число параметров — на 256. По правилу простоты и всем tie-breakers global scalar features отклонены и не переносятся в `S2`; текущий победитель остаётся `S1b`.

`S2` с global-context source-head снизил macro F1 относительно `S1b` на `0.005906`, F1 для `k=1` на `0.027778`, count accuracy на `0.109610`, exact accuracy на `0.010010` и Hit@1-hop на `0.045712`; symmetric distance вырос на `0.051552`. Число параметров увеличилось с `21 252` до `41 796`. Global-context head отклонена, текущим победителем перед `S3` остаётся `S1b`.

`S3` с трёхслойным residual GCN снизил macro F1 относительно `S1b` на `0.010110`, F1 для `k=1` на `0.040791`, count accuracy на `0.092593`, exact accuracy на `0.012513` и Hit@1-hop на `0.075909`; symmetric distance вырос на `0.067526`. Число параметров увеличилось на 4 160. Residual backbone отклонён, текущим победителем перед `S4` остаётся двухслойный `S1b`.

`S4` со специализированными source-head для `k=1/2/3` снизил macro F1 относительно `S1b` на `0.008876`, F1 для `k=1` на `0.010511` и Hit@1-hop на `0.031198`; symmetric distance вырос на `0.034660`. Count accuracy выросла на `0.005005`, exact accuracy — на `0.005506`, но эти вторичные улучшения не компенсируют падение основного score. `S4` отклонён; parent-control для ограниченной сетки `S5` — текущий победитель `S1b` с `lambda_rank=0`.

Первый вариант S5 (`lambda_rank=0.1`, 8 negatives, hard fraction `0.5`) обучался 166,9 минуты. Macro F1 снизился на `0.000734`, F1 для `k=1` — на `0.010511`, count accuracy — на `0.001001`, exact accuracy — на `0.002002`. Distance улучшился на `0.007591`, Hit@1-hop — на `0.010427`, параметры не изменились. По основному score и правилу простоты этот вариант не выбирается; решение о ranking-loss целиком остаётся открытым до завершения заранее фиксированной S5-grid.

Второй вариант S5 (`lambda_rank=0.2`, 8 negatives, hard fraction `0.5`) завершён за 72,7 минуты. Macro F1 снизился на `0.000751`, F1 для `k=1` — на `0.005756`, Hit@1-hop — на `0.002920`. Count accuracy выросла на `0.026527`, exact accuracy на `0.005005`, symmetric distance улучшился на `0.005672`. При основном score ниже контроля вариант не выбирается; различие длительности с первым запуском не интерпретируется как ускорение конфигурации, поскольку число завершённых эпох определяется early stopping.

Третий и четвёртый варианты S5 также не превысили `S1b`: `S5_L02_N4_H05` дал `ΔF1=-0.003570`, а `S5_L02_N8_H075` — `ΔF1=-0.000267`. Ограниченная сетка завершена без принятого ranking-варианта; отдельные улучшения некоторых вторичных метрик не компенсируют отсутствие прироста основного score. Ranking-loss отклонён целиком, победителем остаётся `S1b`.

`S6` с нормированным Multi-Jordan rank дал `ΔF1=-0.001368` относительно `S1b`. F1 для `k=1` снизился на `0.004254`, count accuracy — на `0.011512`, параметров стало на 64 больше. При этом distance улучшился на `0.009718`, Hit@1-hop — на `0.016266`, exact accuracy — на `0.000501`. Эти диагностические улучшения не преодолевают основной критерий; по правилу простоты `S6` отклонён, текущим победителем остаётся `S1b`.

После freeze `S1b` повторен на seed `7027` (тот же validation split, отдельное обучение): F1 `0.353387`, F1 при `k=1/2/3` `0.120871/0.412412/0.526877`, count accuracy `0.649149`, exact set accuracy `0.062563`, distance `1.153987`, Hit@1-hop `0.515098`, Hit@2-hop `0.927344`. Относительно seed `7026` F1 ниже на `0.009610`, count accuracy — на `0.092092`; это демонстрирует необходимость проверки seed-вариации. Сравнение с `S0` на другом seed не является парным контролем. Test и final holdout не открывались.

Третий seed замороженного `S1b`, `7028`, дал validation F1 `0.359243`, count accuracy `0.743243`, exact set accuracy `0.080080`, distance `1.120412`, Hit@1-hop `0.531031`, Hit@2-hop `0.946113`; oracle-k F1 `0.367284`. Итог по `7026/7027/7028`: F1 `0.358542 ± 0.004843`, count accuracy `0.711211 ± 0.053757`, exact set accuracy `0.073073 ± 0.009269`, distance `1.124875 ± 0.027157`, Hit@1-hop `0.534062 ± 0.020646`, Hit@2-hop `0.939217 ± 0.010327` (среднее ± выборочное SD). Сравнение с `S0` пока основано только на development seed `7026`; для оценки устойчивого эффекта компонента нужны согласованные контрольные seed или осторожная формулировка без такого утверждения. Test/final holdout закрыты.

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

## Temporal-v3 paired pilot — 2026-09-28

Реализован отдельный exploratory pipeline без обучения: дополнительный source-blind ранний снимок t=1, тот же финальный snapshot и candidate mask frozen S1b. Поправка scores выбирается только на 540 сбалансированных train-примерах по beta grid 0/0.1/0.25/0.5/1.0; paired validation1998, bootstrapCI и gate deltaF1>=0.02 определяют необходимость inference-подтверждения на7027/7028. Это проверка дополнительной информации, не архитектурное улучшение при одинаковом входе.

Статус: implementation интегрирован в master 2026-09-28 по разрешению пользователя; независимый final review завершён, два важных замечания исправлены (read-only/hash-checked distance cache и полное stage-flow покрытие). Итоговый полный набор181 tests passed; реальный exact replay проверен на6 train-примерах. Colab inference ещё не выполнен. Нет новых F1 или обученных temporal checkpoints. Test/final holdout закрыты. Подробности в `docs/temporal_v3_pilot_status.md` и новом notebook `notebooks/colab_temporal_v3_pilot.ipynb`.

Обновление после Colab: пользователь прислал сохранённые отчёты всех стадий. Beta0.5 выбран на540 train; frozen7026/7027/7028 оценены на том же validation1998 без обучения. Snapshot F1=0.362996/0.353387/0.359243; temporal correction F1=0.538906/0.537421/0.539439. Mean±sample SD=0.358542±0.004843→0.538589±0.001046, mean delta+0.180047. Все paired bootstrap95% CI положительны; primary/confirmation passed=true, count unchanged. Прирост за счёт дополнительного раннего входа, а не замены архитектуры; exploratory validation, не независимый final test. Числа зафиксированы из user-provided stage JSON, manifests/CSV отдельно не сверены. Полная запись в разделе42 development_log.
