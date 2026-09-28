# Temporal-v3: независимая оценка frozen метода без обучения

Дата: 2026-09-28. Статус: письменный дизайн одобрен пользователем сообщением «го».
Пользователь одобрил направление: новые 1998 каскадов, те же три frozen модели,
без переобучения, затем оформление результатов. Этот документ конкретизирует
правила до реализации. Старый test и snapshot holdout не оцениваются.

## Цель и границы

Проверить, сохраняется ли обнаруженное на exploratory validation преимущество
S1b с дополнительным ранним наблюдением на новых IC-каскадах того же графа.
Это не перенос на новую сеть, не новый temporal encoder и не проверка реальных
историй публикаций. Пользователь не хочет долгих обучений ради малых приростов;
требуются отдельные понятные Colab-ячейки и журнал разработки.

Подбор модели прекращён: S1b checkpoints seeds 7026/7027/7028, beta=0.5,
t1=1, max_steps=3, source-blind sampling и исходные признаки/count-head.
Все три checkpoints оцениваются; нельзя выбрать лучший seed после оценки.
Это три оценки на одних новых каскадах, не 5994 независимых примера и не ансамбль.
Текущие exploratory показатели: snapshot F1 0.358542, early-only expected F1
0.448700, temporal F1 0.538589. Они не задают обязательный абсолютный результат
на новой выборке и не используются для выбора новых параметров.

## Выбранный путь

Использовать отдельный новый набор каскадов. Старый test участвовал в v1
диагностике; sealed snapshot_final_holdout зарезервирован для snapshot-вариантов.
Переиспользование test не даёт полностью слепой проверки; открытие snapshot
holdout сейчас тратит его прежнее назначение. Новая выборка сохраняет оба.
Новый набор не требует нового обучения или изменения распределения задачи.

## Точная генерация и сохранение

- Graph: ego_facebook из прежнего facebook_combined.txt.gz; после генерации
  topology fingerprint и node mapping должны совпасть с graph.npz facebook_main.
- Simulation: source_counts=[1,2,3], probabilities=[0.01,0.02,0.03],
  max_steps=3, distance_ranges=[{min:1,max:2},{min:3,max:5}].
- Observation: fractions=[1.0,0.75,0.5], false_positive_count=0,
  hide_source_count=0 (прежний default). Конечный candidate protocol сохраняется,
  включая допущение о доступности источников среди кандидатов.
- Dataset seed=4007026, единственный split independent_holdout=1998;
  min_candidates=5, max_infected_fraction=0.5, max_attempt_factor=100,
  distance_cache_size=512, show_progress=true. Порядок условий прежний:
  distance range, probability, fraction, k. 1998=37*54, k сбалансирован по schedule.
- Не менять существующие generate_dataset/_generate_split или фильтры генерации.
  Не добавлять отбор по раннему покрытию, предсказаниям или качеству модели.
- Все возможные attempt seeds нового набора лежат в [4007026,4406625]
  при существующем правиле simulation_seed=base+2*attempt,
  observation_seed=simulation_seed+1. До генерации проверить этот диапазон
  против объединения сохранённых simulation/observation seeds reference splits.
  После генерации проверить фактическую непересекаемость и уникальность.
  Любое пересечение — стоп; автоматически менять seed/генерировать другую
  выборку запрещено. Случайно совпавшие источники или исходы каскадов не удалять.
- Reference metadata: только simulation_seeds/observation_seeds из старых
  train/validation/test и sealed snapshot_final_holdout. Старые target arrays,
  распределения и метрики holdout до оценки не загружать.
- В Colab корень Drive: /content/drive/MyDrive/diffusion-sources.
  Данные: data/generated/facebook_temporal_v3_independent_holdout.
  Отчёты: reports/runs/temporal_v3_independent_evaluation/v1.
  Reference: data/facebook_main (путь действующего Colab notebook) и
  data/facebook_snapshot_final_holdout. Второй путь — явно проверяемый ожидаемый
  путь; его наличие на Drive пока не подтверждено. Если отсутствует, пользователь
  должен указать фактическое расположение sealed snapshot holdout.
  Frozen runs: reports/runs/facebook_main_v2/s1b/seed_7026 и
  frozen_candidate/seed_7027, frozen_candidate/seed_7028.
- Конфиг реализации: configs/facebook_temporal_v3_independent_holdout.yaml.
  Relative raw-graph path разрешается из проверенного repo root, не случайного cwd.
  Если Drive-пути не существуют, остановиться и запросить точный путь;
  нельзя молча подменить данные или создать новый snapshot holdout.
  Pilot: reports/runs/temporal_v3_paired_pilot/v1; baseline:
  reports/runs/temporal_v3_early_baseline/v1.

Генератор неизбежно создаёт source labels, но ни модель, ни аналитик не используют
новые target-метрики до разрешённой оценки. Seal читает только seeds/schema/file
hashes, без просмотра target распределений или early coverage. Generation summary
содержит прежние технические accepted/attempt/rejection counts, не F1.

## Фиксация входов и жизненный цикл

1. Preflight/freeze: аутентифицировать старые stage manifests/payload hashes,
   locked beta=0.5 и seeds, код historical pilot modules, graph/config/cache и
   все три best_model.pt/config.yaml. Сверить checkpoint identities с прежними
   отчётами; результаты из user-pasted JSON не заменяют проверку реальных файлов.
2. Generate/seal: проверить конфиг и seed metadata, один раз создать набор,
   зафиксировать hashes graph/archive/config/summary, unique seed isolation,
   count=1998 из schema и node mapping. Не считать F1 или source coverage.
3. Open/evaluate: только явное отдельное действие пользователя; до первого
   доступа evaluator к новым labels записать append-only opened marker со всеми
   identities. Оценить 7026, затем 7027, затем 7028 на том же наборе и ранних масках.
4. Summarize: проверить совпадение identities/index order/masks и полный состав
   трёх seed reports, сформировать итог без выбора нового победителя.

Freeze manifest содержит source code hashes (включая новый evaluator, генератор,
scoring/replay/statistics), git revision, runtime versions, raw graph и graph
fingerprints, configs/checkpoints/cache (либо отсутствие cache), старые selection
и pilot/baseline payload hashes, точный протокол и правила статистики. Seal
добавляет dataset file hashes. Никаких вымышленных hashes или fallback checkpoint.
Hash старого cache фиксируется, cache не перезаписывается; отсутствие означает
заранее зафиксированный режим детерминированного пересчёта только в памяти.

После interrupted generation/partial artifacts остановиться без overwrite.
После открытия interrupted inference допускает resume только для того же набора,
того же кода/параметров/checkpoints и identities; это продолжение одной оценки.
Не генерировать другую выборку для улучшения результата. Полный rerun считывает
готовые hash-checked stages, не изменяет их. Повреждение/несовпадение — явный стоп.
Новый output suffix не даёт разрешения на retuning после открытия.

## Методы, replay и запрет утечки

На всех 1998 примерах для каждого frozen seed:

1. Snapshot S1b: исходный predict_joint на финальном snapshot.
2. Early-first baseline: early наблюдённые финальные кандидаты раньше остальных;
   uniform ties внутри групп, аналитический expected F1 без learned source scores.
3. Temporal: sigmoid(logit)+0.5*early_observed; прежний tie-break по исходному
   score, затем ID. То же predicted-k, ограниченное размером candidate set.

Раннее наблюдение: после IC t=1, независимый Bernoulli с observation_fraction;
SeedSequence([5000000,observation_seed,1]), sorted nodes. Не выдавать t=0,
истинные infection times/source labels или oracle-k ранжированию. Пустые ранние
снимки сохраняются. Полный candidate set не сокращается. Labels используются
в replay consistency checks и evaluator, а не в source-blind sampling/scoring.

Replay должен совпасть по sources, infected mask, final observed/candidate mask
и условиям принятого примера; несовпадение не лечится пропуском. Early masks
одинаковы для всех трёх checkpoints. Beta=0 контроль идентичен snapshot.
Count и размер predicted set неизменны между тремя методами на каждом примере.
Индексы ровно range(1998), без лимита, дубликатов или изменения порядка.

Исторический collect_records остаётся train/validation-only; не снимать его
защиту и не маскировать новый split под validation. Новый guarded evaluator
получает только sealed independent_holdout и проверенный freeze manifest.
Исторические replay/scoring модули не переписываются, чтобы не разрушить
исторические code hashes. Старые Colab pilot cells не повторяются.

## Метрики, неопределённость и критерии до открытия

Primary comparison: temporal-minus-snapshot по mean per-cascade F1.
Secondary comparison: temporal-minus-early expected F1.
Публикуются все абсолютные значения, paired deltas, per-seed mean и sample SD.
Для каждой пары сначала усреднить delta по трём seeds для каждого cascade index,
затем парный bootstrap по каскадам, стратифицированный по true k:
2000 repetitions, RNG seed=9282026, percentile 95% CI [.025,.975]. Отдельно
сохранить per-seed CI. CI условен на этих checkpoints/графе/протоколе;
uniform baseline ties уже аналитически интегрированы. Seed SD не заменяет CI.

Подтверждение основного эффекта требует одновременно:

- mean temporal-minus-snapshot >=0.02;
- нижняя граница aggregate paired CI >0;
- temporal-minus-snapshot >0 на каждом training seed;
- средняя delta для k=2 и k=3 не ниже -0.02;
- count prediction неизменен на каждом примере.

Отдельный вывод о преимуществе learned ranking над выбранным ранним контролем
допускается после primary pass и при нижней aggregate CI temporal-minus-early >0
и положительной delta на каждом seed. Secondary threshold=0, без требования
повторить validation delta +0.089888. Это иерархическая проверка двух заявлений;
положительный secondary не спасает failed primary. Никаких дополнительных
выбранных post-hoc гипотез; subgroup результаты описательные без заявлений
о значимости. Если критерий не пройден, публиковать числа и ограниченный вывод,
не менять критерий, beta, seed или модель.

Для snapshot/temporal: precision/recall/F1, exact-set accuracy, count accuracy/MAE,
symmetric distance, Hit@1/2-hop; разрезы k=1/2/3, candidates 1–10/11–20/21–50/51+;
early source recall/all-source coverage/empty fraction. Для early-only только
expected F1 и те же F1 разрезы; выдуманная geometry для усреднённых ties запрещена.
All comparisons сохраняют synthetic observation assumptions; независимый набор
не устраняет известные ограничения задачи и не означает перенос на новый граф.

## Реализация и проверки, входящие в следующий план

Новые компоненты: protected generation/seal/freeze; guarded independent loader
и frozen inference; трёх-seed summary/statistics; отдельный русский notebook.
Переиспользовать существующие models, read-only features, replay, correct_sources,
analytic early F1 и append-only artifact primitives без изменения train pipeline.
Никаких новых product dependencies. Windows локально, Colab CUDA inference;
CPU допустим для unit/integration smoke, гарантированное время не обещается.

Проверить RED→GREEN: пути output/input overlap, существующие/partial outputs,
seed overlap/duplicates, неверные size/config/graph/checkpoint/beta/cache/code
hashes, отсутствие Drive inputs, старые artifacts неизменны, reads metadata-only
до открытия, labels недоступны evaluator до opened marker, exact replay,
beta=0, shared count, empty early masks, все индексы и три seeds, paired aggregate
CI без псевдорепликации, негативные gates, resume с mismatch и kernel imports
после git pull. Полный synthetic end-to-end pipeline — без реального holdout.
Реальный smoke использует только старые development train-примеры, не новые targets.

## Colab handoff и дальнейшее решение

Отдельные ячейки: setup; preflight/freeze; generate/seal; явный open и seed 7026;
resume seed 7027; seed 7028; summary. Direct kernel progress/JSON output,
каждая ошибка останавливает следующий этап. Не использовать Run all.
Между seal и open пользователь видит manifest и решает о запуске оценки.

После положительного подтверждения завершить этот цикл подбора и оформить
таблицы/графики/ограничения для диплома. Отрицательный результат тоже сохранить;
обсуждение нового development цикла не означает повторное использование набора
для выбора модели. Merge/push реализации требует отдельного разрешения.

## Gate перед реализацией

Пользователь одобрил этот письменный дизайн. Следующий этап — просмотр
implementation plan и выбор способа исполнения. До этих этапов продуктовый
код, новый датасет и evaluator не создаются. В этой сессии независимая оценка
не выполнялась и её результат не заявляется.
