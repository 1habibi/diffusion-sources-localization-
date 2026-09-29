# Обучаемое ранжирование поверх frozen S1b: дешёвый exploratory пилот

Дата: 2026-09-29. Статус: пользователь одобрил письменный дизайн сообщением «го».
Implementation plan: `../plans/2026-09-29-temporal-learned-reranker.md`.
Пользователь одобрил Native/inline реализацию. Компоненты и notebook реализованы
в отдельной ветке; полный suite после единственного review/fix pass — 378 passed.
Три Important исправлены через RED→GREEN; Critical/Minor не было.
Финальный CPU smoke6 прошёл за 6.579 s; cache2160/fit/validation ещё не запускались.
Это технический рабочий документ, не чистовик диплома.

## 1. Цель и выбранный подход

Проверить, даёт ли маленький обучаемый ранжировщик заметное улучшение относительно
текущего Temporal-v3 без переобучения GCN и длинных гиперпараметрических поисков.
Пользователь хочет экономить время Colab, иметь точку возврата, а позднее подключить
результат к приложению. Улучшение не гарантируется.

Контроль — frozen S1b + раннее наблюдение t1=1 + бонус beta=0.5. Он независимо
проверен на новых каскадах той же топологии: mean F1=0.523740. Это историческая
метрика другого набора, не порог для прямого сравнения этого пилота.
Все новые разницы считаются парно на одних development-каскадах.

Выбран регуляризованный линейный LogisticRegression по признакам кандидата.
Альтернативы — переобучение temporal-GCN или совместный симуляционный поиск
множества источников — отложены как более дорогие/сложные.

Основные GCN checkpoint и count-head не изменяются. Ранжировщик заменяет правило
`GCN_probability + 0.5 * early_observed`; он не добавляется поверх него с новым
подбираемым beta. Фактический выход — top-predicted-k среди тех же кандидатов.
Модель классификации используется для ранжирования; калибровка вероятностей
источника и превосходство над специализированными ranking losses не заявляются.

## 2. Защищённые входы и границы

Локальная точка возврата: `reports/backups/temporal_v3_20260929/`.
Её checkpoint/config/data/manifest и все прежние результаты только читаются.
Зафиксированные run paths: seed 7026 — `runs/s1b/seed_7026`, seed 7027/7028 —
`runs/frozen_candidate/seed_7027` и `seed_7028`. Data — `data/reference`.
На Colab используются соответствующие исходные пути Drive; хеши должны совпасть
с сохранённым bundle, независимо от расположения и CPU/CUDA.

Разрешены только reference `graph.npz`, `config.yaml`, `train.npz`, а после
train-only gate — `validation.npz`. Новая утилита не получает пути к `test.npz`,
старому `final_holdout.npz` или открытому independent_holdout.
Никакие результаты independent evaluation не используются для выбора C,
признаков, scaler, коэффициентов, остановки или выбора seed.

Replay сохраняет существующую политику `independent-bernoulli-t1-seedsequence-v1`:
ранний шаг t1=1, IC max_steps=3, прежние вероятности/наблюдение/candidate masks.
Нет снимка t=0, истинных времён заражения или принудительного включения источников
в раннее наблюдение. Пустые ранние снимки и сложные каскады не отбрасываются.
Раннее и финальное наблюдения независимы; ранние узлы вне final candidate set
могут участвовать в наблюдаемом окружении, но сами не добавляются в кандидаты.

Старые графы/checkpoint/config/feature layout аутентифицируются SHA256.
Совместимость кода нового компонента фиксируется отдельными hashes/version;
старые frozen manifests не переписываются и не адаптируются под новые хеши.

## 3. Небольшая фиксированная выборка

Train содержит 9990 принятых каскадов, validation — 1998. Первые 2160 train
составляют 40 полных циклов 54 условий генерации. До извлечения признаков
проверяется фактический порядок и баланс сохранённых k/probability/fraction,
уникальность simulation/observation seed и отсутствие пересечения объединений
simulation+observation seeds между fit и dev. Ошибка — остановка, не фильтр.

- Fit: train indices 0..1619, 1620 каскадов (30 полных циклов).
- Selection-dev: train indices 1620..2159, 540 каскадов (10 полных циклов).
- Primary development evaluation: все 1998 validation, только после dev gate.
- Repeats: те же validation-примеры, frozen 7027/7028, только после primary gate.

Все кандидаты одного каскада принадлежат одной части. Нельзя делить строки узлов
случайным train_test_split: это смешает один каскад между fit и selection-dev.
Остальные train-каскады не подключаются автоматически при плохом результате.

Важно: сама frozen GCN ранее обучалась на исходном train, включая selection-dev,
а validation уже использовалась в разработке. Новый split отделяет обучение
ранжировщика от выбора его C, но не является независимой оценкой всей системы.
Все результаты пилота помечаются `exploratory=true`.

## 4. Входы ранжировщика: шесть фиксированных признаков

Чистая функция получает graph, final candidate IDs, вероятности frozen GCN,
early_observed mask и final_observed mask. У неё нет source_labels, true k,
infected mask, infection_times или simulation seed.

Для кандидата v признаки в неизменном порядке:

1. `gcn_probability`: sigmoid frozen source-logit.
2. `early_observed`: 0/1 принадлежность раннему наблюдению.
3. `early_neighbor_fraction`: доля соседей v, наблюдаемых рано.
4. `final_neighbor_fraction`: доля соседей v, наблюдаемых в финальном снимке.
5. `log_degree_normalized`: log(1+degree(v))/log(1+max_degree(graph)).
6. `gcn_probability_x_early`: произведение первого и второго признаков.

Для изолированного узла обе доли соседей равны 0; при max_degree=0 признак
степени равен 0. Neighbor fractions используют всех соседей графа, не только
final-кандидатов. Проверяются формы, уникальные ID, диапазоны и finite values.
Node ID нужен только для выравнивания и tie-break, не как обучаемый признак.
Глобальное знание топологии разрешено, как и в исходной S1b.

Отдельный training/evaluation интерфейс присоединяет бинарные labels к готовой
таблице признаков. При фиксированных наблюдениях и GCN scores смена меток не
должна менять признаки или предсказания. В replay labels доступны только для
проверки восстановления исходной симуляции, не скоринговой функции.

## 5. Обучение и выбор одной настройки

На seed 7026 один раз собирается fit/dev cache. StandardScaler обучается только
на fit candidate rows, без selection-dev или validation; сохраняются mean/scale
и порядок признаков. Нулевую дисперсию обрабатывать штатно, без удаления столбца.

Три кандидата LogisticRegression: C=[0.1, 1.0, 10.0], L2 regularization,
solver=lbfgs, fit_intercept=true, max_iter=1000, tol=1e-8. В установленной версии
scikit-learn использовать актуальный API для L2, зафиксировав effective params
и версию. ConvergenceWarning или non-finite coefficients — ошибка, не повод
автоматически расширить итерации/сетку. Дополнительного class_weight нет.

Вес строки на fit: 0.5/k для истинного источника и 0.5/(n_candidates-k) для
остальных кандидатов. Каждый каскад получает суммарный вес 1 и равные суммарные
веса классов. Все веса затем умножаются на (число fit rows / число fit cascades),
чтобы средний sample_weight был 1. Проверяются 0<k<n_candidates и finite weights;
в исходном наборе min_candidates=5, k<=3. Labels используются здесь как supervision,
но не передаются inference. Candidate cap или negative sampling не вводятся.

Inference использует линейный decision_function на стандартизованных признаках;
sigmoid не нужен для top-k. Tie-break: убывание learned score, затем исходной
GCN probability, затем возрастание node ID. k — прежний predicted_count,
размер выдачи min(k,n_candidates). Count logits и кандидатное множество прежние.

Выбрать единственный C по mean per-cascade F1 на selection-dev. При равенстве
с точностью 1e-12 предпочесть меньшее C. Все варианты имеют одну архитектуру.
После выбора не переобучать на fit+dev: сохранить именно выбранные fit-коэффициенты.
Одна модель/scaler, выбранные на 7026, затем без адаптации применяются к scores
7027/7028. Это проверка устойчивости одного корректора к трём frozen GCN,
не три независимых обучения корректора и не ансамбль.

## 6. Gates и условия остановки

Во всех gates delta = learned minus Temporal-v3(beta=0.5), а не minus snapshot.
Числа считаются по каскадам, не по candidate rows; float tolerance=1e-12.

Train-only dev gate выбранного C:

- mean delta F1 >= +0.02;
- mean exact-set accuracy delta >= 0;
- delta F1 в каждой группе истинного k=1/2/3 >= -0.02;
- count prediction и размер выдачи совпадают с v3 для каждого примера.

Этот gate — бюджетная проверка на данных выбора C; его прохождение не доказывает
обобщение. При failed gate сохранить таблицу трёх C, выбранный результат и причину;
validation и остальные seed не запускать. Без автоматического feature sweep.

Primary validation gate seed 7026:

- те же mean F1/exact-set/k-условия;
- delta F1 в каждом candidate-size bin 1–10/11–20/21–50/51+ >= -0.02;
- нижняя граница парного 95% bootstrap CI delta F1 > 0.

Для группового gate требуются все три k-группы; пустой candidate-size bin
помечается n=0/not_applicable и не включается в сравнение, но не скрывается.
Значения групп не округляются до проверки порогов.

Bootstrap: existing paired_bootstrap_ci, 2000 повторов, RNG seed 9282026,
стратификация true k. Dev CI не используется для выбора C.
До gate исходный snapshot validation F1 должен совпасть с saved run metrics
с допуском 1e-6. Контроль v3 вычисляется тем же кодом correct_sources(beta=.5)
на тех же cache rows; beta=0 обязан точно вернуть native snapshot sources.
Несовпадение snapshot-control останавливает оценку как ошибку воспроизводимости.

После primary pass проверить 7027/7028 без fit/selection повторно. Итоговый gate:
delta F1 положительна в каждом seed; mean delta >= .02; mean exact-set delta >=0;
mean delta F1 в каждой k/candidate-size группе >=-.02; все count-инварианты верны;
нижняя граница CI усреднённых per-cascade deltas >0. Для summary сначала усреднить
дельты трёх seed внутри каждого каскада, затем bootstrap 1998 строк, не 5994
независимых примера. Per-seed CI и sample SD публикуются отдельно.

Passed означает «имеет смысл отдельная независимая проверка», не замену принятого
v3 и не разрешение на старый/новый holdout. Новый sealed evaluation набор,
его бюджет и явное открытие согласовываются отдельно после обсуждения пилота.

## 7. Компоненты и сохранение результатов

Новый namespace `temporal_learned_*`: pure features/ranking, training/selection,
collector/cache и staged runner. Не менять existing snapshot-training CLI,
correct_sources, старые stage manifests, app.py или reference archives.
Переиспользовать replay_early_mask, frozen model construction, metrics,
paired_bootstrap_ci и append-only stage conventions.

Новый output root: `reports/runs/temporal_v3_learned_reranker/v1/`, отдельно
от backup и всех исторических runs. На Drive тот же относительный root проекта.
Перед любым write проверять resolved output/input overlap, включая parents.

Cache имеет schema version, split/indices/seed, ordered feature names,
checkpoint/config/graph/archive/source hashes, replay policy и runtime versions.
Flat candidate features/probabilities/IDs, ragged offsets, early flags,
native snapshot sources и predicted count отделены от supervision labels.
Хранить raw observable features, не PyG GPU-объекты и не целые графы на каждый узел.
Начальный dev cache — 2160 train каскадов 7026; validation/другие seed создаются
только при разрешённых gates. Ошибка hash/schema/shape/coverage — стоп.

Сохранять три C scores, выбранные коэффициенты/intercept/scaler как JSON/NPZ
без pickle, immutable selection manifest, per-cascade predictions,
все привычные метрики (F1, precision/recall, exact, count, distance, Hit@1/2),
разрезы k/candidate size, paired CI, gates и времена стадий.
Маленький JSON-head должен предсказывать идентично trained sklearn head на cache;
для этого требуется отдельный parity test. Коэффициенты не выдавать за причинные
эффекты или calibrated probabilities.

Повтор готовой стадии только при совпадающей identity и hashes. Стадии пишутся
атомарно с complete marker; partial output не принимается за успех и не
перезаписывается молча. Cache пересчёт запрещён при несовпавших входах: новая версия
output либо разбор ошибки. Labels/predictions остаются локальными/Drive-артефактами,
не автоматически попадают в Git или `thesis_report/`.

## 8. Бюджет и notebook

Отдельный пронумерованный notebook: setup/path check → smoke 6 train-примеров
без обучения → train cache 2160 → fit/select/dev gate → primary validation →
conditional repeats → summary. Он запускается локально на CPU или в Colab;
GPU ускоряет GCN inference, LogisticRegression обучается на CPU.
Полное обучение GCN и запуск всех трёх seed на train не предусмотрены.

Печатать начало/конец каждой стадии, входы, cache hit, выбранное устройство,
elapsed time и gate reason. Replay/inference имеют прогресс; smoke показывает
ориентировочную стоимость больших стадий без гарантии одинаковой скорости.
Бюджет каждой вычислительной стадии — 1800 секунд. Проверка времени между
каскадами и до/после каждого fit; превышение — остановка с partial/error,
не сокращённая успешная метрика. Это контроль бюджета на границах операций,
не гарантия принудительного прерывания отдельного solver fit ровно на 1800-й секунде.
Во время одного solver fit прогресс может отсутствовать; печатать старт каждого C,
число rows и длительность после fit, не скрывать ошибки.
Notebook не делает автоматический Run all за пользователя и не запускает обучение
до реализации, тестов, согласования плана и разрешения реального запуска.

## 9. Тесты и критерии готовности реализации

- Pure features: ручной маленький граф, изоляты, пустой early, early вне final,
  точные fractions/degree/interaction, permutation alignment и shape errors.
- Изменение target labels/true k при фиксированных observable inputs не меняет
  признаки или learned inference; GCN не получает source_labels/source_count.
- Group split и 54-condition balance, отсутствие overlap/duplicate seeds.
- Scaler и fit не читают dev/validation; select не читает validation/test/holdout.
- Cascade/class weights суммируются правильно, mean weight=1; float/nonfinite guards.
- Предсказания независимы от labels; native count/cardinality сохранены, стабильные
  tie-breaks, строгая загрузка frozen model, sklearn-vs-JSON scorer parity.
- C selection deterministic, no refit, shared corrector между 7026/7027/7028.
- Gate boundary cases, bootstrap pairing, per-cascade seed average, subgroup deltas.
- Hash tamper/partial stage/protected path guards; отсутствие перезаписи backup.
- Notebook valid Python/JSON, импорт в kernel, видимый output, false gate не
  открывает следующую стадию; timeout не выдаёт неполные данные за полный отчёт.
- Полный существующий test suite до реального пилота и интеграции.

Реализация проводится изолированно от сохранённого v3; старые tests и old
inference остаются совместимыми. Новые проверенные показатели и их пояснения
переносятся в чистовик лишь после решения по результатам; technical design,
неуспешные эксперименты и диагностика остаются в docs/reports/dev-log.

## 10. Следующий шаг

Пользователь просматривает implementation plan и выбирает способ исполнения.
До отдельного согласования плана не писать продуктовый код и не запускать обучение.
