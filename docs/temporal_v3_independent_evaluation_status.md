# Temporal-v3: независимая оценка

Дата: 2026-09-28. Реализация в `codex/temporal-independent-evaluation`.
Протокол утверждён пользователем; публикация и выполнение Colab отдельно.

## Что проверяется

1998 новых IC-каскадов на прежнем Facebook-графе, dataset seed 4007026.
Все frozen checkpoints S1b (7026, 7027, 7028), beta=0.5, t1=1.
Обучения, подбора beta, выбора лучшего seed и фильтрации нет.
Сравниваются snapshot, early-only expected F1 и temporal.
Это независимые каскады на известной топологии, не transfer/new graph.
Исходный candidate protocol предполагает наличие источников в candidate set.

## Рабочий процесс

Новый notebook: `notebooks/colab_temporal_v3_independent_evaluation.ipynb`.
Ячейки: setup → freeze → seal → explicit open+7026 → 7027 → 7028 → summary.
Не Run all. После freeze и seal остановиться и прислать вывод.
Открытие требует отдельного согласования и ввода `OPEN_INDEPENDENT_HOLDOUT`.
После открытия не менять входы/код/зависимости и не заменять набор.

Reference: `/content/drive/MyDrive/diffusion-sources/data/facebook_main`.
Snapshot seed metadata ожидается в `data/facebook_snapshot_final_holdout`.
Если путь отсутствует, уточнить реальное расположение, не создавать замену.
Старый test и snapshot holdout не используются для target-метрик.
Новые данные: `data/generated/facebook_temporal_v3_independent_holdout`.
Отчёты: `reports/runs/temporal_v3_independent_evaluation/v1`.

Raw-файл `REPO/data/raw/facebook_combined.txt.gz` не входит в Git.
Перед freeze notebook при необходимости получает только исходный граф через
существующий downloader из фиксированного SNAP-источника; существующий raw
не заменяется. После рестарта проверяется raw SHA256 сохранённого freeze.
При недоступности источника/несовпадении hash остановиться и восстановить
свою исходную копию по выведенному пути. Это не генерация каскадов.

Freeze проверяет hashes checkpoints/configs, исторических pilot/baseline
артефактов, графа, raw edges, cache либо отсутствия cache, кода и runtime
(Python, NumPy, PyTorch, PyG, NetworkX, SciPy, PyYAML). Новый topology hash
использует канонические пары min/max labeled node ID; исторические
feature/cache fingerprints не изменены.
Seal читает только seeds и schema headers. Все stages append-only;
частичная генерация или сохранение останавливают retry без перезаписи.
Открытие записывается до первого evaluator target read. Обрыв после него
не отменяет факт открытия. Возобновление проверяет актуальные hashes.

## Статистика и интерпретация

Сначала усредняются парные per-cascade deltas трёх seeds, затем bootstrap
1998 каскадов со strata true-k: 2000 повторов, seed 9282026, 95% percentile CI.
5994 независимых примера не заявляются; seeds не являются ансамблем.
CI условен на графе/генераторе/трёх checkpoints. Разрезы описательные.

Primary: mean Δ temporal−snapshot ≥0.02, aggregate CI lower >0,
положительная mean Δ каждого seed, mean k=2/3 Δ ≥−0.02,
count неизменен в каждом примере. Secondary только после primary:
aggregate CI temporal−early lower >0 и положительная Δ каждого seed.
Прежний validation прирост +0.089888 не является новым порогом.
При отрицательном результате сохраняется отчёт, не начинается retuning.

Метрики snapshot/temporal: precision, recall, F1, exact-set accuracy,
count accuracy/MAE, symmetric distance, Hit@1/2-hop и разрезы k/candidate size.
Early-only использует общий GCN count и аналитическое ожидание uniform ties;
расстояние и hop-метрики для него не приписываются.

## Реальное выполнение

Новые Drive-данные не генерировались и не открывались в ходе реализации.
Независимых результатов пока нет. Старые source modules и pipeline неизменны.
Локальные тесты используют синтетические архивы, реальные hashes/persistence,
scoring/bootstrap и отдельную проверку настоящей GCN/PyG; expensive generator
и replay/inference заменяются только на явных тестовых границах.

Финальная локальная проверка: `pytest -q` — 285 passed, 2 прежних предупреждения
`torch.jit.script`, 341.69s. `compileall -q src scripts` и `git diff --check`
прошли. Независимый whole-branch review нашёл три Important: ориентация рёбер
в raw/archive fingerprint, отсутствие raw в свежем clone, неполный runtime
identity. Все три исправлены с RED→GREEN регрессиями (42 targeted tests)
и полным зелёным прогоном 285 тестов. Critical/Minor замечаний нет.
2026-09-29 по разрешению пользователя ветка опубликована на GitHub (PR #1),
реализация `ac4c9d7` интегрирована fast-forward в локальный master.
Повторная проверка объединённого master: 285 passed, 2 прежних предупреждения,
345.99s; compileall и diff-check прошли. Публикация master — следующий шаг.

Drive availability/совместимость артефактов проверяется настоящими freeze/seal
выводами пользователя, не синтетическими fixture. GPU memory/time и качество
неизвестны до реального запуска. Manifest обеспечивает локальную целостность
в доверенном окружении, не защиту от умышленной подделки всех файлов/hashes.

## Полученный независимый результат — 2026-09-29

Этот раздел обновляет состояние выше: master опубликован (`ef56aee`, PR #1
merged), пользователь завершил freeze/seal, explicit open, три seed и summary.
Первоначальный источник — присланные Colab outputs. При подготовке отчёта
29.09.2026 оригинальный summary/payload.json скачан с Drive; SHA-256
980e20c405503cde35ff8f89db934989c994390fd3d203db73b15ac303328030
совпал с summary manifest. Копия: thesis_report/data/independent_summary.json.
Per-cascade CSV/предсказания при оформлении независимо не пересчитывались.

Все 1998 новых IC-каскадов оценены каждым checkpoint без обучения/retuning.
Оба заранее заданных gates прошли; count_unchanged=true.

| Метод | Mean F1 | Sample SD по seeds |
|---|---:|---:|
| Snapshot | 0.344183 | 0.003478 |
| Early-only expected | 0.434872 | 0.004696 |
| Temporal | 0.523740 | 0.001694 |

Temporal−snapshot: +0.179557, paired bootstrap 95% CI [0.166871, 0.192049].
Temporal−early: +0.088868, CI [0.078527, 0.098869]. Оба контраста положительны
на всех seeds. Bootstrap сначала усредняет три deltas для каждого каскада,
затем resamples 1998 каскадов со strata true-k; не 5994 независимых примера.

Exact-set: 6.96%→22.79%; Hit@1-hop: 51.47%→82.10%; symmetric distance:
1.156726→0.636887. Count accuracy неизменна: 70.69%.
Temporal F1 по k=1/2/3: 0.406990/0.545045/0.619186; при 51+ кандидатах
0.358826. Разрезы описательные, subgroup significance не заявляется.

Dataset hash: `b79df4b7bdc2966baaee3a92bf0eea361026362808c3f4587a3d4921e4455950`.
Freeze hash: `bb3468af0d0dbc0b9b5f0e37c502bcfda98495f0894290ec1e728b440d0f0e86`.
Summary: `reports/runs/temporal_v3_independent_evaluation/v1/summary/payload.json`.

Это подтверждение на новых каскадах прежнего графа/генератора, не перенос на
новую топологию. Snapshot не имеет раннего наблюдения; early-only контроль
использует общую GCN count-head и uniform ties, не все возможные эвристики.
Набор после открытия не заменять, параметры по нему не подбирать.
Новых запусков и публикации этой документальной записи не выполнялось.
