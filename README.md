# Multi-source Diffusion Localization

Программная часть диплома по локализации неизвестного множества из 1-3 источников информационной диффузии. Система генерирует IC-каскады на графах, формирует неполные наблюдения, обучает GCN-модели и строит воспроизводимые таблицы и графики.

## Возможности

- многоисточниковые IC- и SI-симуляции;
- Joint Source-Count GCN с узловой и count-головами;
- Node-only GCN и классические baseline-методы;
- oracle-k и estimated-k оценка;
- абляции count-головы, consistency-loss и признаков;
- тесты неполноты, шума, смены процесса и скрытых источников;
- перенос на SNAP email-Eu-core без дообучения;
- Streamlit-демо;
- автоматическое сохранение метрик, checkpoint, CSV и PNG.

## Установка

Проект проверен с Python 3.14 и CPU-версией PyTorch.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.13.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
```

Для CUDA следует установить подходящий официальный wheel PyTorch до `pip install -e`.

## Проверка

```bash
.venv/bin/pytest --cov
.venv/bin/python -m compileall -q src scripts tests app.py
```

Тесты разделены на unit, integration и smoke. Минимальное требование покрытия проекта составляет 80%.

## Единый workflow

Быстрый профиль для проверки полного pipeline:

```bash
.venv/bin/python scripts/run_workflow.py \
  --config configs/workflow_smoke.yaml \
  --output reports/workflows/smoke
```

Пилотный профиль:

```bash
.venv/bin/python scripts/run_workflow.py \
  --config configs/workflow.yaml \
  --output reports/workflows/pilot
```

Workflow последовательно выполняет генерацию данных, обучение Joint и Node-only GCN, абляцию consistency-loss, baseline-оценку, robustness-тесты, смену IC на SI и построение отчетов. Итоговый `workflow_manifest.json` содержит пути и результаты всех этапов.

## Отдельные команды

```bash
# Загрузка основного SNAP-графа
.venv/bin/python scripts/download_data.py

# Facebook-пилот и его диагностика
.venv/bin/python scripts/generate_dataset.py \
  --config configs/facebook_pilot.yaml --output data/generated/facebook_pilot
.venv/bin/python scripts/analyze_dataset.py \
  --data data/generated/facebook_pilot --output reports/datasets/facebook_pilot

# Короткое профилирование GCN на полном Facebook-графе
.venv/bin/python scripts/train_model.py \
  --config configs/train_facebook_profile.yaml \
  --output reports/runs/facebook_profile

# Основной датасет (запускать после проверки свободного места)
.venv/bin/python scripts/generate_dataset.py \
  --config configs/facebook_main.yaml --output data/generated/facebook_main

# Генерация
.venv/bin/python scripts/generate_dataset.py \
  --config configs/pilot.yaml --output data/generated/pilot

# Joint Source-Count GCN
.venv/bin/python scripts/train_model.py \
  --config configs/train.yaml --output reports/runs/pilot

# Node-only GCN
.venv/bin/python scripts/train_node_model.py \
  --config configs/train_node.yaml --output reports/runs/node_pilot

# Baseline-методы
.venv/bin/python scripts/evaluate_baselines.py \
  --data data/generated/pilot --output reports/baselines/pilot

# Общий отчет
.venv/bin/python scripts/build_report.py \
  --run reports/runs/pilot \
  --node-run reports/runs/node_pilot \
  --baselines reports/baselines/pilot \
  --output reports/figures/pilot
```

## Streamlit

Локальное демо использует замороженный Temporal-v3/S1b из
`reports/backups/temporal_v3_20260929`. Это личная резервная копия, исключённая
из Git: её нужно восстановить в эту папку перед расчётом. Проверка файлов и
SHA256 выполняется автоматически. При другом расположении копии задайте
`DIFFUSION_TEMPORAL_BACKUP_DIR` — путь к корню с `backup_manifest.json` и
`local_paths.json`.

Из корня проекта, в активированном Python-окружении:

```bash
python -m pip install -e .
python -m streamlit run app.py
```

На Windows без активации окружения те же команды можно выполнить через
`.venv\Scripts\python.exe`. В интерфейсе задаются истинное число источников
симуляции (1–3), вероятность передачи (0.01/0.02/0.03), наблюдаемая доля и seed.
Кнопка «Рассчитать» создаёт воспроизводимый IC-каскад на фиксированном
Facebook-графе и сравнивает Snapshot с Temporal-v3. Модель сама оценивает
число источников; истинные метки ей не передаются. Это демонстрация одного
симулированного примера, не повторная оценка на holdout и не обучение.

## Google Colab

Для нового **exploratory** обучаемого ранжировщика поверх frozen Temporal-v3:
[пошаговый notebook](notebooks/colab_temporal_learned_reranker.ipynb) и
[статус, ограничения и пояснения метрик](docs/temporal_learned_reranker_status.md).
Для запуска по номерам ячеек: [инструкция Colab](docs/colab_temporal_learned_reranker.md).
Сначала ячейки 1–3 (код/пути/smoke), затем STOP до согласования cache/fit.
Принятая v3 не заменяется; test/holdout этот пилот не открывает.

Завершенные Facebook-эксперименты `v1_baseline` выполнялись на GPU через `notebooks/colab_training.ipynb`. Для последовательных validation-only абляций `snapshot-v2` создан отдельный `notebooks/colab_training_snapshot_v2.ipynb`; он не заменяет и не перезаписывает сценарий `v1`. Подробная инструкция, структура Drive и правила resume описаны в `docs/colab_training.md`. Все существующие training-конфигурации вне `configs/snapshot_v2/` считаются конфигурациями `v1` или его пилотов, если явно не указано иное.

## Материалы для диплома: независимая оценка Temporal-v3

- [Все материалы отчёта и инструкция](thesis_report/README.md).
- [Текст отчёта, таблицы и пояснения](thesis_report/text/temporal_v3_report.md).
- [Формулы: обычные обозначения и строки LaTeX для Word](thesis_report/text/temporal_v3_report_formulas.md).
- [Две схемы Mermaid с подписями](thesis_report/diagrams/temporal_v3_report_diagrams.md).
- [Python notebook с шестью графиками](thesis_report/notebooks/temporal_v3_report_figures.ipynb).

Notebook только читает готовый JSON и сохраняет PNG (300 dpi) / SVG.
Обучения и повторной оценки моделей нет; GPU не нужен. В Colab сначала
смонтируйте Drive по инструкции в первой текстовой ячейке, затем выполните
все ячейки notebook. Для локального запуска используются `numpy` и `matplotlib`
и архивный JSON из `thesis_report/data/`. Рисунки сохраняются в
`thesis_report/figures/` (в Colab — в одноимённую папку проекта на Drive).

В независимой проверке на 1998 новых IC-каскадах прежнего Facebook-графа
средний F1 по трём frozen checkpoints: Snapshot 0.344183, Early-only expected
0.434872, Temporal 0.523740. Это не перенос на другой граф и не доля полностью
правильных ответов; подробные определения и ограничения приведены в отчёте.

## Основные артефакты

Локальная точка возврата Temporal-v3 (веса, данные, итоговые результаты и архив кода):
`reports/backups/temporal_v3_20260929/`. Состав, проверка и ограничения описаны в
[технической инструкции](docs/temporal_v3_local_restore.md). Эта папка исключена
из Git; push не заменяет резервное копирование моделей.

- `data/generated/` - сгенерированные split и топологии;
- `reports/runs/` - checkpoint, история обучения и test-предсказания;
- `reports/figures/` - отчетные таблицы и графики;
- `reports/series/` - серии запусков по нескольким seed;
- `reports/robustness/` - неполнота и шум;
- `reports/process_shift/` - IC против SI;
- `reports/transfer/` - перенос на внешний граф;
- `reports/hidden_source/` - отдельная постановка со скрытым источником.

## Ограничения текущего состояния

Первоначальные небольшие пилоты проверяли pipeline. Для Temporal-v3 завершена
отдельная независимая оценка новых каскадов SNAP ego-Facebook по трём frozen
checkpoints (см. материалы выше). Она не подтверждает перенос на другие
топологии, качество на реальных процессах или превосходство над всеми
опубликованными методами; проверка заявляемой новизны по литературе остаётся
отдельной задачей.
