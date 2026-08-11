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

Перед запуском должны существовать `data/generated/pilot` и `reports/runs/pilot`.

```bash
.venv/bin/streamlit run app.py
```

В приложении можно выбрать источники, вероятность передачи, число тактов, наблюдаемость, шум и random seed. Интерфейс показывает истинные и найденные источники, количество источников, метрики и тепловую карту scores.

## Основные артефакты

- `data/generated/` - сгенерированные split и топологии;
- `reports/runs/` - checkpoint, история обучения и test-предсказания;
- `reports/figures/` - отчетные таблицы и графики;
- `reports/series/` - серии запусков по нескольким seed;
- `reports/robustness/` - неполнота и шум;
- `reports/process_shift/` - IC против SI;
- `reports/transfer/` - перенос на внешний граф;
- `reports/hidden_source/` - отдельная постановка со скрытым источником.

## Ограничения текущего состояния

Текущие результаты получены на маленьком пилоте и предназначены для проверки pipeline. Итоговые научные выводы требуют обучения на SNAP ego-Facebook, не менее трех seed и проверки заявляемой новизны по ближайшим публикациям.
